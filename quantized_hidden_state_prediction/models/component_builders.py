from typing import Optional, Dict

from torch import nn
from torchtune.models.llama3._model_utils import scale_hidden_dim_for_mlp
from torchtune.models.llama3_1._position_embeddings import Llama3ScaledRoPE
from torchtune.modules.tied_linear import TiedLinear

from torchtune.modules import (
    RMSNorm,
    TransformerDecoder,
    TransformerSelfAttentionLayer,
    MultiHeadAttention,
    FeedForward,
    FrozenNF4Linear,
)

from modules.architectures import (
    FusedMultiHeadAttention,
    TransformerLayer,
    TransformerDecoder,
    TransformerDecoderPHi,
    swiglu_mlp,
    LSTMLayer,
    LSTMPHi,
)
from modules.self_prediction import (
    PHiLayer,
    PHiMLP,
)

from modules.quantizers import (
    quantizer_module
)

from torchtune.modules.common_utils import reparametrize_as_dtype_state_dict_post_hook

import wandb

VALID_ATTN_LAYERS = {
    "FusedMultiHeadAttention": FusedMultiHeadAttention,
    "Identity": nn.Identity
}
VALID_MLP_LAYERS = {
    "swiglu_mlp": swiglu_mlp,
    "Identity": nn.Identity,
}


def llama3_mlp(dim: int,
               hidden_dim: int,
               output_dim: Optional[int] = None,
               quantize_base: bool = False) -> FeedForward:
    """
    Factory function to create a Llama 3-style MLP (SwiGLU).

    This function constructs the feed-forward network used in Llama 3, which is
    a Swish-Gated Linear Unit (SwiGLU). It initializes the gate, down, and up
    projection layers and returns them encapsulated in a `FeedForward` module.

    Args:
        dim (int): The input and default output dimension of the MLP.
        hidden_dim (int): The intermediate hidden dimension.
        output_dim (Optional[int], optional): The final output dimension. If None,
            defaults to `dim`. Defaults to None.
        quantize_base (bool, optional): If True, uses quantized linear layers
            (`FrozenNF4Linear`) instead of standard `nn.Linear`. Defaults to False.

    Returns:
        FeedForward: A `FeedForward` module configured with the Llama 3 MLP layers.
    """
    if output_dim is None:
        output_dim = dim
    gate_proj = nn.Linear(dim, hidden_dim, bias=False) if not quantize_base else FrozenNF4Linear(dim, hidden_dim, bias=False)
    down_proj = nn.Linear(hidden_dim, output_dim, bias=False) if not quantize_base else FrozenNF4Linear(hidden_dim, output_dim, bias=False)
    up_proj = nn.Linear(dim, hidden_dim, bias=False) if not quantize_base else FrozenNF4Linear(dim, hidden_dim, bias=False)
    return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj)


def self_prediction_mlp(dim: int,
                        hidden_dim: int,
                        output_dim: Optional[int] = None,
                        num_layers: int = 2) -> PHiMLP:
    """
    Factory function to create a `PHiMLP` for the self-prediction module.

    This function is a simple wrapper that constructs a `PHiMLP` instance. This
    type of MLP is used as a building block within the PHi-Layer, for example
    as the prior predictor or the posterior's encoder network.

    Args:
        dim (int): The input and default output dimension.
        hidden_dim (int): The intermediate hidden dimension.
        output_dim (Optional[int], optional): The final output dimension. If None,
            defaults to `dim`. Defaults to None.
        num_layers (int, optional): The number of layers for the `PHiMLP`, which
            determines its architecture (e.g., linear, SwiGLU, or deep residual).
            Defaults to 2.

    Returns:
        PHiMLP: An instance of the configured `PHiMLP`.
    """
    if output_dim is None:
        output_dim = dim
    return PHiMLP(input_dim=dim,
                  hidden_dim=hidden_dim,
                  output_dim=output_dim,
                  num_layers=num_layers)

def lstm_phi(
    vocab_size: int,
    num_layers: int,
    embed_dim: int,
    max_seq_len: int,
    num_heads: int = 6,
    attn_dropout: float = 0.0,
    rope_base: int = 500000,
    intermediate_dim: Optional[int] = None,
    norm_eps: float = 1e-5,
    scale_factor: int = 8,
    tied_embeddings: bool = True,
    self_critic_loss_factor: float = 0.0,
    phi_loss_factor: float = 1.0,
    use_self_attention: bool = True,
    detach_hidden_states: bool = False,
    detach_targets: bool = False,
    use_self_prediction: bool = True,
    self_prediction_layer_position: Optional[int] = None,
    self_prediction_num_layers: int = 2,
    chance_to_deterministic: float = 0.0,
    deterministic_at_inference: bool = False,
    straight_through_eval: bool = False,
    use_information_bottleneck: bool = True,
    use_hidden_state_prediction: bool = True,
) -> LSTMPHi:
    """
    Build the decoder associated with the Idsia model. This includes:
    - Token embeddings
    - num_layers number of TransformerSelfAttentionLayer blocks
    - RMS Norm layer applied to the output of the transformer
    - Final projection into token space

    Args:
        vocab_size (int): number of tokens in vocabulary.
        num_layers (int): number of layers in the transformer decoder.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. User should ensure
            `num_heads` % `num_kv_heads` == 0. For standard MHA set `num_kv_heads` == `num_heads`,
            for GQA `num_kv_heads` < `num_heads`, and for MQA set `num_kv_heads` == 1.
        embed_dim (int): embedding dimension for self-attention
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        intermediate_dim (Optional[int]): intermediate dimension for MLP. If not specified,
            this is computed using :func:`~torchtune.modules.scale_hidden_dim_for_mlp`
        norm_eps (float): epsilon in RMS norms.
        scale_factor (int): scaling factor for RoPE. Default: 8

        attn_postnorm (bool): whether to apply norm after attn, before output proj. Default: False
        attn_gating (bool): whether to apply gating right before output proj. Default: False
        attn_postnorm_kind (str): kind of norm to apply after attn. Default: "nn.LayerNorm"
        beta (bool): whether to apply beta (ex. deltanet, etc). Default: False
        conv_size (int): size of conv layer after K, Q (like in mamba). Use -1 to ignore. Default: -1
        attn_layer_type (str): type of attn layer. Use "" or "identity" to skip. Default: ""
        mlp_layer_type (str): type of mlp layer. Use "" or "identity" to skip. Default: ""
        tied_embeddings (bool): whether to use tied embeddings. Qwen uses for <7B, others use <1B, ... Default: False
        kl_div_loss_factor (float): factor for KL divergence loss. If 0., no KL loss is initialized. Default: 0.0
        self_critic_loss_factor (float): factor for self critic loss. If 0., no self critic loss is initialized. Default: 0.0
        previous_hidden_loss_factor (float): factor for loss for predicting the previous hidden state (of the last
            layer). If 0., no previous hidden state prediction head is initialized. Default: 0.0
        next_hidden_loss_factor (float): factor for loss for predicting the next hidden state (of the last layer).
            If 0., no next hidden state prediction head is initialized. Default: 0.0
        self_prediction_layer_position (Optional[int]): position of the self prediction layer. If None, the layer is
            added at the end of the decoder. Default: None
        self_prediction_loss (str): type of loss to use for self prediction, with the options "mse" and "nll".
            Default: "mse"

    Returns:
        TransformerDecoder: Instantiation of Llama3.1 model.
    """

    head_dim = embed_dim // num_heads
    num_kv_heads = num_heads
    rope = Llama3ScaledRoPE(dim=head_dim, max_seq_len=max_seq_len, base=rope_base, scale_factor=scale_factor)
    hidden_dim = intermediate_dim if intermediate_dim else scale_hidden_dim_for_mlp(embed_dim)

    layers = []
    for _ in range(num_layers):
        layer = LSTMLayer(
            size=embed_dim,
            norm=RMSNorm(embed_dim, eps=norm_eps),
        )
        layers.append(layer)
    layers = nn.ModuleList(layers)

    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    nn.init.uniform_(tok_embeddings.weight, a=-1e-4, b=1e-4)
    output_proj = nn.Linear(embed_dim, vocab_size, bias=False) if not tied_embeddings else TiedLinear(tok_embeddings)

    # Conditionally build the PHi self-prediction layer
    self_prediction_layer = None
    if use_self_prediction:
        prior_attention = None
        if use_self_attention and phi_loss_factor > 0.0:
            prior_attention = MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
                k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
                v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
                output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
                pos_embeddings=rope,
                max_seq_len=max_seq_len,
                attn_dropout=attn_dropout,
            )

        self_prediction_layer = PHiLayer(
            d_model=embed_dim,
            posterior_mlp=nn.Linear(embed_dim, 2 * embed_dim, bias=False),
            decoder_mlp=nn.Linear(embed_dim, embed_dim, bias=False),
            prior_prediction_mlp=self_prediction_mlp(dim=embed_dim,
                                                     hidden_dim=hidden_dim,
                                                     output_dim=2 * embed_dim,
                                                     num_layers=self_prediction_num_layers),
            prior_prediction_attention=prior_attention,
            sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps) if prior_attention else None,
            self_critic_loss_factor=self_critic_loss_factor,
            next_loss_factor=phi_loss_factor,
            detach_hidden_states=detach_hidden_states,
            detach_targets=detach_targets,
            chance_to_deterministic=chance_to_deterministic,
            deterministic_at_inference=deterministic_at_inference,
            straight_through_eval=straight_through_eval,
            use_hidden_state_prediction=use_hidden_state_prediction,
            use_information_bottleneck=use_information_bottleneck,
        )

    return LSTMPHi(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
        tied_embeddings=tied_embeddings,
        output_hidden_states=None,
        self_prediction_layer=self_prediction_layer,
        self_prediction_layer_position=self_prediction_layer_position,
    )


def llama3_phi(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    rope_base: int = 500000,
    intermediate_dim: Optional[int] = None,
    norm_eps: float = 1e-5,
    scale_factor: int = 8,
    tied_embeddings: bool = True,
    self_critic_loss_factor: float = 0.0,
    phi_loss_factor: float = 1.0,
    use_self_attention: bool = True,
    detach_hidden_states: bool = False,
    detach_targets: bool = False,
    use_self_prediction: bool = True,
    self_prediction_layer_position: Optional[int] = None,
    self_prediction_num_layers: int = 2,
    chance_to_deterministic: float = 0.0,
    deterministic_at_inference: bool = False,
    straight_through_eval: bool = False,
    use_information_bottleneck: bool = True,
    use_hidden_state_prediction: bool = True,
    self_prediction_information_bottleneck : Optional[str] = 'continuous',
    self_prediction_module: Optional[Dict] = None,
    latent_loss_factor: float = None,
) -> TransformerDecoderPHi:
    """
    Factory function to build a Llama 3-style Transformer model integrated
    with a PHi (Prediction of Hidden states) layer.

    Args:
        vocab_size (int): The size of the vocabulary.
        num_layers (int): The number of Transformer decoder layers.
        num_heads (int): The number of attention heads.
        num_kv_heads (int): The number of key/value heads for Grouped Query Attention.
        embed_dim (int): The embedding dimension.
        max_seq_len (int): The maximum sequence length.
        attn_dropout (float, optional): Dropout rate for attention. Defaults to 0.0.
        rope_base (int, optional): The base frequency for RoPE. Defaults to 500000.
        intermediate_dim (Optional[int], optional): The intermediate dimension for the MLP.
            If None, it's calculated based on `embed_dim`. Defaults to None.
        norm_eps (float, optional): Epsilon for RMSNorm. Defaults to 1e-5.
        scale_factor (int, optional): Scale factor for RoPE. Defaults to 8.
        tied_embeddings (bool, optional): Whether to tie input and output embeddings. Defaults to True.
        self_critic_loss_factor (float, optional): Weight for the self-critic loss to
            prevent posterior collapse in the PHi layer. Defaults to 0.0.
        phi_loss_factor (float, optional): Weight for the PHi loss itself. Defaults to 1.0.
        use_self_attention (bool, optional): Whether to use self-attention in the PHi
            layer's autoregressive prior. Defaults to True.
        detach_hidden_states (bool, optional): If True, detaches hidden states before the
            PHi layer. Defaults to False.
        detach_targets (bool, optional): If True, detaches the posterior targets for the
            PHi loss. Defaults to False.
        use_self_prediction (bool, optional): Master switch to enable the PHi layer.
            Defaults to True.
        self_prediction_layer_position (Optional[int], optional): The layer index *after which*
            the PHi layer is inserted. Defaults to the final layer.
        self_prediction_num_layers (int, optional): Number of layers in the PHi prior's MLP.
            Defaults to 2.
        chance_to_deterministic (float, optional): Probability of using deterministic sampling
            in the PHi layer during training. Defaults to 0.0.
        deterministic_at_inference (bool, optional): If True, always use deterministic
            sampling during inference. Defaults to False.
        straight_through_eval (bool, optional): If True, bypass the PHi bottleneck during
            evaluation. Defaults to False.
        use_information_bottleneck (bool, optional): If True, enables the variational
            information bottleneck in the PHi layer. Defaults to True.
        use_hidden_state_prediction (bool, optional): If True, enables the self-prediction
            mechanism in the PHi layer. Defaults to True.

    Returns:
        TransformerDecoderPHi: An instance of the configured model.
    """
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    rope = Llama3ScaledRoPE(dim=head_dim, max_seq_len=max_seq_len, base=rope_base, scale_factor=scale_factor)
    hidden_dim = intermediate_dim if intermediate_dim else scale_hidden_dim_for_mlp(embed_dim)

    print(self_prediction_module)

    # Build the stack of Transformer layers
    layers = []
    for _ in range(num_layers):
        self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
        )
        mlp = llama3_mlp(dim=embed_dim, hidden_dim=hidden_dim)
        layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
            mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        )
        layers.append(layer)
    layers = nn.ModuleList(layers)

    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    nn.init.uniform_(tok_embeddings.weight, a=-1e-4, b=1e-4)
    output_proj = nn.Linear(embed_dim, vocab_size, bias=False) if not tied_embeddings else TiedLinear(tok_embeddings)
    sa_norm = RMSNorm(dim=embed_dim, eps=norm_eps)
    prior_prediction_mlp = self_prediction_mlp(dim=embed_dim, 
                                                   hidden_dim=hidden_dim,
                                                   output_dim=[self_prediction_module["codebook_dim"] if self_prediction_information_bottleneck != 'continuous' else 2*embed_dim][0],
                                                   num_layers=self_prediction_num_layers)

    # Conditionally build the PHi self-prediction layer
    self_prediction_layer = None
    if use_self_prediction:
        prior_attention = None

        posterior_mlp, quantizer_mlp, decoder_mlp = quantizer_module(self_prediction_information_bottleneck, 
                                                                     self_prediction_module,
                                                                     embed_dim,)

        # TODO: remove the dynamic assigning of embedding dim in prior attn, do it upstream
        if use_self_attention and phi_loss_factor > 0.0:
            # print('embed_dim: ',[embed_dim if quantizer_module is None else self_prediction_module.codeword_dim][0] )
            # TODO: number of priors is equal to the number of quantizers
            prior_attention = MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
                k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
                v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
                output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
                pos_embeddings=rope,
                max_seq_len=max_seq_len,
                attn_dropout=attn_dropout,
            )
            sa_norm = RMSNorm(dim=embed_dim, eps=norm_eps)

            if self_prediction_information_bottleneck == 'residual_quantize':
                num_quantizers = self_prediction_module["num_quantizers"]
                prior_prediction_mlp = nn.ModuleList([self_prediction_mlp(dim=embed_dim, 
                                                   hidden_dim=hidden_dim,
                                                   output_dim=[self_prediction_module["codebook_dim"] if self_prediction_information_bottleneck != 'continuous' else 2*embed_dim][0],
                                                   num_layers=self_prediction_num_layers) for _ in range(num_quantizers)])

        self_prediction_layer = PHiLayer(
            d_model=embed_dim,
            posterior_mlp=posterior_mlp,
            quantizer_mlp=quantizer_mlp,
            decoder_mlp=decoder_mlp,
            prior_prediction_mlp=prior_prediction_mlp,
            prior_prediction_attention=prior_attention,
            sa_norm=sa_norm if prior_attention else None,
            self_critic_loss_factor=self_critic_loss_factor,
            next_loss_factor=phi_loss_factor,
            detach_hidden_states=detach_hidden_states,
            detach_targets=detach_targets,
            chance_to_deterministic=chance_to_deterministic,
            deterministic_at_inference=deterministic_at_inference,
            straight_through_eval=straight_through_eval,
            use_hidden_state_prediction=use_hidden_state_prediction,
            use_information_bottleneck=use_information_bottleneck,
            reconstruction_loss_factor=self_prediction_module["reconstruction_loss_factor"] if self_prediction_module is not None else None,
            latent_loss_factor=latent_loss_factor if latent_loss_factor is not None else self_prediction_module["latent_loss_factor"] if self_prediction_module is not None else 0.0,
        )

        # wandb.watch(self_prediction_layer, log='all') # watching phi model

    # Assemble the final model
    return TransformerDecoderPHi(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
        tied_embeddings=tied_embeddings,
        output_hidden_states=None,
        self_prediction_layer=self_prediction_layer,
        self_prediction_layer_position=self_prediction_layer_position,
    )
