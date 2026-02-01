import logging

from typing import Callable, Optional, Union

import numpy as np
import math

import torch.nn
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from torchtune.modules.attention_utils import _MaskType, _sdpa_or_flex_attention
from torchtune.modules.feed_forward import FeedForward
from torchtune.modules.kv_cache import KVCache
from torchtune.modules.transformer import _get_clones
from modules.self_prediction import PHiLossCollector


logger = logging.getLogger(__name__)


def swiglu_mlp(dim: int) -> FeedForward:
    """
    Build the MLP layer associated with the Llama model.
    """
    hidden_dim = dim * 8 // 3
    gate_proj = nn.Linear(dim, hidden_dim, bias=False)
    down_proj = nn.Linear(hidden_dim, dim, bias=False)
    up_proj = nn.Linear(dim, hidden_dim, bias=False)
    return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj)


class BaseAttentionPlaceholder(torch.nn.Module):
    """Placeholder class to template-implement cache logic."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.kv_cache = None

    def common_init(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        pos_embeddings: Optional[torch.nn.Module] = None,
        q_norm: Optional[nn.Module | bool] = None,
        k_norm: Optional[nn.Module | bool] = None,
        kv_cache: Optional[KVCache] = None,
        max_seq_len: int = 4096,
        is_causal: bool = True,
        attn_dropout: float = 0.0,
        beta: bool = False,
        betabias: bool = False,
        attn_gating: bool = False,
        attn_postnorm: bool = False,
        conv_size: int = -1,
        attn_postnorm_kind: str = "nn.LayerNorm",
        tied_embeddings: bool = False,
        **kwargs,
    ):
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        # Set attributes
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.is_causal = is_causal

        # optional specific attributes
        self.beta = beta
        self.betabias = betabias
        self.attn_gating = attn_gating
        self.attn_postnorm = attn_postnorm
        self.conv_size = conv_size
        self.attn_postnorm_kind = attn_postnorm_kind
        self.tied_embeddings = tied_embeddings
        self.q_norm = q_norm
        self.k_norm = k_norm

        # Set layers
        self.kv_cache = kv_cache
        self.pos_embeddings = pos_embeddings

        self.q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        if self.beta:
            self.beta_proj = nn.Linear(embed_dim, num_heads, bias=False)
        self.output_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # most optional stuff
        if self.attn_gating:
            self.gating_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        if self.attn_postnorm:
            self.attn_post_norm = eval(self.attn_postnorm_kind)(
                embed_dim, elementwise_affine=False, bias=False
            )

        if self.conv_size > 1:
            from fla.modules.convolution import ShortConvolution

            self.conv_q = ShortConvolution(
                num_heads * head_dim,
                kernel_size=self.conv_size,
                activation=None,
                use_fast_conv1d=False,
            )
            self.conv_k = ShortConvolution(
                num_heads * head_dim,
                kernel_size=self.conv_size,
                activation=None,
                use_fast_conv1d=False,
            )
            self.conv_v = ShortConvolution(
                num_heads * head_dim,
                kernel_size=self.conv_size,
                activation=None,
                use_fast_conv1d=False,
            )

        if self.betabias:
            self.beta_bias = nn.Parameter(torch.zeros(num_heads) + 3**0.5)

        # Use flex attention if supported and we are sample packing
        self._attention_call = _sdpa_or_flex_attention()

    def setup_cache(self, batch_size: int, dtype: torch.dtype, max_seq_len: int) -> None:
        """Setup key value caches for attention calculation. If called
        after kv_cache is already setup, this will be skipped.
        """
        # Don't overwrite user defined kv_cache from init
        if self.kv_cache is not None:
            logger.warning(
                "Key value caches are already setup. You cannot call ``setup_caches()`` twice. Skipping."
            )
        else:
            self.kv_cache = True

    def reset_cache(self):
        """Reset the key value caches."""
        return


class FusedMultiHeadAttention(BaseAttentionPlaceholder):
    """
    Similar to torchtune.modules.attention.MultiHeadAttention
    but with fused projection layers.
    """
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        pos_embeddings: Optional[torch.nn.Module] = None,
        q_norm: Optional[nn.Module] = None,
        k_norm: Optional[nn.Module] = None,
        kv_cache: Optional[KVCache] = None,
        max_seq_len: int = 4096,
        is_causal: bool = True,
        attn_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()

        if attn_dropout < 0 or attn_dropout > 1:
            raise ValueError(f"attn_dropout ({embed_dim}) must be between 0.0 and 1.0")

        if bool(q_norm) ^ bool(k_norm):
            raise ValueError("q and k norm must be set together")

        self.common_init(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            pos_embeddings=pos_embeddings,
            q_norm=q_norm,
            k_norm=k_norm,
            kv_cache=kv_cache,
            max_seq_len=max_seq_len,
            is_causal=is_causal,
            attn_dropout=attn_dropout,
            **kwargs,
        )
        self._attention_call = _sdpa_or_flex_attention()

    def setup_cache(self, batch_size: int, dtype: torch.dtype, max_seq_len: int) -> None:
        """Setup key value caches for attention calculation. If called
        after kv_cache is already setup, this will be skipped.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            max_seq_len (int): maximum sequence length model will be run with.
        """
        # Don't overwrite user defined kv_cache from init
        if self.kv_cache is not None:
            logger.warning(
                "Key value caches are already setup. You cannot call ``setup_caches()`` twice. Skipping."
            )
        else:
            self.kv_cache = KVCache(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dtype=dtype,
            )

    def reset_cache(self):
        """Reset the key value caches."""
        if self.kv_cache is None:
            raise RuntimeError(
                "Key value caches are not setup. Call ``setup_caches()`` first."
            )
        self.kv_cache.reset()

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        *,
        mask: Optional[_MaskType] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """See: torchtune.modules.attention.MultiHeadAttention.forward
        Only difference is we use QKV projections by default.

        Notation used for tensor shapes:
            - b: batch size
            - s_x: sequence length for x
            - s_y: sequence length for y
            - n_h: num heads
            - n_kv: num kv heads
            - d: embed dim
            - h_d: head dim
        """
        # x has shape [b, s_x, d]
        # y has shape [b, s_y, d]
        b, s_x, d_x = x.shape
        s_y = y.shape[1] if y is not None else 0

        # q has shape [b, s_x, num_heads * head_dim]
        q = self.q_proj(x)

        # number of queries per key/value
        q_per_kv = self.num_heads // self.num_kv_heads
        q = q.view(b, s_x, self.num_kv_heads * q_per_kv, self.head_dim)

        # Apply positional embeddings
        if self.pos_embeddings is not None:
            q = self.pos_embeddings(q, input_pos=input_pos)

        # [b, n_h, s_x, h_d]
        q = q.transpose(1, 2)

        # Normalize q
        if self.q_norm:
            q = self.q_norm(q)

        if y is None:
            if self.kv_cache is None:
                raise ValueError(
                    "Must provide y input or use kv_cache to enable streaming decoding"
                )
            k = self.kv_cache.k_cache
            v = self.kv_cache.v_cache
        else:
            # Update k and v shape, positional embeddings, and normalization

            # k has shape [b, s_y, num_kv_heads * head_dim]
            # v has shape [b, s_y, num_kv_heads * head_dim]
            k = self.k_proj(y)
            v = self.v_proj(y)

            # Apply positional embeddings
            # k: [b, s_y, n_kv, h_d]
            k = k.view(b, s_y, -1, self.head_dim)
            if self.pos_embeddings is not None:
                k = self.pos_embeddings(k, input_pos=input_pos)

            # View + expand + reshape bring num_kv_heads to num_heads for k and v
            # to match q.

            # k: [b, s_y, n_kv, 1, h_d]
            # v: [b, s_y, n_kv, 1, h_d]
            k = k.view(b, s_y, self.num_kv_heads, 1, self.head_dim)
            v = v.view(b, s_y, self.num_kv_heads, 1, self.head_dim)

            # If needed, expand the key and value tensors to have the same shape
            # as the query tensor by copying values across the relevant dim
            if self.num_heads != self.num_kv_heads:
                k = k.expand(b, s_y, self.num_kv_heads, q_per_kv, self.head_dim)
                v = v.expand(b, s_y, self.num_kv_heads, q_per_kv, self.head_dim)

            if self.conv_size > 1:
                k = self.conv_q(k)
                v = self.conv_q(v)

            # [b, s, n_h, h_d]
            k = k.reshape(b, s_y, -1, self.head_dim)
            v = v.reshape(b, s_y, -1, self.head_dim)

            # [b, n_h, s, h_d]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Normalize k
            if self.k_norm:
                k = self.k_norm(k)

            # Update key-value cache
            if self.kv_cache is not None:
                k, v = self.kv_cache.update(k, v)

        output = self._attention_call(
            q,
            k,
            v,
            mask=mask,
            dropout_p=self.attn_dropout,
            is_causal=self.kv_cache is None and mask is None and self.is_causal,
        )

        # reshape the output to be the same shape as the input
        output = output.transpose(1, 2).contiguous().view(b, s_x, -1)

        if self.attn_postnorm:
            output = self.attn_post_norm(output)

        if self.attn_gating:
            output = F.silu(self.gating_proj(x)) * output

        return self.output_proj(output)


class TransformerLayer(nn.Module):
    """
    Transformer layer derived from the Llama2 model. Normalization is applied before the attention **and** FF layer.

    Args:
        attn (nn.Module): Attention module.
        mlp (nn.Module): Feed-forward module.
        sa_norm (Optional[nn.Module]): Normalization to be applied before self-attention.
        mlp_norm (Optional[nn.Module]): Normalization to be applied before the feed-forward layer.
        sa_scale (Optional[nn.Module]): Module to scale self-attention output.
        mlp_scale (Optional[nn.Module]): Module to scale the feed-forward output.
    """

    def __init__(
        self,
        attn: nn.Module,
        mlp: nn.Module,
        *,
        sa_norm: Optional[nn.Module] = None,
        mlp_norm: Optional[nn.Module] = None,
        sa_scale: Optional[nn.Module] = None,
        mlp_scale: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.attn = attn
        self.mlp = mlp
        self.sa_norm = sa_norm
        self.mlp_norm = mlp_norm
        self.sa_scale = sa_scale or nn.Identity()
        self.mlp_scale = mlp_scale or nn.Identity()

    def setup_cache(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: int,
        decoder_max_seq_len: int,
    ) -> None:
        """Setup key value caches for attention calculation.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            encoder_max_seq_len (int): this parameter is ignored in this layer.
            decoder_max_seq_len (int): maximum cache sequence length.
        """
        encoder_max_seq_len = 0 * encoder_max_seq_len
        self.attn.setup_cache(batch_size, dtype, max_seq_len=decoder_max_seq_len)

    @property
    def cache_enabled(self) -> bool:
        """Check if the key value caches are setup."""
        return self.attn.kv_cache is not None

    def reset_cache(self):
        """Reset the key value caches."""
        self.attn.reset_cache()

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: Optional[_MaskType] = None,
        input_pos: Optional[torch.Tensor] = None,
        **kwargs: dict,
    ) -> torch.Tensor:
        """See torchtune.modules.transformer.TransformerSelfAttentionLayer.forward"""
        # Input tensor and attention output have the same shape
        # [b, s, d]
        # Norm applied before self-attention
        h = self.sa_norm(x)
        if not isinstance(self.attn, torch.nn.Identity):
            attn_out = self.attn(h, h, mask=mask, input_pos=input_pos)

            # Residual connection; shape: [batch_size, seq_length, embed_dim]
            h = self.sa_scale(attn_out) + x

        # Norm applied before the feedforward layer
        mlp_out = self.mlp_norm(h)
        if not isinstance(self.mlp, torch.nn.Identity):
            mlp_out = self.mlp_scale(self.mlp(mlp_out))

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        out = h + mlp_out
        return out


class TransformerDecoder(nn.Module):
    """Transformer decoder module, based on the Llama 3.2 architecture."""
    def __init__(
        self,
        *,
        tok_embeddings: torch.nn.Embedding,
        layers: Union[nn.Module, list[nn.Module], nn.ModuleList],
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        norm: nn.Module,
        output: Union[nn.Linear, Callable],
        num_layers: Optional[int] = None,
        output_hidden_states: Optional[list[int]] = None,
        tied_embeddings: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(layers, nn.ModuleList):
            pass
        elif isinstance(layers, list):
            layers = nn.ModuleList(layers)
        else:
            if not isinstance(layers, nn.Module):
                raise AssertionError("num_layers is defined, layers must be a module")
            if num_layers is None:
                raise AssertionError("num_layers is not defined, layers must be a list")
            layers = _get_clones(layers, num_layers)

        self.tok_embeddings = tok_embeddings
        self.layers = layers
        self.norm = norm
        self.output = output
        self.output_hidden_states = output_hidden_states or []
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.causal_mask = None
        self.num_output_chunks = 0
        self.tied_embeddings = tied_embeddings

        # attributes for KV caches during inference
        self.encoder_max_cache_seq_len = None
        self.decoder_max_cache_seq_len = None

    def set_num_output_chunks(self, num_output_chunks: int) -> None:
        """Used to save memory in combination with :class:`~torchtune.modules.loss.CEWithChunkedOutputLoss`.
        This should be called before the first forward pass, in the recipe."""
        self.num_output_chunks = num_output_chunks

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: Optional[int] = None,
        decoder_max_seq_len: Optional[int] = None,
    ):
        """torchtune.modules.transformer.TransformerDecoder.setup_caches"""
        if decoder_max_seq_len is not None:
            self.decoder_max_cache_seq_len = decoder_max_seq_len
        else:
            self.decoder_max_cache_seq_len = self.max_seq_len

        
        for layer in self.layers:
            layer.setup_caches(
            batch_size,
            dtype,
            encoder_max_seq_len=self.encoder_max_cache_seq_len,
            decoder_max_seq_len=self.decoder_max_cache_seq_len,
            )
            # if hasattr(layer, "_checkpoint_wrapped_module"):
            #     layer._checkpoint_wrapped_module.setup_cache(
            #         batch_size,
            #         dtype,
            #         encoder_max_seq_len=self.encoder_max_cache_seq_len,
            #         decoder_max_seq_len=self.decoder_max_cache_seq_len,
            #     )
            # else:
            #     layer.setup_cache(
            #         batch_size,
            #         dtype,
            #         encoder_max_seq_len=self.encoder_max_cache_seq_len,
            #         decoder_max_seq_len=self.decoder_max_cache_seq_len,
            #     )

    def caches_are_setup(self) -> bool:
        return self.encoder_max_cache_seq_len is not None

    def encoder_caches_are_enabled(self) -> bool:
        """Checks if there are any :class:`~torchtune.modules.TransformerCrossAttentionLayer`,
        or :class:`~torchtune.modules.fusion.FusionLayer` layers which have cache enabled.
        """
        return self.encoder_max_cache_seq_len is not None

    def decoder_caches_are_enabled(self) -> bool:
        """Check if there are any :class:`~torchtune.modules.TransformerCrossAttentionLayer`
        layers which have cache enabled."""
        return self.decoder_max_cache_seq_len is not None

    def caches_are_enabled(self) -> bool:
        return self.decoder_caches_are_enabled()

    def reset_caches(self):
        """Reset the key value caches."""
        if not (self.encoder_caches_are_enabled() or self.decoder_caches_are_enabled()):
            raise RuntimeError(
                "Key value caches are not setup. Call ``setup_caches()`` first."
            )

        for layer in self.layers:
            layer.reset_cache()

    @torch.compiler.disable
    def chunked_output(self, last_hidden_state: torch.Tensor) -> list[torch.Tensor]:
        """torchtune.modules.transformer.TransformerDecoder.chunked_output"""
        return [
            self.output(chunk)
            for chunk in last_hidden_state.chunk(self.num_output_chunks, dim=1)
        ]

    def _validate_inputs(
        self,
        seq_len: int,
        mask: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ):
        """torchtune.modules.transformer.TransformerDecoder._validate_inputs"""

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) of input tensor should be smaller "
                f"than max_seq_len ({self.max_seq_len})"
            )

        if self.decoder_caches_are_enabled():
            if mask is None:
                raise ValueError(
                    "KV-caches for self-attention layers are setup for inference mode, causal masks must be provided!"
                    " Use the `mask` arg to provide a causal mask."
                )

        if self.encoder_caches_are_enabled():
            if encoder_mask is None:
                raise ValueError(
                    "KV-caches for cross-attention/fusion layers are setup for inference mode, causal masks must be provided!"
                    " Use the `encoder_mask` arg to provide a causal mask."
                )

        if (
            self.encoder_caches_are_enabled() or self.decoder_caches_are_enabled()
        ) and input_pos is None:
            raise ValueError(
                "KV-caches are setup for inference mode, input positions must be provided!"
            )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        mask: Optional[_MaskType] = None,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """See torchtune.modules.transformer.TransformerDecoder.forward

        Shape notation:
            - b: batch size
            - s: token sequence length
            - s_e: encoder sequence length
            - v: vocab size
            - d: token embed dim
            - d_e: encoder embed dim
            - m_s: max seq len
        """
        # input tensor of shape [b, s]
        bsz, seq_len = tokens.shape

        self._validate_inputs(
            seq_len, mask=mask, encoder_mask=encoder_mask, input_pos=input_pos
        )

        # shape: [b, s, d]
        h = self.tok_embeddings(tokens)

        hidden = []
        for i, layer in enumerate(self.layers):
            if i in self.output_hidden_states:
                hidden.append(h)
            # shape: [b, s, d]
            h = layer(
                h,
                mask=mask,
                encoder_input=encoder_input,
                encoder_mask=encoder_mask,
                input_pos=input_pos,
            )

        # shape: [b, s, d]
        h = self.norm(h)

        if self.num_output_chunks > 0:
            output = self.chunked_output(h)
        else:
            # shape: [b, seq_len, out_dim]
            output = self.output(h).float()

        # Output list if hidden states are requested, otherwise just the output
        output = output if not hidden else [*hidden, output]
        return output


class TransformerDecoderPHi(TransformerDecoder):
    """
    A Transformer decoder, based on the Llama 3.2 architecture, extended with a
    PHi (Prediction of Hidden states) layer.

    This class inherits from `TransformerDecoder` and integrates a `PHiLayer` at a
    specified position within the decoder stack. During the forward pass, after the
    target layer has processed its input, the resulting hidden state is passed
    through the PHi layer. This computes the PHi loss and other metrics, which are
    collected for use in the main training loss. The new hidden state from the PHi
    layer is then passed to the subsequent decoder layers.

    Args:
        *args: Positional arguments passed to the parent `TransformerDecoder`.
        self_prediction_layer (Optional[nn.Module], optional): An instance of the
            `PHiLayer` to be inserted into the model. Defaults to None.
        self_prediction_layer_position (Optional[int], optional): The index of the
            decoder layer *after which* the PHi layer should be applied. If None,
            it defaults to being placed after the last layer. Defaults to None.
        pad_token_id (int, optional): The ID of the padding token, required for masking
            in the PHi layer. Defaults to 0.
        **kwargs: Keyword arguments passed to the parent `TransformerDecoder`.
    """
    def __init__(
        self,
        *args,
        self_prediction_layer: Optional[nn.Module] = None,
        self_prediction_layer_position: Optional[int] = None,
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.self_prediction_layer = self_prediction_layer
        self.pad_token_id = pad_token_id
        self.self_prediction_losses = PHiLossCollector()
        if self_prediction_layer_position is None:
            self_prediction_layer_position = len(self.layers) - 1
        self.self_prediction_layer_position = self_prediction_layer_position

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        mask: Optional[_MaskType] = None,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """
        Defines the forward pass, integrating the PHi layer.

        The data flows through the decoder layers as usual. After the layer at
        `self_prediction_layer_position`, the hidden state is passed to the
        PHi layer. The PHi layer returns a new hidden state, which continues
        through the rest of the stack, and also produces losses that are
        collected by this class.

        Args:
            tokens (torch.Tensor): Input token IDs with shape `[batch_size, seq_len]`.
            mask (Optional[_MaskType], optional): The attention mask. Defaults to None.
            input_pos (Optional[torch.Tensor], optional): Positional embeddings. Defaults to None.
            **kwargs: Additional keyword arguments for extensibility.

        Returns:
            Union[torch.Tensor, list[torch.Tensor]]: The final output logits from the model,
                optionally preceded by requested intermediate hidden states.
        """

        # Shape notation:
        #     - b: batch size
        #     - s: token sequence length
        #     - s_e: encoder sequence length
        #     - v: vocab size
        #     - d: token embed dim
        #     - d_e: encoder embed dim
        #     - m_s: max seq len

        # input tensor of shape [b, s]
        bsz, seq_len = tokens.shape

        self._validate_inputs(
            seq_len, mask=mask, encoder_mask=encoder_mask, input_pos=input_pos
        )
        padding_mask = tokens == self.pad_token_id

        # shape: [b, s, d]
        h = self.tok_embeddings(tokens)

        hidden = []
        for i, layer in enumerate(self.layers):
            if i in self.output_hidden_states:
                hidden.append(h)

            h = layer(
                h,
                mask=mask,
                encoder_input=encoder_input,
                encoder_mask=encoder_mask,
                input_pos=input_pos,
            )

            if i == self.self_prediction_layer_position and self.self_prediction_layer is not None:
                self_prediction_dict = self.self_prediction_layer(
                    h, padding_mask, mask=mask, input_pos=input_pos
                )
                h = self_prediction_dict["h"]

                for key, value in self_prediction_dict.items():
                    if key == "h":
                        continue
                    if value.numel() < 1:
                        continue
                    self.self_prediction_losses.add_loss(key.replace('_', ' '), value)

        # shape: [b, s, d]
        # last_hidden = self.norm(h)

        if self.num_output_chunks > 0:
            output = self.chunked_output(h)
        else:
            # shape: [b, seq_len, out_dim]
            output = self.output(h).float()

        # Output list if hidden states are requested, otherwise just the output
        output = output if not hidden else [*hidden, output]
        return output

    def get_additional_losses(self) -> dict:
        """
        Retrieves and resets the losses collected from the PHi layer.

        This method separates the collected losses into two groups:
        1.  Training Losses: Scalar losses intended to be added to the main training objective.
        2.  Logging Losses: Token-wise metrics averaged for logging and analysis.

        After retrieval, the internal loss collector is reset.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: A tuple containing
                (training_losses, logging_losses).
        """
        losses = self.self_prediction_losses.losses
        train_losses = {k: v for k, v in losses.items() if "tokenwise" not in k}
        logging_losses = {
            k: (v * (v != 0.0)).sum() / (v != 0.0).sum()
            for k, v in losses.items()
            if "tokenwise" in k
        }
        self.self_prediction_losses.reset()
        return train_losses, logging_losses

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: Optional[int] = None,
        decoder_max_seq_len: Optional[int] = None,
    ):
        """Extends parent method to also set up the PHi layer's cache."""
        super().setup_caches(
            batch_size,
            dtype,
            encoder_max_seq_len=encoder_max_seq_len,
            decoder_max_seq_len=decoder_max_seq_len,
        )
        if self.self_prediction_layer is not None:
            self.self_prediction_layer.setup_cache(
                batch_size, dtype, max_seq_len=decoder_max_seq_len
            )

    def reset_caches(self):
        """Extends parent method to also reset the PHi layer's cache."""
        super().reset_caches()
        if self.self_prediction_layer is not None:
            self.self_prediction_layer.reset_cache()


class LSTMLayer(nn.Module):
    """
    A single layer for an LSTM network, featuring a residual connection.

    This module wraps a standard `nn.LSTM` module. In the forward pass, it first
    applies an optional normalization to the input, then processes the result
    through the LSTM. Finally, it adds the pre-normalized input to the LSTM's
    output, creating a residual (or skip) connection.

    Args:
        size (int): The dimensionality of the input, hidden, and output features.
        norm (Optional[nn.Module], optional): An optional normalization layer
            to be applied to the input. Defaults to None.
    """
    def __init__(
        self,
        size: int,
        *,
        norm: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(size, size, batch_first=True)
        self.norm = norm

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: dict,
    ) -> torch.Tensor:
        """
        Defines the forward pass of the LSTMLayer.

        Args:
            x (torch.Tensor): The input tensor of shape `[batch_size, seq_length, size]`.
            **kwargs (dict): Accepts additional keyword arguments for API compatibility,
                but they are not used.

        Returns:
            torch.Tensor: The output tensor after the LSTM and residual connection.
        """
        h = self.norm(x)
        lstm_h, lstm_c = self.lstm(h)

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        out = h + lstm_h
        return out


class LSTMPHi(nn.Module):
    """
    An LSTM-based sequence model integrated with a PHi (Prediction of Hidden states) layer.

    This class defines a complete autoregressive model using a stack of `LSTMLayer`
    modules. It serves as an RNN-based alternative to the Transformer architecture
    for studying self-prediction.

    A `PHiLayer` is inserted at a specified position within the stack of LSTM layers.
    During the forward pass, this layer computes the PHi loss by creating an
    information bottleneck on the hidden state and measuring the model's ability to
    predict it. The resulting losses are collected for use in the main training objective.

    Args:
        tok_embeddings (th.nn.Embedding): The token embedding layer.
        layers (Union[nn.Module, list, nn.ModuleList]): A single `LSTMLayer` to be cloned,
            or a pre-defined list of layers.
        max_seq_len (int): The maximum sequence length the model can handle.
        norm (nn.Module): The final normalization layer applied after the LSTM stack.
        output (Union[nn.Linear, Callable]): The final output projection layer.
        num_layers (Optional[int]): The number of LSTM layers. Required if `layers` is
            a single module to be cloned.
        output_hidden_states (Optional[list[int]]): A list of layer indices from which
            to output hidden states. Defaults to None.
        tied_embeddings (bool): Whether input and output embeddings are tied. Defaults to False.
        self_prediction_layer (Optional[nn.Module]): An instance of `PHiLayer` to be
            inserted into the model.
        self_prediction_layer_position (Optional[int]): The index of the layer *after which*
            the PHi layer is applied. Defaults to the last layer.
        pad_token_id (int): The ID of the padding token for masking. Defaults to 0.
    """

    def __init__(
        self,
        *,
        tok_embeddings: torch.nn.Embedding,
        layers: Union[nn.Module, list[nn.Module], nn.ModuleList],
        max_seq_len: int,
        norm: nn.Module,
        output: Union[nn.Linear, Callable],
        num_layers: Optional[int] = None,
        output_hidden_states: Optional[list[int]] = None,
        tied_embeddings: bool = False,
        self_prediction_layer: Optional[nn.Module] = None,
        self_prediction_layer_position: Optional[int] = None,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        if isinstance(layers, nn.ModuleList):
            pass
        elif isinstance(layers, list):
            layers = nn.ModuleList(layers)
        else:
            if not isinstance(layers, nn.Module):
                raise AssertionError("num_layers is defined, layers must be a module")
            if num_layers is None:
                raise AssertionError("num_layers is not defined, layers must be a list")
            layers = _get_clones(layers, num_layers)

        self.tok_embeddings = tok_embeddings
        self.layers = layers
        self.norm = norm
        self.output = output
        self.output_hidden_states = output_hidden_states or []
        self.max_seq_len = max_seq_len
        self.num_output_chunks = 0
        self.tied_embeddings = tied_embeddings

        self.self_prediction_layer = self_prediction_layer
        self.pad_token_id = pad_token_id
        self.self_prediction_losses = PHiLossCollector()
        if self_prediction_layer_position is None:
            self_prediction_layer_position = len(self.layers) - 1
        self.self_prediction_layer_position = self_prediction_layer_position

    def set_num_output_chunks(self, num_output_chunks: int) -> None:
        """Used to save memory in combination with :class:`~torchtune.modules.loss.CEWithChunkedOutputLoss`.
        This should be called before the first forward pass, in the recipe."""
        self.num_output_chunks = num_output_chunks

    @torch.compiler.disable
    def chunked_output(self, last_hidden_state: torch.Tensor) -> list[torch.Tensor]:
        """torchtune.modules.transformer.TransformerDecoder.chunked_output"""
        return [
            self.output(chunk)
            for chunk in last_hidden_state.chunk(self.num_output_chunks, dim=1)
        ]

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        mask: Optional[_MaskType] = None,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """See torchtune.modules.transformer.TransformerDecoder.forward

        Shape notation:
            - b: batch size
            - s: token sequence length
            - s_e: encoder sequence length
            - v: vocab size
            - d: token embed dim
            - d_e: encoder embed dim
            - m_s: max seq len
        """
        # input tensor of shape [b, s]
        bsz, seq_len = tokens.shape

        padding_mask = tokens == self.pad_token_id

        # shape: [b, s, d]
        h = self.tok_embeddings(tokens)

        hidden = []
        for i, layer in enumerate(self.layers):
            if i in self.output_hidden_states:
                hidden.append(h)
            # shape: [b, s, d]
            h = layer(h)

            if i == self.self_prediction_layer_position and self.self_prediction_layer is not None:
                self_prediction_dict = self.self_prediction_layer(h, padding_mask)
                h = self_prediction_dict["h"]

                for key, value in self_prediction_dict.items():
                    if key == "h":
                        continue
                    if value.numel() < 1:
                        continue
                    self.self_prediction_losses.add_loss(key.replace('_', ' '), value)

        # shape: [b, s, d]
        h = self.norm(h)

        if self.num_output_chunks > 0:
            output = self.chunked_output(h)
        else:
            # shape: [b, seq_len, out_dim]
            output = self.output(h).float()

        # Output list if hidden states are requested, otherwise just the output
        output = output if not hidden else [*hidden, output]
        return output

    def get_additional_losses(self) -> dict:
        losses = self.self_prediction_losses.losses
        train_losses = {k: v for k, v in losses.items() if "tokenwise" not in k}
        logging_losses = {
            k: (v * (v != 0.0)).sum() / (v != 0.0).sum()
            for k, v in losses.items()
            if "tokenwise" in k
        }
        self.self_prediction_losses.reset()
        return train_losses, logging_losses


def get_constant_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Creates a learning rate scheduler with a linear warmup followed by a constant learning rate.

    This scheduler increases the learning rate linearly from 0 to the optimizer's
    initial LR over the course of `num_warmup_steps`. After the warmup period,
    the learning rate is held constant at the initial LR for the remainder of training.

    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        num_warmup_steps (int): The number of steps for the linear warmup phase.
        num_training_steps (int): The total number of training steps. Note: this
            argument is not used in this specific scheduler but is often included
            for API consistency with other schedulers.
        last_epoch (int, optional): The index of the last epoch when resuming
            training. Defaults to -1.

    Returns:
        LambdaLR: A PyTorch learning rate scheduler instance.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def sequential_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_decay_steps: int,
    decay_lr_upto: float,
    num_training_steps: int,
    last_epoch=-1,
) -> LambdaLR:
    max_lr = optimizer.param_groups[0]['lr']
    decay_lower_bound = decay_lr_upto / max_lr
    def lr_lambda(current_step: int) -> float:
        # Phase 1: Warmup (0 to num_warmup_steps)
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        
        # Phase 2: Cosine decay (num_warmup_steps to num_warmup_steps + num_decay_steps)
        elif current_step < num_warmup_steps + num_decay_steps:
            progress = (current_step - num_warmup_steps) / num_decay_steps
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return decay_lower_bound + (1.0 - decay_lower_bound) * cosine_decay
        
        # Phase 3: Constant (after num_warmup_steps + num_decay_steps)
        else:
            return decay_lower_bound    
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def cos_anneal(e0, e1, t0, t1, e):
    """ ramp from (e0, t0) -> (e1, t1) through a cosine schedule based on e \in [e0, e1] """
    alpha = max(0, min(1, (e - e0) / (e1 - e0))) # what fraction of the way through are we
    alpha = 1.0 - math.cos(alpha * math.pi/2) # warp through cosine
    t = alpha * t1 + (1 - alpha) * t0 # interpolate accordingly
    return t

class temperature_scheduler:
    # The relaxation temperature τ is annealed from 1 to 1/16 over the first 150,000 updates.
    def __init__(self, global_steps: int, temp_start: float, temp_end: float):
        self.global_steps = global_steps
        self.temp_start = float(temp_start)
        self.temp_end = float(temp_end)

    def __call__(self, curr_step: int):
        t = cos_anneal(0, self.global_steps, self.temp_start, self.temp_end, curr_step)
        return t

class beta_scheduler:
    def __init__(self, saturation_steps : int = 5000, beta_max : float = 8.5e-3):
    # The KL weight β is increased from 0 to 6.6 over the first 5000 updates
    # "We divide the overall loss by 256 × 256 × 3, so that the weight of the KL term
    # becomes β/192, where β is the KL weight."
    # TODO: OpenAI uses 6.6/192 but kinda tricky to do the conversion here... about 5e-4 works for this repo so far... :\
        self.saturation_steps = saturation_steps
        self.beta_max = float(beta_max)
    
    def __call__(self, curr_step: int):
        t = cos_anneal(0, self.saturation_steps, 0.0, self.beta_max, curr_step)
        return t