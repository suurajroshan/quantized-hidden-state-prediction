from functools import reduce
from typing import Callable, Optional, Union, Dict, Any, List

import numpy as np

import torch as th
import torch.nn
import torch.nn.functional as F
from torch import nn
from torch.nn import ModuleList

from torchtune.modules.attention_utils import _MaskType, _sdpa_or_flex_attention
from torchtune.modules.feed_forward import FeedForward
from torchtune.modules.kv_cache import KVCache
from torchtune.modules.transformer import _get_clones

import wandb
from einops import repeat, rearrange


def gaussian_kl(pred_q, pred_logvar, target_mu, target_logvar):
    # Calculate the variance ratio and the squared difference of means
    var_ratio = torch.exp(target_logvar - pred_logvar)
    t = (target_mu - pred_q).pow(2) / torch.exp(pred_logvar)

    # KL divergence formula, summed over features and averaged over the batch
    kl_div = 0.5 * (pred_logvar - target_logvar + var_ratio + t - 1)

    return torch.sum(kl_div, dim=-1)


def gaussian_entropy(log_var):
    # Determine the dimensionality D from the last dimension of the input tensor
    if len(log_var.shape) > 1:
        dim = log_var.shape[-1]
    else:
        dim = log_var.shape[0]

    # The formula combines the constant part with the sum of log-variances
    # We sum over the last dimension (the feature dimension)
    return 0.5 * (dim * (1.0 + np.log(2 * np.pi)) + torch.sum(log_var, dim=-1))


class PHiMLP(nn.Module):
    """
    A flexible Multi-Layer Perceptron (MLP) with SwiGLU and residual connections.

    This module is a building block for the PHi layer, suitable for use in the
    prior predictor or decoder. Its architecture is determined
    by the `num_layers` parameter:
    - `num_layers = 1`: The MLP is a simple linear transformation.
    - `num_layers = 2`: The MLP uses a standard SwiGLU (Swish-Gated Linear Unit) block.
    - `num_layers > 2`: The MLP becomes a deep network of SwiGLU blocks with
      residual skip connections between them.

    Args:
        input_dim (int): The dimension of the input features.
        hidden_dim (int): The dimension of the hidden layers.
        output_dim (int): The dimension of the output features.
        num_layers (int): The number of layers, which dictates the architecture.
        activation (nn.Module, optional): The activation function to use within the
            SwiGLU blocks. Defaults to nn.SiLU().
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int,
                 activation: nn.Module = nn.SiLU()):
        super().__init__()
        if num_layers == 1:
            hidden_dim = input_dim
        self.gate_layers = nn.ModuleList()
        self.projection_layers = nn.ModuleList()
        current_input_dim = input_dim
        for l in range(1, num_layers):
            self.gate_layers.append(nn.Linear(current_input_dim, hidden_dim))
            self.projection_layers.append(nn.Linear(current_input_dim, hidden_dim))
            current_input_dim = hidden_dim
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = activation

    def forward(self, x):
        residual = 0
        for gate_layer, projection_layer in zip(self.gate_layers, self.projection_layers):
            gate = self.activation(gate_layer(x))
            proj = projection_layer(x)
            x = residual + gate * proj
            residual = x
        return self.output_layer(x)


class PHiLayer(torch.nn.Module):
    """
    Implements the PHi (Prediction of Hidden states) layer.

    This module measures the complexity of a sequence model's computation
    by creating an information bottleneck on its hidden states and calculating
    a loss based on the model's ability to predict its own future states.

    Args:
        d_model (int): The dimensionality of the hidden states.
        posterior_mlp (torch.nn.Module): The network that encodes the hidden state
            into the parameters of the posterior distribution `q`.
        decoder_mlp (torch.nn.Module): The network that decodes the latent variable `z`
            back into a hidden state representation.
        prior_prediction_mlp (torch.nn.Module): The MLP part of the autoregressive
            prior model `p`.
        prior_prediction_attention (Optional[torch.nn.Module], optional): The attention
            part of the autoregressive prior model `p`. Defaults to None.
        sa_norm (Optional[nn.Module], optional): Normalization layer applied before the
            prior prediction MLP. Defaults to nn.Identity().
        self_critic_loss_factor (float, optional): Weighting factor for the self-critic
            loss, used to prevent posterior collapse.  Defaults to 0.1.
        next_loss_factor (float, optional): Weighting factor for the PHi loss. Defaults to 0.1.
        detach_hidden_states (bool, optional): If True, detaches the incoming hidden
            states from the computation graph. Defaults to False.
        detach_targets (bool, optional): If True, detaches the target distributions
            (the posterior) during PHi loss calculation. Defaults to False.
        full_information_blockage (bool, optional): If True, forces the latent
            variable `z` to have zero information, for ablation. Defaults to False.
        chance_to_deterministic (float, optional): Probability of making the sampling
            of `z` deterministic during training. Defaults to 0.0.
        deterministic_at_inference (bool, optional): If True, uses the mean of the
            posterior instead of sampling during inference. Defaults to False.
        straight_through_eval (bool, optional): If True, passes the original hidden
            state `h` through the layer during evaluation, bypassing the bottleneck.
            Defaults to False.
        use_information_bottleneck (bool, optional): If True, enables the variational
            information bottleneck. Defaults to True.
        use_hidden_state_prediction (bool, optional): If True, enables the self-prediction
            mechanism. Defaults to True.
    """
    def __init__(
        self,
        d_model: int,
        posterior_mlp: torch.nn.Module,
        quantizer_mlp: torch.nn.Module,
        decoder_mlp: torch.nn.Module,
        prior_prediction_mlp: torch.nn.Module,
        prior_prediction_attention: torch.nn.Module | List[nn.Module] | None = None,
        sa_norm: Optional[nn.Module] = None,
        self_critic_loss_factor: float = 0.1,
        next_loss_factor: float = 0.1,
        detach_hidden_states: bool = False,
        detach_targets: bool = False,
        full_information_blockage: bool = False,
        chance_to_deterministic: float = 0.0,
        deterministic_at_inference: bool = False,
        straight_through_eval: bool = False,
        use_information_bottleneck: bool = True,
        use_hidden_state_prediction: bool = True,
        reconstruction_loss_factor: float = 1.0,
        latent_loss_factor: float = 1.0,
    ):
        super().__init__()
        self.posterior_mlp = posterior_mlp
        self.quantizer = quantizer_mlp
        self.decoder_mlp = decoder_mlp
        self.prior_prediction_mlp = prior_prediction_mlp
        self.prior_prediction_attention = prior_prediction_attention
        self.sa_norm = sa_norm or nn.Identity()
        self.next_loss_factor = next_loss_factor
        self.self_critic_loss_factor = self_critic_loss_factor

        if isinstance(prior_prediction_mlp, ModuleList):
            self.initial_embedding = torch.nn.Parameter(torch.zeros(self.quantizer.num_quantizers, 1, 1, d_model))
        else:
            self.initial_embedding = torch.nn.Parameter(torch.zeros(1, 1, d_model))

        self.detach_hidden_states = detach_hidden_states
        self.detach_targets = detach_targets
        # self.detach_predictions = True
        self.full_information_blockage = full_information_blockage
        self.chance_to_deterministic = chance_to_deterministic
        self.deterministic_at_inference = deterministic_at_inference
        self.straight_through_eval = straight_through_eval
        self.use_information_bottleneck = use_information_bottleneck
        self.use_hidden_state_prediction = use_hidden_state_prediction

        self.reconstruction_loss_factor = reconstruction_loss_factor
        self.latent_loss_factor = latent_loss_factor

    def forward(self,
                h: torch.Tensor,
                padding_mask: torch.Tensor,
                mask: Optional[_MaskType] = None,
                input_pos: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Defines the forward pass of the PHi layer.

        Args:
            h (torch.Tensor): The input hidden states from the main model.
            padding_mask (torch.Tensor): The padding mask for the sequence.
            mask (Optional[_MaskType], optional): The causal attention mask for the
                autoregressive prior. Defaults to None.
            input_pos (Optional[torch.Tensor], optional): Positional encodings for
                the attention mechanism. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing the new hidden state `h`,
                the PHi loss `phi_loss`, and other metrics.
        """
        return_dict = {}
        if self.detach_hidden_states:
            h = h.detach()
        padding_mask = ~padding_mask

        if self.quantizer is None:
            # --- Information Bottleneck ---
            # 1. Compute posterior distribution q(z|h) and sample latent z
            distribution = self.posterior_mlp(h)
            q_mean, q_logvar = distribution.chunk(2, dim=-1)
            q_logvar = torch.clamp(q_logvar, -5, 10)

            if self.full_information_blockage:
                # block all information in the latent space by having zero mean and log variance
                q_mean = q_mean * 0.0
                q_logvar = q_logvar * 0.0

            # 2. Sample from the posterior using the reparameterization trick
            use_information_bottleneck = self.training or not self.deterministic_at_inference
            use_information_bottleneck = use_information_bottleneck and self.use_information_bottleneck
            if use_information_bottleneck: # Only at training time
                noise = torch.exp(0.5 * q_logvar) * torch.randn_like(q_mean)
                if self.chance_to_deterministic > 0.0:
                    deterministic = torch.rand_like(q_mean[:, 0, 0]) < self.chance_to_deterministic
                    noise = noise * ~deterministic.view(-1, 1, 1)
                z = q_mean + noise
            else:
                z = q_mean

            # 3. Self-critic loss to prevent posterior collapse
            self_critic_scores = -F.gaussian_nll_loss(
                q_mean.unsqueeze(1),
                z.unsqueeze(0),
                q_logvar.exp().unsqueeze(1),
                reduction="none").sum(-1).transpose(1, 2)  # shape: (batch_size, seq_len, d_model)
            self_critic_targets = torch.arange(self_critic_scores.shape[2])[:, None].repeat(1, self_critic_scores.shape[1])
            self_critic_losses = F.cross_entropy(
                self_critic_scores.reshape(-1, self_critic_scores.shape[-1]),
                self_critic_targets.flatten().to(h.device),
                reduction="none",
            )
            self_critic_losses = (self_critic_losses * padding_mask.flatten()).view_as(padding_mask)
            self_critic_loss = self_critic_losses.sum() / padding_mask.sum()
            return_dict["self_critic_loss"] = self_critic_loss * self.self_critic_loss_factor

            # --- Latent loss ---
            latent_losses = gaussian_kl(torch.zeros_like(q_mean), 
                                        torch.log(torch.ones_like(q_logvar)),
                                        q_mean, 
                                        q_logvar)
            latent_losses = latent_losses * padding_mask
            return_dict["tokenwise_latent_losses"] = latent_losses
            latent_loss = latent_losses.sum() / padding_mask.sum()
            return_dict["latent_loss"] = latent_loss * self.latent_loss_factor
            entropies = gaussian_entropy(q_logvar)
            entropies = entropies * padding_mask
            return_dict["tokenwise_latent_entropy"] = entropies
            
            # --- Self Prediction ---
            # 4. Compute autoregressive prior p(z_t | z_{<t})
            prediction_input = z
            if self.prior_prediction_attention is not None:
                prediction_input = self.prior_prediction_attention(prediction_input, prediction_input,
                                                                mask=mask, input_pos=input_pos)
            # Shift input for next-step prediction
            prediction_input = prediction_input[:, :-1]  # (batch_size, seq_len - 1, d_model)
            prediction_input = torch.cat((self.initial_embedding.expand(prediction_input.shape[0], -1, -1),
                                        prediction_input), dim=1)  # (batch_size, seq_len, d_model)
            prediction_mean = self.prior_prediction_mlp(self.sa_norm(prediction_input))

            if prediction_mean.shape[-1] == 2 * h.shape[-1]:
                # Split the prediction mean and log variance
                prediction_mean, prediction_logvar = prediction_mean.chunk(2, dim=-1)
                prediction_logvar = torch.clamp(prediction_logvar, -5, 10)

            if not self.use_hidden_state_prediction:
                # Use unit gaussian as a prior if hidden state prediction is not used
                prediction_mean = torch.zeros_like(prediction_mean)
                prediction_logvar = torch.zeros_like(prediction_logvar)

            # 5. Calculate PHi Loss (KL divergence between prior and posterior)
            target_mean = q_mean
            target_logvar = q_logvar
            if self.detach_targets:
                target_mean = target_mean.detach()
                target_logvar = target_logvar.detach()

            target_padding_mask = padding_mask

            # if self.detach_predictions:
            #     prediction_mean_detached = prediction_mean.detach()
            #     prediction_logvar_detached = prediction_logvar.detach()
            #     phi_losses_pred_detach = gaussian_kl(
            #         prediction_mean_detached,
            #         prediction_logvar_detached,
            #         target_mean,
            #         target_logvar,
            #     )
            #     phi_losses_pred_detach = phi_losses_pred_detach * target_padding_mask
            #     return_dict["tokenwise_phi_losses_posterior"] = phi_losses_pred_detach
            #     loss = phi_losses_pred_detach.sum() / target_padding_mask.sum()
            #     return_dict["phi_loss_posterior"] = loss * self.next_loss_factor

            if self.use_information_bottleneck:
                phi_losses = gaussian_kl(
                    prediction_mean,
                    prediction_logvar,
                    target_mean,
                    target_logvar,
                )
            else:
                phi_losses = F.mse_loss(prediction_mean, target_mean, reduction="none")
            phi_losses = phi_losses * target_padding_mask
            return_dict["tokenwise_phi_losses"] = phi_losses
            loss = phi_losses.sum() / target_padding_mask.sum()
            return_dict["phi_loss"] = loss * self.next_loss_factor

            if self.decoder_mlp is not None:
                h_new = self.decoder_mlp(z)
            else:
                h_new = z

            if self.straight_through_eval and not self.training:
                h_new = h

            return_dict["h"] = h_new
            return return_dict

        else:
            h_new, entropies, latent_losses, inds, zs = self.quantizer(h)
            target_padding_mask = padding_mask

            # Reconstruction loss
            l2_norm = F.mse_loss(h, h_new, reduction='none').sum(dim=-1) * target_padding_mask
            return_dict["tokenwise_reconstruction_loss"] = l2_norm
            recon_loss = l2_norm.sum() / padding_mask.sum()
            return_dict["reconstruction_loss"] = recon_loss * self.reconstruction_loss_factor

            # Loss term -> KL divergence of the uniform prior / do not train but good to observe
            latent_losses = latent_losses.sum(-1)
            latent_losses = latent_losses * padding_mask
            for i in range(self.quantizer.num_quantizers):
                return_dict[f"tokenwise_latent_losses{i}"] = latent_losses[i]
            latent_loss = latent_losses.sum() / padding_mask.sum()
            return_dict["latent_loss"] = latent_loss * self.latent_loss_factor
            entropy = entropies * padding_mask
            for i in range(self.quantizer.num_quantizers):
                return_dict[f"tokenwise_latent_entropy{i}"] = entropy[i]

            ind_one_hot = F.one_hot(inds, num_classes=zs.size(-1))  # (batch, seq_len, num_codes)
            self_critic_scores = torch.einsum('q i s c, q j s c -> q i j s', zs, ind_one_hot.to(zs.dtype))  # (batch_z, batch_ind_one_hot, seq_len)
            self_critic_scores = self_critic_scores.transpose(2, 3)  # (batch_z, seq_len, batch_ind_one_hot)
            self_critic_targets = torch.arange(self_critic_scores.shape[3])[:, None].repeat(self_critic_scores.shape[0], 1, self_critic_scores.shape[2])  # (batch, seq_len)
            self_critic_losses = F.cross_entropy(
                self_critic_scores.reshape(-1, self_critic_scores.shape[-1]).float(),
                self_critic_targets.flatten().to(zs.device),
                reduction="none",
            )
            self_critic_losses = (self_critic_losses * repeat(padding_mask, 
                                                              '... -> n ...', 
                                                              n=self.quantizer.num_quantizers).flatten()).view_as(repeat(padding_mask, 
                                                                                                                        '... -> n ...', 
                                                                                                                        n=self.quantizer.num_quantizers))
            return_dict["tokenwise_self_critic_loss"] = self_critic_losses
            self_critic_loss = self_critic_losses.sum() / padding_mask.sum()
            return_dict["self_critic_loss"] = self_critic_loss * self.self_critic_loss_factor

            ### Self Prediction ################################################################
            # compute autoregressive prior based on the previous latent variables
            prediction_z = []
            if self.prior_prediction_attention is not None:
                prediction_input = self.quantizer.quantized_residuals_per_quantizer
                shift_prediction_input = prediction_input[:, :, :-1]
                shift_prediction_input = torch.cat((self.initial_embedding.expand(-1, shift_prediction_input.shape[1], -1, -1),
                                          shift_prediction_input), dim=2)
                residual_to_add = self.quantizer.get_zqr_summed
                for q in range(self.quantizer.num_quantizers):
                    attned_input = self.prior_prediction_attention(shift_prediction_input[q], shift_prediction_input[q],
                                                                    mask=mask, input_pos=input_pos)
                    if self.quantizer.num_quantizers > 1:
                        residual_to_add = residual_to_add[:-1]
                        residual_to_add = torch.cat((torch.zeros_like(residual_to_add[0]).unsqueeze(0),
                                                            residual_to_add), dim=0)
                        attned_input = attned_input + residual_to_add[q]
                        p_xi_r = self.prior_prediction_mlp[q](self.sa_norm(attned_input))
                    prediction_z.append(p_xi_r)
            prediction_z = torch.stack(prediction_z)

            
            # 1. detach posterior ( train only the prior )
            target_z = zs
            if self.detach_targets:
                target_z = target_z.detach()
            
            target_padding_mask = padding_mask

            categroical_input = F.log_softmax(prediction_z, dim=-1) # prior / log probs
            categorical_target = F.log_softmax(target_z, dim=-1) # posterior / log probs
            phi_losses_prior = F.kl_div(categroical_input, categorical_target, reduction='none', log_target=True)
            
            phi_losses_prior = phi_losses_prior.sum(dim=-1) * target_padding_mask
            for i in range(self.quantizer.num_quantizers):
                return_dict[f'tokenwise_phi_losses{i}'] = phi_losses_prior[i]
            loss = phi_losses_prior.sum() / target_padding_mask.sum()
            return_dict['phi_loss'] = loss * self.next_loss_factor

            if self.straight_through_eval and not self.training:
                h_new = h

            # compute perplexity and cluster use for debugging
            # for i in range(self.quantizer.num_quantizers):
            #     ind = inds[i]
            #     encodings = F.one_hot(ind, self.quantizer.num_embeddings).float().reshape(-1, self.quantizer.num_embeddings)
            #     avg_probs = encodings.mean(0)
                
            #     #compute the codebook perplexity
            #     perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp() # exp(Entropy) = perplexity of a probabability distribution
            #     cluster_use = torch.sum(avg_probs > 0)
            #     return_dict[f"tokenwise_perplexity{i}"] = perplexity
            #     return_dict[f"tokenwise_cluster_use{i}"] = cluster_use

            return_dict["h"] = h_new
            return return_dict


    def setup_cache(
        self,
        batch_size: int,
        dtype: th.dtype,
        *,
        max_seq_len: int,
    ) -> None:
        if self.prior_prediction_attention is not None:
            self.prior_prediction_attention.setup_cache(batch_size, dtype, max_seq_len=max_seq_len)

    @property
    def cache_enabled(self) -> bool:
        """Check if the key value caches are set up."""
        enabled = True
        if self.prior_prediction_attention is not None:
            enabled &= self.prior_prediction_attention.kv_cache is not None
        return enabled

    def reset_cache(self):
        """Reset the key value caches."""
        if self.prior_prediction_attention is not None:
            self.prior_prediction_attention.reset_cache()


class PHiLossCollector:
    def __init__(self):
        """
        A simple utility class for accumulating named losses.

        This class provides a straightforward way to collect and sum multiple loss
        values (e.g., PHi loss, self-critic loss) during a training or evaluation
        loop before they are logged or used for backpropagation.
        """
        self.losses = {}

    def add_loss(self, name: str, loss: torch.Tensor):
        """
        Adds a loss value to the running total for a given name.

        If the loss name does not already exist in the collector, it is
        initialized to zero before the new value is added.

        Args:
            name (str): The identifier for the loss.
            loss (torch.Tensor): The loss tensor to add.
        """
        if name not in self.losses:
            self.losses[name] = 0
        self.losses[name] += loss

    def reset(self):
        """Clears all accumulated losses."""
        self.losses = {}