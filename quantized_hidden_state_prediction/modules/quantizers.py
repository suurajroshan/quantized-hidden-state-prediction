import os
import inspect
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from scipy.cluster.vq import kmeans2
from torchtune.modules import MultiHeadAttention
from torchtune.models.llama3_1._position_embeddings import Llama3ScaledRoPE
from einops import rearrange, repeat


class VQVAEQuantize(nn.Module):
    """
    Neural Discrete Representation Learning, van den Oord et al. 2017
    https://arxiv.org/abs/1711.00937

    Follows the original DeepMind implementation
    https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
    """
    def __init__(self, 
                num_embeddings : int = 1024, 
                embedding_dim : int = 512):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim


        self._kld_scale = 1.

        self.embed = nn.Embedding(num_embeddings, embedding_dim)

        self.register_buffer('data_initialized', torch.zeros(1))

        # TODO: temporary fix, resolve later
        self.num_quantizers = 1

    def forward(self, z):
        """
        bsz: batch size
        s : sequence length
        hid_dim : hidden dimension
        """

        bsz, s, hid_dim = z.size()
        flatten = z.reshape(-1, self.embedding_dim)

        if not self.data_initialized.item() and self.training:
            print('running kmeans!')
            rp = torch.randperm(flatten.size(0))
            #TODO: check for Warning: One of the clusters is empty. Re-run kmeans with a different initialization. 
            # return fun(*args, **kwargs)
            kd = kmeans2(flatten[rp].to(torch.float32).data.cpu().numpy(), 
                        self.num_embeddings, 
                        minit='points',)
            self.embed.weight.data.copy_(torch.from_numpy(kd[0]))
            self.data_initialized.fill_(1)

        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed.weight.t()
            + self.embed.weight.pow(2).sum(1, keepdim=True).t()
        )

        _, ind = (-dist).max(1)
        ind = ind.view(bsz, s)

        z_q = self.embed_code(ind)
        commitment_cost = 0.25
        diff = commitment_cost * ( ( z_q.detach() - z ).pow(2) + ( z_q - z.detach() ).pow(2) )
        diff *= self._kld_scale

        z_q = z + (z_q - z).detach() # noop in forward pass, straight-through gradient estimator in backward pass
        return z_q, diff, [ind]

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.weight)


class GumbelQuantize(nn.Module):
    def __init__(self, num_embeddings: int = 1024, embedding_dim:int=512):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        self.proj = nn.Linear(embedding_dim, num_embeddings, bias=False) # nn.Sequential(nn.Linear(embedding_dim, num_embeddings), nn.ReLU(inplace=True))

    def forward(self, logits):
        one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=2, hard=True) # shape: (bsz, num_tokens, num_embeddings)
        z_q = einsum('b s n, n d -> b s d', one_hot, self.embed.weight) # shape: (bsz, num_tokens, embedding_dim)
        ind = one_hot.argmax(dim=2)

        qy = F.softmax(logits, dim=2)
        # KL term from the ELBO with a uniform prior
        diff = - torch.log(qy * self.num_embeddings + 1e-10) / qy

        return z_q, diff, ind


# helper function
def default(a, b):
    return a if a else b

class ResidualQuantize(nn.Module):
    def __init__(self, 
                 num_quantizers : int, 
                 num_embedding : int, 
                 dim : int, 
                 posterior_mlp : nn.Module | None = None, 
                 decoder_mlp : nn.Module | None = None):
        super().__init__()

        self.num_quantizers = num_quantizers
        self.num_embeddings = num_embedding # this is per codebook
        self.dim = dim

        self.codebooks = self._init_codebooks()
        self.posteriors = default(posterior_mlp, self._get_posteriors())
        self.decoder = default(decoder_mlp, self._get_decoder())

    def _get_codes_and_indices(self, x, codebook_idx):
        temperature = 1e-6 if not self.training else self.temperature
        one_hot = F.gumbel_softmax(x, tau=temperature, dim=2, hard=True) # shape: (bsz, num_tokens, num_embeddings)
        z_q = einsum('b s n, n d -> b s d', one_hot, self.codebooks[codebook_idx].weight) # shape: (bsz, num_tokens, embedding_dim)
        ind = one_hot.argmax(dim=2)
        qy = F.softmax(x, dim=2)

        # KL term from the ELBO with a uniform prior
        diff = qy * torch.log(qy * self.num_embeddings + 1e-10)
        return z_q, diff, ind

    def get_entropy(self, x):
        qy = F.softmax(x, dim=2)
        entropy = - torch.sum(qy*torch.log(qy + 1e-10), dim=-1)
        return entropy
        
    
    def _get_decoder(self):
        shared_decoder = nn.Sequential(
            decoder(self.dim, self.dim*2),
            decoder(self.dim*2, self.dim),)
        return shared_decoder
    #     for _ in range(1, self.num_quantizers):
    #         decoder.append(nn.Linear(self.dim, self.dim))
    #     return decoder

    def _get_posteriors(self):
        posterior = nn.ModuleList([
            nn.Sequential(encoder(self.dim, self.dim*2//3), 
                          encoder(self.dim*2//3, self.num_embeddings),) for _ in range(self.num_quantizers)
            ])
        # for m in range(1, self.num_quantizers):
        #     posterior_m = nn.Sequential(
        #         encoder(self.dim, self.dim*2),
        #         encoder(self.dim*2, self.num_embeddings),)
        #     posterior.append(posterior_m)
        return posterior

    def _init_codebooks(self):
        codebooks = nn.ModuleList([nn.Embedding(self.num_embeddings, self.dim) for _ in range(self.num_quantizers)])
        # for m in range(1, self.num_quantizers):
        #     cb_init = nn.Embedding(self.num_embeddings, self.dim)
        #     cb_init.weight.data.normal_(mean=0.0, std=1/(2**m))
            # codebooks.append(cb_init)
        return codebooks

    def forward(self, x):
        residual = x
        quantized_out = 0.
        h_reconstructions = []
        self.clear_cache_zqs()
        self.clear_cache_zqr_summed()
        
        indices = []
        latent_losses = []
        entropies = []
        zs = []
        # TODO: modify posterior to handle both single module and module list 
        for i in range(self.num_quantizers):
            z = self.posteriors[i](residual)  # shape: (bsz, seq_len, num_embeddings)
            entropy = self.get_entropy(z)
            z_q, latent_loss, ind = self._get_codes_and_indices(z, i)

            self.cache_zqs(z_q)
            
            residual = residual - z_q.detach()
            
            quantized_out = quantized_out + z_q
            self.cache_zqr_summed(quantized_out)
            
            indices.append(ind)
            latent_losses.append(latent_loss)
            zs.append(z)
            entropies.append(entropy)
            h_reconstructions.append(self.decoder(z_q))

        self.zqs_final = quantized_out
        self._h_reconstructions = torch.stack(h_reconstructions, dim=0)
        xhat = self.decoder(quantized_out)
        indices = torch.stack(indices, dim=0)
        latent_losses = torch.stack(latent_losses, dim=0)
        zs = torch.stack(zs, dim=0)
        entropies = torch.stack(entropies, dim=0)
        return xhat, entropies, latent_losses, indices, zs

    @property
    def quantized_residuals_per_quantizer(self):
        return torch.stack(self._zqs)

    @property
    def get_quantizer_zqs(self):
        return torch.cat((rearrange(self.zqs_final, '... -> 1 ...'), self.get_quantized_inputs), dim=0)

    @property
    def get_reconstructions(self):
        return self._h_reconstructions

    def cache_zqs(self, z_q):
        self._zqs.append(z_q)

    def clear_cache_zqs(self):
        self._zqs = []

    def cache_zqr_summed(self, zqr):
        self._zqr_summed.append(zqr)

    def clear_cache_zqr_summed(self):
        self._zqr_summed = []

    @property
    def get_zqr_summed(self):
        return torch.stack(self._zqr_summed)

    def save_codebooks(self, step, path = f'checkpoints/codebooks', filename='last'):
        filepath = os.path.join(path, f'{filename}_{step}.pt')
        # assert os.path.exists(os.path.dirname(filepath))
        if not os.path.exists(os.path.dirname(path)):
            print('creating directory for saving codebooks')
            os.makedirs(path, exist_ok=True)
        torch.save(self.codebooks, filepath)
        print(f'saved codebooks at {filepath}')

class encoder(nn.Module):
    def __init__(self, dim: int = 768, num_embeddings : int = 1024):
        super().__init__()
        self.linear= nn.Linear(dim, num_embeddings)
        self.batch_norm = nn.BatchNorm1d(num_embeddings)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x.transpose(1,2)).transpose(1,2)
        x = self.relu(x)
        return x

class decoder(nn.Module):
    def __init__(self, in_dim : int = 768, out_dim : int = 768):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.batch_norm = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x.transpose(1,2)).transpose(1,2)
        x = self.relu(x)
        return x

        

# ##### QINCO Quantizer #####
# class QincoStep(nn.Module):
#     def __init__(self, 
#                  num_embeddings: int = 1024, 
#                  embedding_dim:int = 512, 
#                  num_residual_blocks:int=2,
#                  res_block_hidden_dim: int = 1024,
#                  init_codebook = None):
#         super().__init__()
#         assert init_codebook is not None
#         self.num_embeddings = num_embeddings
#         self.embedding_dim = embedding_dim
#         self.num_residual_blocks = num_residual_blocks
#         self.h_dim = res_block_hidden_dim

#         self.codebook = nn.Embedding(num_embeddings, embedding_dim)
#         self.codebook.weight.data = init_codebook.weight.data.clone()
#         self.concat_mlp = nn.Linear(2*embedding_dim, embedding_dim)

#         self.residual_blocks = []
#         for i in range(num_residual_blocks):
#             residual_block = nn.Sequential(nn.Linear(embedding_dim, res_block_hidden_dim, bias=False),
#                                            nn.ReLU(inplace=True),
#                                            nn.Linear(res_block_hidden_dim, embedding_dim, bias=False),)
#             self.add_module(f"residual_block_{i}", residual_block)
#             self.residual_blocks.append(residual_block)

#     def encode(self, x_tilde, x):
#         zqs = self.codebook.weight
#         bsz, seq_len, dim = x.shape

#         zqs_r = zqs.repeat(bsz, 1, 1).reshape(-1, self.embedding_dim)
#         x_tilde_r = x_tilde[:, None, ...].repeat(1, self.num_embeddings, 1).reshape(-1, self.embedding_dim)

#         cc = torch.cat((zqs_r, x_tilde_r), dim=1)
#         zqs_r = zqs_r + self.concat_mlp(cc)

#         for res_block in self.resiual_blocks:
#             zqs_r = zqs_r + res_block(zqs_r)

#         zqs_r = zqs_r.reshape(bsz, self.num_embeddings, self.embedding_dim) + x_tilde.reshape(bsz, 1, self.embedding_dim)
#         codes, x_tilde_next = assign_batch_multiple(x, zqs_r)

#         return codes, x_tilde_next - x_tilde
    
#     def decode(self, x_tilde, codes):
#         zqs = self.codebook(codes)
#         cc = torch.cat((zqs, x_tilde), dim=1)
#         zqs = zqs + self.concat_mlp(cc)

#         for res_block in self.residual_blocks:
#             zqs = zqs + res_block(zqs)

#         return zqs
    
# class QincoQuantize(nn.Module):
#     def __init__(self, num_embeddings: int = 1024, 
#                  embedding_dim : int = 512, 
#                  num_quantizers : int = 3,
#                  init_codebooks = None):
#         super().__init__()

#         assert init_codebooks is not None

#         self.num_embeddings = num_embeddings
#         self.embedding_dim = embedding_dim
#         self.num_quantizers = num_quantizers

#         self.codebook0 = nn.Embedding(num_embeddings, embedding_dim)
#         self.codebook0.weight.data = init_codebooks[0].weight.data.clone()

#         self.stages = []
#         for m in range(1, num_quantizers):
#             stage = QincoStep(init_codebook=init_codebooks[m])
#             self.add_module(f"stage{m}", stage)
#             self.stages.append(stage)
    
#     def decode(self, codes):
#         x_tilde = self.codebook0(codes[0])
#         for i, stage in enumerate(self.stages):
#             x_tilde = x_tilde + stage.decode(x_tilde, codes[i+1])
#         return x_tilde
    
#     def encode(self, x):
#         bsz, seq_len, dim = x.shape
#         codes = torch.zeros(bsz, self.num_quantizers, dtype=int, device=x.device)
        
#         code0 = assign_codes(x, self.codebook0.weight)
#         codes[:,0] = code0

#         x_tilde = self.codebook0(code0)

#         for i, stage in enumerate(self.stages):
#             codes[:, i+1], toadd = stage.encode(x_tilde, x)
#             x_tilde = x_tilde + toadd
#         return codes, x_tilde 

#     def forward(self, x):
#         codes, x_tilde = self.encode(x)
#         losses = torch.zeros(self.num_quantizers)
#         x_tilde = self.codebook0(codes[:,0])
#         losses[0] = (x_tilde - x).pow(2).sum()
#         for i, stage in enumerate(self.stages):
#             x_tilde = x_tilde + self.decode(x_tilde, codes[:, i+1])
#             losses[i+1] = (x_tilde - x).pow(2).sum()
#         return codes, x_tilde, losses
    

# qinco quantize
class QincoQuantize(nn.Module):
    def __init__(self, 
                 num_quantizers : int, 
                 num_embedding : int, 
                 dim : int, 
                 posterior_mlp : nn.Module | None = None, 
                 decoder_mlp : nn.Module | None = None,
                 num_residual_blocks:int=2,
                 init_codebook = None):
        super().__init__()

        self.num_quantizers = num_quantizers
        self.num_embeddings = num_embedding # this is per codebook
        self.dim = dim

        self.codebooks = self._init_codebooks(init_codebook)
        self.posteriors = default(posterior_mlp, self._get_posteriors())
        self.decoder = default(decoder_mlp, self._get_decoder())

        self.attend_to_inputs = self.attn_inputs()
        self.concat_mlp = self.init_concat_mlps()
        self.codebook_prediction_mlp = self.init_residual_blocks_mlp(num_residual_blocks)

    def _get_codes_with_weights(self, x, codebook_weights, codebook_idx):
        z = self.posteriors[0](x)

        one_hot = F.gumbel_softmax(z, tau=self.temperature, dim=2, hard=True) # shape: (bsz, num_tokens, num_embeddings)
        z_q = einsum('b s n, b n s d -> b s d', one_hot, codebook_weights) # shape: (bsz, num_tokens, embedding_dim)
        ind = one_hot.argmax(dim=2)
        qy = F.softmax(z, dim=2)

        # KL term from the ELBO with a uniform prior
        diff = qy * torch.log(qy * self.num_embeddings + 1e-10)
        # return z_q, diff, ind
        return z_q, diff, ind
    
    def _get_codes_and_indices(self, x, codebook_idx):
        z = self.posteriors[0](x)

        one_hot = F.gumbel_softmax(z, tau=self.temperature, dim=2, hard=True) # shape: (bsz, num_tokens, num_embeddings)
        z_q = einsum('b s n, n d -> b s d', one_hot, self.codebooks[codebook_idx].weight) # shape: (bsz, num_tokens, embedding_dim)
        ind = one_hot.argmax(dim=2)
        qy = F.softmax(z, dim=2)

        # KL term from the ELBO with a uniform prior
        diff = qy * torch.log(qy * self.num_embeddings + 1e-10)
        # return z_q, diff, ind
        return z_q, diff, ind
    
    def _get_decoder(self):
        shared_decoder = nn.Sequential(
            decoder(self.dim, self.dim*2),
            decoder(self.dim*2, self.dim),)
        return shared_decoder
    #     for _ in range(1, self.num_quantizers):
    #         decoder.append(nn.Linear(self.dim, self.dim))
    #     return decoder

    def _get_posteriors(self):
        posterior = nn.ModuleList([
            nn.Sequential(encoder(self.dim, self.dim*2//3), 
                          encoder(self.dim*2//3, self.num_embeddings),)
            ])
        return posterior

    def _init_codebooks(self, init_codebook):
        assert init_codebook is not None
        codebooks = nn.ModuleList([])
        for m in range(self.num_quantizers):
            cb_init = nn.Embedding(self.num_embeddings, self.dim)
            cb_init.weight.data = init_codebook[m].weight.data.clone()
            codebooks.append(cb_init)
        return codebooks

    def init_concat_mlps(self):
        concat_mlps = nn.ModuleList([None,])
        for m in range(1, self.num_quantizers):
            concat_mlp = nn.Linear(2*self.dim, self.dim)
            concat_mlps.append(concat_mlp)
        return concat_mlps

    def attn_inputs(self):
        attnd_to_inputs = nn.ModuleList([None,])
        num_heads = 8
        num_kv_heads = 8
        head_dim = self.dim // num_heads
        rope = Llama3ScaledRoPE(dim=head_dim, max_seq_len=2048, base=500_000, scale_factor=32)
        for m in range(1, self.num_quantizers):
            attnd_to_inputs.append(MultiHeadAttention(
                embed_dim=self.dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_proj=nn.Linear(self.dim, num_heads * head_dim, bias=False),
                k_proj=nn.Linear(self.dim, num_kv_heads * head_dim, bias=False),
                v_proj=nn.Linear(self.dim, num_kv_heads * head_dim, bias=False),
                output_proj=nn.Linear(self.dim, self.dim, bias=False),
                pos_embeddings=rope,
                max_seq_len=2048,
                attn_dropout=0.,)
            )
        return attnd_to_inputs

    def init_residual_blocks_mlp(self, L):
        all_residual_blocks = nn.ModuleList([None, ])
        residual_blocks = nn.ModuleList([])
        for _ in range(1, self.num_quantizers):
            for l in range(L):
                residual_block = nn.Sequential(
                    nn.Linear(self.dim, self.dim, bias=False), nn.ReLU(), nn.Linear(self.dim, self.dim, bias=False)
                )
                residual_blocks.append(residual_block)
            all_residual_blocks.append(residual_blocks)
        return all_residual_blocks

    def encode(self, x):
        indices = []
        latent_losses = []
        zs = []
        xhat, _, code0 = self._get_codes_and_indices(x, 0)
        indices.append(code0)

        for i in range(1, self.num_quantizers):
            codebook_weights = self.codebooks[i].weight
            ne, d = codebook_weights.shape
            bsz, seq_len, d = xhat.shape

            xhat = self.attend_to_inputs[i](xhat, xhat, mask=None, input_pos=torch.arange(seq_len).repeat(bsz,1).to(xhat.device))
            xhat = repeat(xhat, 'b s d -> b ne s d', ne=ne)

            codebook_weights = repeat(codebook_weights, 'ne d -> b ne s d', b=bsz, s=seq_len)

            cc = torch.cat((rearrange(codebook_weights, 'b ne s d -> (b ne s) d'), 
                            rearrange(xhat, 'b ne s d -> (b ne s) d')), 
                            dim=1)

            codebook_weights = codebook_weights + rearrange(self.concat_mlp[i](cc), 
                                                            '(b ne s) d -> b ne s d', 
                                                            b=bsz, ne=ne, s=seq_len)

            for rblock in self.codebook_prediction_mlp[i]:
                codebook_weights = codebook_weights + rblock(codebook_weights)

            codebook_weights = codebook_weights + xhat
            xhat, _ ,ind = self._get_codes_with_weights(x, codebook_weights, i)

            indices.append(ind)
            
            # z_q, latent_loss, ind = self._get_codes_and_indices(z, i)

            # self.cache_zqs(z_q)
            
            # residual = residual - z_q.detach()
            
            # quantized_out = quantized_out + z_q
            
            # indices.append(ind)
            # latent_losses.append(latent_loss)
            # zs.append(z)
        
        indices = torch.stack(indices, dim=0)
        # latent_losses = torch.stack(latent_losses, dim=0)
        # zs = torch.stack(zs, dim=0)
        return indices, xhat

    def decode(self, xhat, codes, concat_mlp, codebook_idx):
        zqs = self.codebooks[codebook_idx](codes)

        # print(xhat.shape)

        xhat = self.attend_to_inputs[codebook_idx](xhat, xhat, mask=None, input_pos=torch.arange(xhat.size(1)).repeat(xhat.size(0),1).to(xhat.device))
        # xhat = xhat.mean(dim=1, keepdim=True).repeat(1, self.num_embeddings, 1)

        cc = torch.cat((zqs, xhat), 2)
        zqs = zqs + concat_mlp(cc)

        for residual_block in self.codebook_prediction_mlp[codebook_idx]:
            zqs = zqs + residual_block(zqs)

        return zqs
        
    
    def forward(self, x):
        quantized_out = 0.
        self.clear_cache_zqs()

        with torch.no_grad():
            # xhat, _, code0 = self._get_codes_and_indices(x, 0)
            # xhat = self.codebook0.weight[code0]
            indices, xhat = self.encode(x)

        losses = []
        xhat = self.codebooks[0](indices[0])
        losses.append(F.mse_loss(xhat, x, reduction='none').sum(-1))

        for i in range(1, self.num_quantizers):
            xhat = xhat + self.decode(xhat, indices[i], self.concat_mlp[i], i)
            losses.append(F.mse_loss(xhat, x, reduction='none').sum(-1))

        # TODO: modify posterior to handle both single module and module list 
        losses = torch.stack(losses, dim=0)
        quantized_out = self.decoder(xhat)
        return quantized_out, indices, losses


    @property
    def get_quantized_inputs(self):
        return torch.stack(self._zqs)

    def cache_zqs(self, z_q):
        self._zqs.append(z_q)

    def clear_cache_zqs(self):
        self._zqs = [[] for _ in range(self.num_quantizers)][0]

    def save_codebooks(self, path = f'checkpoints/codebooks', filename='last.pt'):
        filepath = os.path.join(path, filename)
        assert os.path.exists(filepath)
        if not os.path.exists(path):
            print('creating directory for saving codebooks')
        os.makedirs(path, exist_ok=True)
        torch.save(self.codebooks, filepath)
        print(f'saved codebooks at {filepath}')


def quantizer_module(self_prediction_information_bottleneck,
                         self_prediction_module,
                         embed_dim):
    quantizer_module = {
            'vector_quantize': VQVAEQuantize,
            'gumbel_quantize': GumbelQuantize,
            'residual_quantize':  ResidualQuantize, #  TestRQ, TestResQt  # ResidualQuantize,
            'residual_quantize_qinco': QincoQuantize,
            'continuous': None,
        }[self_prediction_information_bottleneck]

    if self_prediction_information_bottleneck == 'gumbel_quantize':
        codeword_dim = self_prediction_module['codeword_dim']
        codebook_dim = self_prediction_module['codebook_dim']
        embed_dim = codeword_dim  # override embed dim if using quantization
        hidden_dim = codeword_dim * 8 // 3 # TODO: currrently a heuristic, can be parameterized later
        print(f'number of embeddings: {codebook_dim}')
        print(f'embedding dimension: {codeword_dim}')
        posterior_mlp = encoder(codebook_dim, codeword_dim)
        # decoder_mlp = decoder(codeword_dim, embed_dim)
        decoder_mlp = decoder(codeword_dim, embed_dim)
        quantizer_mlp = quantizer_module(codebook_dim, codeword_dim) #  shared_encoder_flag, shared_decoder_flag)

    elif self_prediction_information_bottleneck == 'residual_quantize':
        codeword_dim = self_prediction_module['codeword_dim']
        codebook_dim = self_prediction_module['codebook_dim']
        num_quantizers = self_prediction_module["num_quantizers"]
        # shared_encoder_flag = self_prediction_module["shared_encoder"]
        # shared_decoder_flag = self_prediction_module["shared_decoder"]
        embed_dim = codeword_dim  # override embed dim if using quantization
        hidden_dim = codeword_dim * 8 // 3 # TODO: currrently a heuristic, can be parameterized later
        print(f'number of embeddings: {codebook_dim}')
        print(f'embedding dimension: {codeword_dim}')
        # quantizer_mlp = quantizer_module(num_embeddings=codebook_dim, embedding_dim=codeword_dim)
        posterior_mlp = encoder(codebook_dim, codeword_dim)
        # decoder_mlp = decoder(codeword_dim, embed_dim)
        decoder_mlp = decoder(codeword_dim, embed_dim)
        quantizer_mlp = quantizer_module(num_quantizers, codebook_dim, codeword_dim,decoder_mlp=decoder_mlp) #  shared_encoder_flag, shared_decoder_flag)
    
    elif self_prediction_information_bottleneck == 'vector_quantize':
        codeword_dim = self_prediction_module['codeword_dim']
        codebook_dim = self_prediction_module['codebook_dim']
        embed_dim = codeword_dim  # override embed dim if using quantization
        hidden_dim = codeword_dim * 8 // 3 # TODO: currrently a heuristic, can be parameterized later
        print(f'number of embeddings: {codebook_dim}')
        print(f'embedding dimension: {codeword_dim}')
        # quantizer_mlp = quantizer_module(num_embeddings=codebook_dim, embedding_dim=codeword_dim)
        posterior_mlp = encoder(codeword_dim, codeword_dim)
        decoder_mlp = decoder(codeword_dim, embed_dim)
        quantizer_mlp = quantizer_module(codebook_dim, codeword_dim)

    elif self_prediction_information_bottleneck == 'residual_quantize_qinco':
        codeword_dim = self_prediction_module['codeword_dim']
        codebook_dim = self_prediction_module['codebook_dim']
        num_quantizers = self_prediction_module["num_quantizers"]
        assert self_prediction_module["qinco"]["saved_codebooks"] is not None or ''
        print('loading pretrained codebooks from ', self_prediction_module["qinco"]["saved_codebooks"])
        saved_codebooks = torch.load(self_prediction_module["qinco"]["saved_codebooks"], weights_only=False)
        embed_dim = codeword_dim  # override embed dim if using quantization
        hidden_dim = codeword_dim * 8 // 3 # TODO: currrently a heuristic, can be parameterized later
        print(f'number of embeddings: {codebook_dim}')
        print(f'embedding dimension: {codeword_dim}')
        # quantizer_mlp = quantizer_module(num_embeddings=codebook_dim, embedding_dim=codeword_dim)
        posterior_mlp = encoder(codebook_dim, codeword_dim)
        decoder_mlp = decoder(codeword_dim, embed_dim)
        # init_codebook = self_prediction_module["qinco"]["saved_codebooks"]
        quantizer_mlp = quantizer_module(num_quantizers, codebook_dim, codeword_dim, init_codebook=saved_codebooks,) #  shared_encoder_flag, shared_decoder_flag)
    
    else:
        posterior_mlp = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        quantizer_mlp = quantizer_module
        decoder_mlp=nn.Linear(embed_dim, embed_dim, bias=False)

    return posterior_mlp, quantizer_mlp, decoder_mlp