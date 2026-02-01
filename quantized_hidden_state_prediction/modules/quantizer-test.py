class ResidualQuantizeStep(nn.Module):
    def __init__(self, dim , num_embeddings, stage, shared_encoder=None, shared_decoder=None):
        super().__init__()

        self.dim = dim
        self.num_embeddings = num_embeddings

        # if not shared_encoder:
        self.posterior = nn.Sequential(encoder(self.dim, 2048), 
                        encoder(2048, self.num_embeddings))
        
        self.codebook = nn.Embedding(num_embeddings, dim)
        # self.codebook.weight.data.normal_(mean=0.0, std=1/(2**stage))
        
        # if not shared_decoder:
        #     self.decoder = nn.Sequential(decoder(self.dim, 1024),
        #                                  decoder(1024, self.dim))

    def encode(self, x, shared_encoder: torch.nn.Module | None = None):
        posterior = default(shared_encoder, self.posterior)
        z = posterior(x)
        return z

    # def decode(self, z_q, shared_decoder: torch.nn.Module | None = None):
    #     decoder = default(shared_decoder, self.decoder)
    #     x_tilde = decoder(z_q)
    #     return x_tilde

    def quantize(self, z):
        assert self.temperature is not None
        one_hot = F.gumbel_softmax(z, tau=self.temperature, dim=2, hard=True) # shape: (bsz, num_tokens, num_embeddings)
        z_q = einsum('b s n, n d -> b s d', one_hot, self.codebook.weight) # shape: (bsz, num_tokens, embedding_dim)        
        ind = one_hot.argmax(dim=2)
        qy = F.softmax(z, dim=2)

        # KL term from the ELBO with a uniform prior
        latent_loss = qy * torch.log(qy * self.num_embeddings + 1e-10)
        return z_q, latent_loss, ind

    def forward(self, res, z_hat, shared_encoder=None, shared_decoder=None):
        z = self.encode(res, shared_encoder=shared_encoder)
        z_q, latent_loss, code_idxs = self.quantize(z)
        res = res  - z_q.detach()
        z_hat = z_hat + z_q
        # x_tilde = self.decode(z_hat, shared_decoder=shared_decoder)
        return res, code_idxs, z, z_hat # , x_tilde


class ResidualQuantize(nn.Module):
    def __init__(self, 
                 num_quantizers : int, 
                 num_embedding : int, 
                 dim : int, 
                 shared_encoder_flag : bool = False,
                 shared_decoder_flag : bool = True):
                #  posterior_mlp : nn.Module | None = None, 
                #  decoder_mlp : nn.Module | None = None):
        super().__init__()

        self.num_quantizers = num_quantizers
        self.num_embeddings = num_embedding # embeddings per codebook
        self.dim = dim

        self.shared_encoder = None
        self.shared_decoder = None
        # if shared_encoder_flag:
        #     self.shared_encoder = nn.Sequential(encoder(self.dim, 2048), 
        #                                         encoder(2048, self.num_embeddings))
        # if shared_decoder_flag:
        #     self.shared_decoder = nn.Sequential(decoder(self.dim, 1024),
        #                                         decoder(1024, self.dim))
        self.shared_decoder = nn.Sequential(decoder(self.dim, 1024),
                                                decoder(1024, self.dim))
        
        self.stages = []
        for m in range(self.num_quantizers):
            rq_step = ResidualQuantizeStep(self.dim, self.num_embeddings, m)
            self.add_module(f"stage{m}", rq_step)
            self.stages.append(rq_step)

    def forward(self, x):
        residual = x
        z_hat = 0.
        # latent_losses = torch.zeros(self.num_quantizers, x.shape[0])

        reconstruction_losses_per_stage = torch.zeros(self.num_quantizers, *x.shape).to(x.device)
        self.zs = torch.zeros(self.num_quantizers, *x.shape[:-1], self.num_embeddings).to(x.device)
        self.indices = torch.zeros_like(self.zs[..., 0], dtype=torch.long).to(x.device)
        losses = {}
        for m, stage in enumerate(self.stages):
            stage.temperature = self.temperature
            
            residual, code_idxs, z, z_q_summed = stage(residual, z_hat, self.shared_encoder, self.shared_decoder)
            
            # reconstruction_losses_per_stage[m] = (x_hat - x).pow(2)
            self.zs[m, ...] = z
            self.indices[m, ...] = code_idxs
        x_hat = self.shared_decoder(z_q_summed)
        # losses['reconstruction_losses'] = reconstruction_losses_per_stage
        return x_hat, residual, losses

    @property
    def get_zs(self):
        return self.zs

    @property
    def get_code_indices(self):
        return self.indices

    # @property
    # def get_quantized_inputs(self):
    #     return torch.stack(self._zqs)

    # def cache_zqs(self, z_q):
    #     self._zqs.append(z_q)

    # def clear_cache_zqs(self):
    #     self._zqs = [[] for _ in range(self.num_quantizers)][0]


class RQStep(nn.Module):
    def __init__(self, m, posteriors, codebooks): # , decoders):
        super().__init__()
        if not isinstance(posteriors, nn.Module):
            self.posterior = posteriors()
        else:
            self.posterior = posteriors

        # if not isinstance(decoders, nn.Module):
        #     self.decoder = decoders()
        # else:
        #     self.decoder = decoders

        if not isinstance(codebooks, nn.Module):
            self.codebook = codebooks[m]
        else:   
            self.codebook = codebooks

    def get_codes(self, ind):
        one_hot = F.one_hot(ind, num_classes=self.codebook.weight.shape[0]).to(self.codebook.weight.dtype)
        return einsum('b s n, n d -> b s d', one_hot, self.codebook.weight)

    def forward(self, res):
        z = self.posterior(res)
        one_hot = F.gumbel_softmax(z, tau=self.temperature, dim=2, hard=True) # shape: (bsz, num_tokens, num_embeddings)
        z_q = einsum('b s n, n d -> b s d', one_hot, self.codebook.weight) # shape: (bsz, num_tokens, embedding_dim)
        ind = one_hot.argmax(dim=2)
        qy = F.softmax(z, dim=2)

        # KL term from the ELBO with a uniform prior
        diff = qy * torch.log(qy * self.codebook.weight.shape[0] + 1e-9)
        return z, z_q, diff, ind

class TestResQt(nn.Module):
    def __init__(self, num_quantizers, num_embedding, dim, 
                 posterior_mlp : nn.Module | None = None, 
                 decoder_mlp : nn.Module | None = None):
        super().__init__()

        self.num_quantizers = num_quantizers
        self.num_embeddings = num_embedding # this is per codebook
        self.dim = dim

        self.codebooks = self._init_codebooks()
        self.posteriors = default(posterior_mlp, self._get_posteriors)
        # self.posterior = nn.Linear(self.dim, self.num_embeddings)
        # self.decoders = self._get_decoders()
        self.decoder = default(decoder_mlp, self._get_decoder)
        # for i in range(self.num_quantizers):
        self.stages = []
        for m in range(self.num_quantizers):
            rq_step = RQStep(m, self.posteriors, self.codebooks) # , self.decoder)
            self.add_module(f"stage{m}", rq_step)
            self.stages.append(rq_step)
    
    def _get_decoder(self):
        decoder = nn.Sequential(
            decoder(self.dim, self.dim*2),
            decoder(self.dim*2, self.dim),)
        return decoder

    def _get_posteriors(self):
        posterior = nn.Sequential(encoder(self.dim, self.dim*2), 
                                   encoder(self.dim*2, self.num_embeddings),)
        return posterior

    def _init_codebooks(self):
        codebooks = [nn.Embedding(self.num_embeddings, self.dim)]
        for m in range(1, self.num_quantizers):
            cb_init = nn.Embedding(self.num_embeddings, self.dim)
            cb_init.weight.data.normal_(mean=0.0, std=1/(2**m))
            codebooks.append(cb_init)
        return codebooks


    def forward(self, x):
        residual = x
        quantized_out = 0.
        self.clear_cache_zqs()
        
        indices = []
        latent_losses = []
        zs = []

        # TODO: modify posterior to handle both single module and module list 
        for stage in self.stages:
            stage.temperature = self.temperature
            z, _, latent_loss, ind = stage(residual)
            z_q = stage.get_codes(ind)
            residual = residual - z_q.detach()
            quantized_out = quantized_out + z_q
            
            indices.append(ind)
            latent_losses.append(latent_loss)
            zs.append(z)
        
        x_hat = self.decoder(quantized_out)
        indices = torch.stack(indices, dim=0)
        latent_losses = torch.stack(latent_losses, dim=0)
        zs = torch.stack(zs, dim=0)
        return x_hat, latent_losses, indices, zs

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
