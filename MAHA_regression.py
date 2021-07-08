import torch
import torch.nn as nn
import torch.nn.functional as F

from module import *
from image_embedding import *

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        
class MAHA(nn.Module):
    def __init__(self, dim_input, dim_output, ln, phase, batch_size, num_inds, dim_hidden=128, num_heads=4):
        super(MAHA, self).__init__()
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.phase = phase
        self.batch_size = batch_size
        
        self.x_emb = nn.Sequential(nn.Linear(dim_input, dim_hidden),
                                   nn.ReLU(),
                                   nn.Linear(dim_hidden, dim_hidden))
            
        dim_feature = dim_hidden * dim_output * 2
        self.r_ST = SetTransformer(dim_hidden + dim_output, dim_hidden, dim_feature, num_heads, num_inds, 1, ln)
        self.z_ST = SetTransformer(dim_hidden + dim_output, dim_hidden, dim_feature * 2, num_heads, num_inds, 1, ln)
         
        self.param_MLP = nn.Sequential(nn.Linear(dim_hidden, dim_hidden),
                                       nn.ReLU(),
                                       nn.Linear(dim_hidden, dim_hidden))
        
        self.layernorm = nn.LayerNorm(dim_hidden)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
         
    def stochastic_distribution(self, task_param):
        length = int(task_param.size(-1)/2)
        task_mu = task_param[:, :, :length]
        task_omega = task_param[:, :, length:]
        task_sigma = 0.1+0.9*self.sigmoid(task_omega)
        task_dist = torch.distributions.normal.Normal(task_mu, task_sigma)
        return task_dist
    
    def reparameterize(self, dist):
        eps = torch.randn(size = dist.stddev.size()).to(device)
        sample = dist.mean + dist.stddev * eps
        return sample
    
    def param_encoder(self, x, y, kind):
        inp = torch.cat((x,y), 2)
        if kind == 'deterministic':
            task_param = self.r_ST(inp)
        elif kind == 'stochastic':
            task_param = self.z_ST(inp)
        return task_param
    
    def compute_NLL(self, x0, y, r, z):
        r = r.reshape(-1, self.dim_output * 2, self.dim_hidden)
        z = z.reshape(self.batch_size, self.dim_output * 2, self.dim_hidden)
        dec_param = r + self.param_MLP(z)
        dec_param = self.layernorm(dec_param).permute(0,2,1)
            
        param = torch.matmul(x0, dec_param)
        mu = param[:,:,:self.dim_output]
        omega = param[:,:,self.dim_output:]
        sigma = 0.1+0.9*self.softplus(omega)
            
        y_dist = torch.distributions.normal.Normal(mu, sigma)
        NLL = -y_dist.log_prob(y).sum(-1).mean()
        MSE = ((y-mu)**2).sum(-1).mean()
        return y_dist, NLL, MSE
        
                  
    def forward(self, Cx, Cy, Tx, Ty):
        c0 = self.x_emb(Cx)
        t0 = self.x_emb(Tx)
        
        r_c = self.param_encoder(c0, Cy, 'deterministic')
        z_c_param = self.param_encoder(c0, Cy, 'stochastic')
        z_c_dist = self.stochastic_distribution(z_c_param)
            
        if self.training == False:
            y_dist, NLL, measure = self.compute_NLL(t0, Ty, r_c, z_c_dist.mean)
            return y_dist, NLL, measure    
        
        if self.phase == 'pretrain':
            r_t = self.param_encoder(t0, Ty, 'deterministic')
            r_t_hat = r_t.mean(0, True)
        
            z_t_param = self.param_encoder(t0, Ty, 'stochastic')
            z_t_param_hat = z_t_param.mean(1, True)
            z_t_hat_dist = self.stochastic_distribution(z_t_param_hat)
            z_t_hat = self.reparameterize(z_t_hat_dist)
        
            z_c_param_hat = z_c_param.mean(1, True)
            z_c_hat_dist = self.stochastic_distribution(z_c_param_hat)
        
            KL = torch.distributions.kl.kl_divergence(z_t_hat_dist, z_c_hat_dist).sum(-1).mean()
            y_dist, NLL, measure = self.compute_NLL(t0, Ty, r_t_hat, z_t_hat)
            return y_dist, KL, NLL, measure
        
        elif self.phase == 'finetuning':
            z_t_param = self.param_encoder(t0, Ty, 'stochastic')
            z_t_dist = self.stochastic_distribution(z_t_param)
            z_t = self.reparameterize(z_t_dist)
            
            KL = torch.distributions.kl.kl_divergence(z_t_dist, z_c_dist).sum(-1).mean()
            y_dist, NLL, measure = self.compute_NLL(t0, Ty, r_c, z_t)
            return y_dist, KL, NLL, measure