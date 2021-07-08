import torch
import torch.nn as nn
import torch.nn.functional as F

from module import *
from image_embedding import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
class MAHA(nn.Module):
    def __init__(self, dim_input, ln, phase, batch_size, num_inds, dp_rate, dim_hidden=128, num_heads=4):
        super(MAHA, self).__init__()
        self.dim_hidden = dim_hidden
        self.phase = phase
        self.batch_size = batch_size
        
        if dim_input == 84*84*3:
            if phase == 'pretrain':
                is_pretrain=True
            else:
                is_pretrain=False
                
            self.x_emb = ImageEmbedding(dim_hidden, dp_rate, is_pretrain)
            
        else:
            self.x_emb = nn.Sequential(nn.Dropout(p=dp_rate),
                                       nn.Linear(dim_input, dim_hidden),
                                       nn.ReLU(),
                                       nn.Linear(dim_hidden, dim_hidden))
            
        dim_feature = dim_hidden
        self.r_ST = SetTransformer(dim_hidden, dim_hidden, dim_feature, num_heads, num_inds, 1, ln)
        self.z_ST = SetTransformer(dim_hidden, dim_hidden, dim_feature * 2, num_heads, num_inds, 1, ln)
            
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
        task_param = torch.tensor([]).to(device)
        for batch_idx in range(self.batch_size):
            class_param = torch.tensor([]).to(device)
            for class_idx in range(y.size(-1)):
                index = class_idx == y[batch_idx].max(-1).indices
                inp = x[batch_idx][index].reshape(1, -1, self.dim_hidden)
                if kind == 'deterministic':
                    rep = self.r_ST(inp)
                elif kind == 'stochastic':
                     rep = self.z_ST(inp)
                class_param = torch.cat((class_param, rep), 1)
            task_param = torch.cat((task_param, class_param), 0)        
        return task_param
    
    def compute_NLL(self, x0, y, r, z):
        r = r.reshape(r.size(0), -1, self.dim_hidden)
        z = z.reshape(z.size(0), -1, self.dim_hidden)
        dec_param = r + self.param_MLP(z)
            
        dec_param = self.layernorm(dec_param).permute(0,2,1)
        dec_param = dec_param.reshape(self.batch_size, self.dim_hidden, -1)
            
        logits = torch.matmul(x0, dec_param)
            
        y_dist = torch.distributions.one_hot_categorical.OneHotCategorical(logits=logits)
        NLL = -y_dist.log_prob(y).sum(-1).mean() 
        accur = ((y.max(-1).indices == logits.max(-1).indices).sum(-1) * 100.0 / y.size(1)).mean()
        return NLL, accur
               
                  
    def forward(self, Cx, Cy, Tx, Ty):
        c0 = self.x_emb(Cx)
        t0 = self.x_emb(Tx)
        
        r_c = self.param_encoder(c0, Cy, 'deterministic')
        z_c_param = self.param_encoder(c0, Cy, 'stochastic')
        z_c_dist = self.stochastic_distribution(z_c_param)
            
        if self.training == False:
            NLL, measure = self.compute_NLL(t0, Ty, r_c, z_c_dist.mean)
            return NLL, measure    
        
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
            NLL, measure = self.compute_NLL(t0, Ty, r_t_hat, z_t_hat)
            return KL, NLL, measure
        
        elif self.phase == 'finetuning':
            z_t_param = self.param_encoder(t0, Ty, 'stochastic')
            z_t_dist = self.stochastic_distribution(z_t_param)
            z_t = self.reparameterize(z_t_dist)
            
            KL = torch.distributions.kl.kl_divergence(z_t_dist, z_c_dist).sum(-1).mean()
            NLL, measure = self.compute_NLL(t0, Ty, r_c, z_t)
            return KL, NLL, measure
