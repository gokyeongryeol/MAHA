import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from module import *
from image_embedding import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
class MAHA(nn.Module):
    def __init__(self, dim_input, ln, phase, batch_size, num_inds, dim_hidden=128, num_heads=4):
        super(MAHA, self).__init__()
        self.dim_hidden = dim_hidden
        self.phase = phase
        self.batch_size = batch_size
                
        self.feature_extractor = resnet18()
        self.x_emb = nn.Sequential(self.feature_extractor,
                                   nn.Linear(512, dim_hidden))
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
        class_param = torch.tensor([]).to(device)
        for class_idx in range(y.size(-1)):
            index = class_idx == y.max(-1).indices
            inp = x[index].reshape(1, -1, self.dim_hidden)
            if kind == 'deterministic':
                rep = self.r_ST(inp)
            elif kind == 'stochastic':
                rep = self.z_ST(inp)
            class_param = torch.cat((class_param, rep), 1)
        return class_param                    
        
    def compute_NLL(self, x0, y, r, z):
        r = r.reshape(1, -1, self.dim_hidden)
        z = z.reshape(1, -1, self.dim_hidden)
        dec_param = r + self.param_MLP(z)
        
        dec_param = self.layernorm(dec_param).permute(0,2,1)
        dec_param = dec_param.squeeze()
        logits = torch.matmul(x0, dec_param)
            
        y_dist = torch.distributions.one_hot_categorical.OneHotCategorical(logits=logits)
        NLL = -y_dist.log_prob(y).sum(-1).mean() 
        accur = ((y.max(-1).indices == logits.max(-1).indices).sum(-1) * 100.0 / y.size(0))
        return NLL, accur
    
    def way_pool(self, param_list):
        param_list = [param_list[itr].mean(1, True) for itr in range(len(param_list))]
        return param_list
        
    def way_augment(self, y):
        current = [y_b.size(-1) for y_b in y]
        max_len = max(current)
        total = [max_len] * len(current) 
        share = [a_i // b_i for a_i, b_i in zip(total, current)]
        remainder = [a_i % b_i for a_i, b_i in zip(total, current)]
        index_list = [list(np.arange(a_i)) * b_i + list(np.arange(c_i)) 
                      for a_i, b_i, c_i in zip(current, share, remainder)]
        return index_list
    
    def forward(self, Cx, Cy, Tx, Ty):
        c0_list, t0_list = [], []
        r_c_list, r_t_list, z_c_param_list, z_t_param_list = [], [], [], []
        total_NLL, total_measure = torch.tensor([0.0]).to(device), torch.tensor([0.0]).to(device)
        for batch_idx in range(self.batch_size):
            c0 = self.x_emb(Cx[batch_idx])
            t0 = self.x_emb(Tx[batch_idx])
            c0_list.append(c0)
            t0_list.append(t0)
            
            r_c = self.param_encoder(c0, Cy[batch_idx], 'deterministic')
            r_t = self.param_encoder(t0, Ty[batch_idx], 'deterministic')
            r_c_list.append(r_c)
            r_t_list.append(r_t)
            
            z_c_param = self.param_encoder(c0, Cy[batch_idx], 'stochastic')
            z_t_param = self.param_encoder(t0, Ty[batch_idx], 'stochastic')
            z_c_param_list.append(z_c_param)
            z_t_param_list.append(z_t_param)
            
            z_c_dist = self.stochastic_distribution(z_c_param)
            
            NLL, measure = self.compute_NLL(t0, Ty[batch_idx], r_c, z_c_dist.mean)
            total_NLL += NLL
            total_measure += measure
            
        if self.training == False:
            return total_NLL / self.batch_size, total_measure / self.batch_size
        
        if self.phase == 'pretrain':
            index_list = self.way_augment(Ty)
            index_list = torch.tensor(index_list).to(device)
            
            r_t_list = torch.stack([r_t_list[itr].squeeze()[index_list[itr]] for itr in range(self.batch_size)])
            r_t_list = r_t_list.mean(0, True)
            r_t_list = [r_t_list[:,:Ty[itr].size(-1),:] for itr in range(self.batch_size)]
            
            z_t_param_list = self.way_pool(z_t_param_list)
            z_c_param_list = self.way_pool(z_c_param_list)
            
        total_KL, total_NLL, total_measure = torch.tensor([0.0]).to(device), torch.tensor([0.0]).to(device), torch.tensor([0.0]).to(device)
        for batch_idx in range(self.batch_size):
            z_t_dist = self.stochastic_distribution(z_t_param_list[batch_idx])
            z_c_dist = self.stochastic_distribution(z_c_param_list[batch_idx])
            KL = torch.distributions.kl.kl_divergence(z_t_dist, z_c_dist).sum(-1).mean()
            total_KL += KL
                        
            z_t = self.reparameterize(z_t_dist)

            if self.phase == 'pretrain':    
                NLL, measure = self.compute_NLL(t0_list[batch_idx], Ty[batch_idx], r_t_list[batch_idx], z_t)
                
            elif self.phase == 'finetuning':
                NLL, measure = self.compute_NLL(t0_list[batch_idx], Ty[batch_idx], r_c_list[batch_idx], z_t)
            
            total_NLL += NLL
            total_measure += measure
            
        return total_KL / self.batch_size, total_NLL / self.batch_size, total_measure / self.batch_size