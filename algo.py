import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal

from func import SetTransformer
from module import MLP, Prior, Encoder, Decoder


class NP(nn.Module):
    def __init__(self, x_dim, hid_dim, r_dim, z_dim, y_dim,
                 lr, clipping, model):
        super(NP, self).__init__()
        
        self.x_dim, self.y_dim = x_dim, y_dim
        self.model = model

        self.z_encoder = Encoder(x_dim, hid_dim, z_dim, y_dim, 
                                 is_stochastic=True, 
                                 is_attention='ANP' in self.model,
                                 is_multiply='mul' in self.model)
        
        self.decoder = Decoder(x_dim, hid_dim, r_dim, z_dim, y_dim,
                               is_attention='ANP' in self.model,
                               is_multiply='mul' in self.model)
        
        self.optim = optim.Adam(self.parameters(), lr=lr)
        self.loss = {'KL':[], 'NLL':[]}
        self.clipping = clipping
        
    def forward(self, C, T, update_, return_, KL=None, NLL=None):
        if KL is None and NLL is None:
            Tx, Ty = T[:,:,:self.x_dim], T[:,:,-self.y_dim:]
            num_target = Tx.size(1)

            z_prior, z_p, _ = self.z_encoder.calc_latent(C, num_target, 
                                                         Tx=Tx if 'ANP' in self.model else None)
            z_posterior, z_q, _ = self.z_encoder.calc_latent(T, num_target, 
                                                             Tx=Tx if 'ANP' in self.model else None)

            KL = kl_divergence(z_posterior, z_prior).sum(dim=-1).mean()

            z = z_q if self.training else z_p
            
            Ty_dist, Ty_hat = self.decoder.calc_y(C, Tx, z)

            NLL = -Ty_dist.log_prob(Ty).sum(dim=-1).mean()

        NP_loss = KL + NLL
            
        if update_:
            self.optim.zero_grad()
            NP_loss.backward()
            if self.clipping:
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optim.step()       
            
            self.loss['KL'].append(KL.item())
            self.loss['NLL'].append(NLL.item())
        
        if return_:
            T_MSE = F.mse_loss(Ty, Ty_hat).item()
            return T_MSE, Ty_dist


class FELD(nn.Module):
    def __init__(self, x_dim, hid_dim, r_dim, z_dim, y_dim,
                 lr, clipping):
        super(FELD, self).__init__()

        self.x_dim, self.y_dim = x_dim, y_dim
        self.hid_dim = hid_dim

        self.x_MLP = nn.Sequential(nn.Linear(x_dim, hid_dim),
                                   nn.ReLU(),
                                   nn.Linear(hid_dim, hid_dim))

        fea_dim = hid_dim * y_dim * 2
        self.r_ST = SetTransformer(hid_dim + y_dim, 1, fea_dim)
        self.z_ST = SetTransformer(hid_dim + y_dim, 1, fea_dim * 2)

        self.W_MLP = nn.Sequential(nn.Linear(hid_dim, hid_dim),
                                   nn.ReLU(),
                                   nn.Linear(hid_dim, hid_dim))

        self.layernorm = nn.LayerNorm(hid_dim)

        self.optim = optim.Adam(self.parameters(), lr=lr)
        self.loss = {'KL':[], 'NLL':[]}
        self.clipping = clipping

    def _calc_latent(self, set_, is_stoc):
        x, y = set_[:,:,:self.x_dim], set_[:,:,-self.y_dim:]
        x0 = self.x_MLP(x)
        set_ = torch.cat([x0, y], dim=-1)

        if not is_stoc:
            lat = self.r_ST(set_)
            return lat
        else:
            param = self.z_ST(set_)
            mu, omega = torch.chunk(param, chunks=2, dim=-1)
            sigma = 0.1 + 0.9 * torch.sigmoid(omega)

            dist = Normal(mu, sigma)
            lat = dist.rsample() if self.training else dist.mean
            return lat, dist

    def _calc_W(self, r, z):
        r = r.reshape(-1, self.y_dim * 2, self.hid_dim)
        z = z.reshape(-1, self.y_dim * 2, self.hid_dim)
        W = r + self.W_MLP(z)
        W = self.layernorm(W).permute(0,2,1)
        return W

    def _calc_y(self, t0, W):
        param = torch.matmul(t0, W)
        mu, omega = torch.chunk(param, chunks=2, dim=-1)
        sigma = 0.1+0.9*F.softplus(omega)

        dist = Normal(mu, sigma)
        pred = dist.rsample() if self.training else dist.mean
        return dist, pred

    def forward(self, C, T, update_, return_, KL=None, NLL=None):
        if KL is None and NLL is None:
            rc = self._calc_latent(C, False)
            zc, zc_dist = self._calc_latent(C, True)

            Tx, Ty = T[:,:,:self.x_dim], T[:,:,-self.y_dim:]
            t0 = self.x_MLP(Tx)
            if self.training:
                zt, zt_dist = self._calc_latent(T, True)
                KL = kl_divergence(zt_dist, zc_dist).sum(dim=-1).mean()

                Wct = self._calc_W(rc, zt)
                Ty_dist, Ty_hat = self._calc_y(t0, Wct)
            else:
                KL = 0.0

                Wcc = self._calc_W(rc, zc)
                Ty_dist, Ty_hat = self._calc_y(t0, Wcc)

            NLL = -Ty_dist.log_prob(Ty).sum(dim=-1).mean()

        FELD_loss = KL + NLL

        if update_:
            self.optim.zero_grad()
            FELD_loss.backward()
            if self.clipping:
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optim.step()       
            
            self.loss['KL'].append(KL.item() if isinstance(KL, torch.Tensor) else KL)
            self.loss['NLL'].append(NLL.item() if isinstance(NLL, torch.Tensor) else NLL)

        if return_:
            T_MSE = F.mse_loss(Ty, Ty_hat).item()
            return T_MSE, Ty_dist


class MAHA(FELD):
    def __init__(self, *args, phase='pretrain', **kwargs):
        super(MAHA, self).__init__(*args, **kwargs)

        self.phase = phase
        self.loss = {'KL':[], 'NLL':[], 'aux_NLL':[], 'aux_ENT':[]}

    def forward(self, C, T, update_, return_, KL=None, NLL=None):
        if KL is None and NLL is None:
            rc = self._calc_latent(C, False)
            zc, zc_dist = self._calc_latent(C, True)

            Tx, Ty = T[:,:,:self.x_dim], T[:,:,-self.y_dim:]
            t0 = self.x_MLP(Tx)
            if self.training:
                zt, zt_dist = self._calc_latent(T, True)
                
                if self.phase == 'pretrain':
                    KL = kl_divergence(zt_dist, zc_dist).sum(dim=-1).mean()
                    aux_NLL = 0.0
                    aux_ENT = 0.0

                    rt = self._calc_latent(T, False)
                    rt = rt.mean(dim=0, keepdim=True)

                    Wtt = self._calc_W(rt, zt)
                    Ty_dist, Ty_hat = self._calc_y(t0, Wtt)

                elif self.phase == 'finetune':
                    KL = 0.0

                    Wct = self._calc_W(rc, zt.detach())
                    Ty_dist, Ty_hat = self._calc_y(t0.detach(), Wct)

                    with torch.no_grad():
                        rt = self._calc_latent(T, False)
                        Wtt = self._calc_W(rt, zt.detach())
                        T_dist, _ = self._calc_y(t0.detach(), Wtt)
                    
                    aux_NLL = -T_dist.log_prob(Ty_hat).sum(dim=-1).mean()
                    aux_ENT = -(Ty_hist.loc * torch.log(Ty_hist.loc + 1e-10)).sum(dim=1).mean()

            else:
                KL = 0.0
                aux_NLL = 0.0
                aux_ENT = 0.0

                Wcc = self._calc_W(rc, zc)
                Ty_dist, Ty_hat = self._calc_y(t0, Wcc)

            NLL = -Ty_dist.log_prob(Ty).sum(dim=-1).mean()

        MAHA_loss = KL + NLL + aux_NLL - aux_ENT

        if update_:
            self.optim.zero_grad()
            MAHA_loss.backward()
            if self.clipping:
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optim.step()
            
            self.loss['KL'].append(KL.item() if isinstance(KL, torch.Tensor) else KL)
            self.loss['NLL'].append(NLL.item() if isinstance(NLL, torch.Tensor) else NLL)
            self.loss['aux_NLL'].append(aux_NLL.item() if isinstance(aux_NLL, torch.Tensor) else aux_NLL)
            self.loss['aux_ENT'].append(aux_ENT.item() if isinstance(aux_ENT, torch.Tensor) else aux_ENT)

        if return_:
            T_MSE = F.mse_loss(Ty, Ty_hat).item()
            return T_MSE, Ty_dist

