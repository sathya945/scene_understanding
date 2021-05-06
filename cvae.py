import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResBlock(nn.Module):
    def __init__(self, dim=512):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(in_features=dim, out_features=dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(in_features=dim, out_features=dim)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += residual
        out = F.relu(out)
        return out

class BPS_CVAE(nn.Module):
    def __init__(self,n_dim_scene_feat_beta=1024+16,n_dim_body=72,F_hs_size=1024,latent_size = 512):
        super(BPS_CVAE, self).__init__()
        self.fc1 = nn.Linear(in_features=n_dim_scene_feat_beta, out_features=F_hs_size)
        self.bn1 = nn.BatchNorm1d(F_hs_size)
        self.res1 = ResBlock(dim=n_dim_body)
        self.res2 = ResBlock(dim=n_dim_body)

        # latent vec
        self.mu_fc = nn.Linear(in_features=F_hs_size+n_dim_body,out_features=latent_size)
        self.logvar_fc = nn.Linear(in_features=F_hs_size+n_dim_body,out_features=latent_size)
        
        # decoder
        self.res3 = ResBlock(dim=latent_size+F_hs_size)
        self.res4 = ResBlock(dim=latent_size+F_hs_size)
        self.output_fc = nn.Linear(in_features=latent_size+F_hs_size,out_features=n_dim_body)
        self.latent_size = latent_size

    def _sampler(self, mu, logvar):
        var = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(var.size()).normal_().to(device)
        return eps.mul(var).add_(mu)

    def forward(self, M_o,scene_feat_beta):  #F_hs = scenefeature +  shape, location and orientation inputs {β, t, r}
        
        #encoder
        F_hs = F.relu(self.bn1(self.fc1(scene_feat_beta)))
        x = F.relu(self.res1(M_o))
        x = F.relu(self.res2(x))
        x = torch.cat([x,F_hs],dim=-1)
        
        #latent
        mu = self.mu_fc(x)
        logvar = self.logvar_fc(x)
        
        #sample
        x = self._sampler(mu, logvar)
        
        #decoder
        x = torch.cat([x,F_hs],dim=-1)
        output = self.output_fc(F.relu(self.res4(F.relu(self.res3(x)))))
        return output,mu,logvar


    def sample(self, batch_size, scene_feat_beta):
        F_hs = F.relu(self.bn1(self.fc1(scene_feat_beta)))
        x = torch.randn([batch_size,self.latent_size], dtype=torch.float32).to(device)
        x = torch.cat([x,F_hs],dim=-1)
        output = self.output_fc(F.relu(self.res4(F.relu(self.res3(x)))))
        return output


    # def interpolate(self, scene_feat, interpolate_len=5):
    #     eps_start = torch.randn([1, self.n_bps_feat, self.eps_d], dtype=torch.float32).to(device)
    #     eps_end = torch.randn([1, self.n_bps_feat, self.eps_d], dtype=torch.float32).to(device)
    #     eps_list = [eps_start]

    #     for i in range(interpolate_len):
    #         cur_eps = eps_start + (i+1) * (eps_end - eps_start) / (interpolate_len+1)
    #         eps_list.append(cur_eps)
    #     eps_list.append(eps_end)

    #     gen_list = []
    #     for eps in eps_list:
    #         x = F.relu(self.bn3(self.fc3(eps)))  # eps_d-->512, [bs, 1, 512]
    #         x = F.relu(self.bn_resblc3(self.fc_resblc3(torch.cat([x, scene_feat[0]], dim=-1))))  # 1024-->512
    #         x = self.res_block3(x)  # [bs, 1, 512]
    #         x = F.relu(self.bn_resblc4(self.fc_resblc4(torch.cat([x, scene_feat[1]], dim=-1))))  # 1024-->512
    #         x = self.res_block4(x)  # [bs, 1, 512]
    #         x = F.relu(self.bn4(self.fc4(torch.cat([x, scene_feat[2]], dim=-1))))  # 1024-->1024 [bs, 1, 1024]
    #         sample = F.relu(self.bn5(self.fc5(x)))  # 1024 --> 10000, [bs, 1, 10000]
    #         gen_list.append(sample)
    #     return gen_list

