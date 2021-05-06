import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import glob

from dataset import S3DIS,PROX
from encoder import DGCNN_semseg
from cvae import BPS_CVAE

import argparse

parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
parser.add_argument('--num_points', type=int, default=4096,
                        help='num of points to use')
parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


lis = glob.glob("./dataset/PROXD_cleaned/BasementSittingBooth*")
print(lis)
dataset = PROX(frames_lis=lis)
train_loader = DataLoader(dataset,batch_size=8,num_workers=2,drop_last=True)

Pointnet_encoder = DGCNN_semseg(args,only_encode=True).to(device)
Pointnet_encoder = nn.DataParallel(Pointnet_encoder)
Pointnet_encoder.load_state_dict(torch.load("models/model_1.t7"))
Pointnet_encoder.eval()

net = BPS_CVAE(n_dim_scene_feat_beta=1024+16+3,n_dim_body=106,F_hs_size=1024,latent_size = 512).to(device)
 
net.load_state_dict(torch.load("models/net_scene.t7"))
net.eval()


for i, data in enumerate(train_loader, 0):
    if i>20:
        break
    scene,M_o,t_beta_r = data
    scene,M_o,t_beta_r = scene.to(device),M_o.to(device),t_beta_r.to(device)
    scene = scene.permute(0, 2, 1)
    scene_feat = Pointnet_encoder(scene)

    scene_feat_beta = torch.cat([scene_feat,t_beta_r],1)
    # scene_feat_beta = scene_feat_beta.repeat(8,1)
    pred_Mo = net.sample(scene_feat_beta.shape[0],scene_feat_beta)
    np.save("data2/"+str(i)+"Pred",pred_Mo.cpu().detach().numpy())
    np.save("data2/"+str(i)+"Gt",M_o.cpu().detach().numpy())
    # np.save("data/"+str(i)+"_" + name + "_scene",scene.cpu().detach().numpy())
