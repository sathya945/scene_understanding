import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn


from dataset import S3DIS,PROX
from encoder import DGCNN_semseg
from cvae import BPS_CVAE

import argparse
import glob

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

lis = glob.glob("./dataset/PROXD_cleaned/*")

# lis = [x for x in lis if "N3OpenArea" not in x]
print(len(lis),"Training List")
print(lis)
dataset = PROX(path="./dataset/",frames_lis=lis)
train_loader = DataLoader(dataset,batch_size=8,num_workers=2,drop_last=True)



lis = glob.glob("./dataset/PROXD_cleaned/N3OpenArea*")
print(lis)
print(len(lis),"Testing List")
dataset_test = PROX(frames_lis=lis)
test_loader = DataLoader(dataset_test,batch_size=8,num_workers=1,drop_last=True)




Pointnet_encoder = DGCNN_semseg(args,only_encode=True).to(device)
Pointnet_encoder = nn.DataParallel(Pointnet_encoder)
Pointnet_encoder.load_state_dict(torch.load("models/model_1.t7"))
Pointnet_encoder.eval()


net = BPS_CVAE(n_dim_scene_feat_beta=1024+16+3,n_dim_body=106,F_hs_size=1024,latent_size = 512).to(device)
# net = nn.DataParallel(net)


mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

min_test_loss = 1e+8


def test():
    loss = 0
    i = 0
    net.eval()
    for i, data in enumerate(test_loader, 0):
        scene,M_o,t_beta_r = data
        scene,M_o,t_beta_r = scene.to(device),M_o.to(device),t_beta_r.to(device)
        scene = scene.permute(0, 2, 1)
        scene_feat = Pointnet_encoder(scene)
        scene_feat_beta = torch.cat([scene_feat,t_beta_r],1)
        pred_Mo = net.sample(scene_feat_beta.shape[0],scene_feat_beta)
        recon_loss = mse_loss(pred_Mo,M_o)
        loss = loss + recon_loss.item()
        if i<20:
            np.save("data/"+str(i)+"Pred",pred_Mo.cpu().detach().numpy())
            np.save("data/"+str(i)+"Gt",M_o.cpu().detach().numpy())
    loss = loss/i
    global min_test_loss
    print(loss)
    if loss < min_test_loss:
        min_test_loss = loss
        torch.save(net.state_dict(),'models/net_scene.t7')
        print("saving model",flush=True)


for epoch in range(40):
    running_loss = 0
    for i, data in enumerate(train_loader,0):
        scene,M_o,t_beta_r = data
        scene,M_o,t_beta_r = scene.to(device),M_o.to(device),t_beta_r.to(device)
        scene = scene.permute(0, 2, 1)
        scene_feat = Pointnet_encoder(scene)
        # print(M_o.shape)
        scene_feat_beta = torch.cat([scene_feat,t_beta_r],1)
        # print(scene_feat_beta.shape,M_o.shape)
        output,mu,log_var = net(M_o,scene_feat_beta)

        kl = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        recon_loss = mse_loss(output,M_o)
        # print(output.shape)
        loss = kl + recon_loss

        # zero the parameter gradients
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            # running_loss = 0.0
            # scene_feat = Pointnet_encoder(scene)
            # scene_feat_beta = torch.cat([scene_feat,t_beta_r],1)
            # pred_Mo = net.sample(scene_feat_beta.shape[0],scene_feat_beta)

            # np.save("Pred",pred_Mo.cpu().detach().numpy())
            # np.save("Gt",M_o.cpu().detach().numpy())
            # torch.save(net.state_dict(),'models/net_scene.t7')
            # print("",flush=True)
            # optimizer.zero_grad()
            if i%5000 == 4999 or i == len(train_loader)-1:
                with torch.no_grad():
                    test()
            net.train()
            running_loss =0




