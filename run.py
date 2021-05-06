import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from torch.utils.data import DataLoader

from dataset import S3DIS,PROX
from encoder import DGCNN_semseg

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


# dataset = S3DIS(num_points=4096)
dataset = PROX(sample_scene_points=4096*3)



train_loader = DataLoader(dataset,batch_size=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = DGCNN_semseg(args).to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load("./models/model_1.t7"))

model.eval()

# data, seg = dataset.__getitem__(0)
data, seg  = next(iter(train_loader))




# print(data[:,3:6])
data, seg = data.to(device), seg.to(device)
data = data.permute(0, 2, 1)
seg_pred = model(data)
seg_pred = seg_pred.permute(0, 2, 1).contiguous()
pred = seg_pred.max(dim=2)[1]
# seg_np = seg.cpu().numpy()
pred_np = pred.cpu().numpy()
data_np = data.cpu().numpy()
np.save("data",data_np)
# np.save("seg",seg_np)
np.save("pred",pred_np)

    # parser.add_argument('--num_points', type=int, default=4096,
    #                     help='num of points to use')
    # parser.add_argument('--dropout', type=float, default=0.5,
    #                     help='dropout rate')
    # parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
    #                     help='Dimension of embeddings')
    # parser.add_argument('--k', type=int, default=20, metavar='N',
    #                     help='Num of nearest neighbors to use')
    # parser.add_argument('--model_root', t1ype=str, default='', metavar='N',
    #                     help='Pretrained model root')
