import os
import sys
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.pointcloud import *
import trimesh
import pickle
import smplx
import json
import random

def load_data_semseg(file="ply_data_all_0.h5"):
    f = h5py.File(file, 'r+')
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    return data , label


class S3DIS(Dataset):
    def __init__(self, num_points=4096):
        self.data, self.seg = load_data_semseg()
        self.num_points = num_points

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        seg = self.seg[item][:self.num_points]
        seg = torch.LongTensor(seg)
        return pointcloud, seg

    def __len__(self):
        return self.data.shape[0]

# class PROX(Dataset):
#     def __init__(self, num_points=4096):
#         self.data = load_ply("/home/venkata.sathya/scene_understanding/MPH1Library.ply")
#         self.num_points = num_points
#     def __getitem__(self,pos=None):
#         pos = self.data[np.random.random_integers(self.data.shape[0]),:3]
#         cropped_data    = sample_and_crop(self.data,pos,self.num_points)
#         return cropped_data.astype(np.float32),np.random.randint(10,cropped_data.shape[0])
#     def __len__(self):
#         return 100




class PROX():
    def __init__(self,path="./dataset/",sample_scene_points=4096,frames_lis=glob.glob("./dataset/PROXD_cleaned/*")):
        self.path = path
        self.load_scenes()
        
        arr_lis = []
        brr_lis = []
        for file in frames_lis:
            print(file)
            arr = np.load(file)
            name = file.split("/")[-1].split("_")[0]
            scene_id = self.scene_id[name]
            brr = np.ones((arr.shape[0],1))*scene_id
            arr_lis.append(arr)
            brr_lis.append(brr)


        self.data = np.vstack(arr_lis)
        self.data_sceneid = np.vstack(brr_lis).flatten().astype(np.int)
        self.sample_scene_points =sample_scene_points
    
    def load_scenes(self):
        scene_path = self.path+"scenes/*"
        self.scene_cloud = []
        self.scene_id = {}
        idx = 0 
        for f in glob.glob(scene_path):
            name = f.split("/")[-1].split(".")[0]            # scene_feat = Pointnet_encoder(scene)
            # scene_feat_beta = torch.cat([scene_feat,t_beta_r],1)
            # pred_Mo = net.sample(scene_feat_beta.shape[0],scene_feat_beta)

            # np.save("Pred",pred_Mo.cpu().detach().numpy())
            # np.save("Gt",M_o.cpu().detach().numpy())
            # torch.save(net.state_dict(),'models/net_scene.t7')

            print("Loading "+name)
            data = load_ply(f)
            self.scene_cloud.append(data)
            self.scene_id[name] = idx
            idx = idx + 1
        print("Scenes Loaded")
    
    def __getitem__(self,i):
        i = np.random.randint(self.data.shape[0])
        body_params = self.data[i]
        transl = body_params[0:3]
        t_beta_r = body_params[0:19]
        try :
            scene =  sample_and_crop(self.scene_cloud[self.data_sceneid[i]],transl,self.sample_scene_points)
        except :
            # print("ERROR taking other sample")
            return self.__getitem__(i-1)
        return scene.astype(np.float32),body_params.astype(np.float32),t_beta_r.astype(np.float32)
    def __len__(self):
        return self.data.shape[0]

# class PROX_img():
#      def __init__(self,path="./dataset/",sample_scene_points=4096,frames_lis=glob.glob("./dataset/PROXD_cleaned/*")):
#         self.path = path
#         self.load_scenes()
        
#         arr_lis = []
#         brr_lis = []
#         for file in frames_lis:
#             print(file)
#             arr = np.load(file)
#             name = file.split("/")[-1].split("_")[0]
#             scene_id = self.scene_id[name]
#             brr = np.ones((arr.shape[0],1))*scene_id
#             arr_lis.append(arr)
#             brr_lis.append(brr)


#         self.data = np.vstack(arr_lis)
#         self.data_sceneid = np.vstack(brr_lis).flatten().astype(np.int)
#         self.sample_scene_points =sample_scene_points
#     def load_scenes(self):
#         scene_path = self.path+"scenes/*"
#         self.scene_cloud = []
#         self.scene_id = {}
#         idx = 0 
#         for f in glob.glob(scene_path):
#             name = f.split("/")[-1].split(".")[0]            # scene_feat = Pointnet_encoder(scene)
#             # scene_feat_beta = torch.cat([scene_feat,t_beta_r],1)
#             # pred_Mo = net.sample(scene_feat_beta.shape[0],scene_feat_beta)

#             # np.save("Pred",pred_Mo.cpu().detach().numpy())
#             # np.save("Gt",M_o.cpu().detach().numpy())
#             # torch.save(net.state_dict(),'models/net_scene.t7')

#             print("Loading "+name)
#             data = load_ply(f)
#             self.scene_cloud.append(data)
#             self.scene_id[name] = idx
#             idx = idx + 1
#         print("Scenes Loaded")
    
#     def __getitem__(self,i):
#         i = np.random.randint(self.data.shape[0])
#         body_params = self.data[i]
#         transl = body_params[0:3]
#         t_beta_r = body_params[0:19]
#         try :
#             scene =  sample_and_crop(self.scene_cloud[self.data_sceneid[i]],transl,self.sample_scene_points)
#         except :
#             # print("ERROR taking other sample")
#             return self.__getitem__(i-1)
#         return scene.astype(np.float32),body_params.astype(np.float32),t_beta_r.astype(np.float32)
#     def __len__(self):
#         return self.data.shape[0]



