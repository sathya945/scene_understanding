import os
import os.path as osp
import cv2
import numpy as np
import json

import torch
import pickle
from typing import NewType, List, Union
import torch.nn.functional as F

Tensor = NewType('Tensor', torch.Tensor)



#https://github.com/vchoutas/smplx/blob/f4206853a4746139f61bdcf58571f2cea0cbebad/smplx/utils.py#L119 
def batch_rodrigues(
    rot_vecs: Tensor,
    epsilon: float = 1e-8
) -> Tensor:
    assert len(rot_vecs.shape) == 2, (
        f'Expects an array of size Bx3, but received {rot_vecs.shape}')

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device
    dtype = rot_vecs.dtype

    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True, p=2)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def batch_rot2aa(
    Rs: Tensor, epsilon: float = 1e-7
) -> Tensor:
    cos = 0.5 * (torch.einsum('bii->b', [Rs]) - 1)
    cos = torch.clamp(cos, -1 + epsilon, 1 - epsilon)

    theta = torch.acos(cos)

    m21 = Rs[:, 2, 1] - Rs[:, 1, 2]
    m02 = Rs[:, 0, 2] - Rs[:, 2, 0]
    m10 = Rs[:, 1, 0] - Rs[:, 0, 1]
    denom = torch.sqrt(m21 * m21 + m02 * m02 + m10 * m10 + epsilon)

    axis0 = torch.where(torch.abs(theta) < 0.00001, m21, m21 / denom)
    axis1 = torch.where(torch.abs(theta) < 0.00001, m02, m02 / denom)
    axis2 = torch.where(torch.abs(theta) < 0.00001, m10, m10 / denom)

    return theta.unsqueeze(1) * torch.stack([axis0, axis1, axis2], 1)



def get_vec(file_path):
    f = open(file_path,'rb')
    data = pickle.load(f)
    vec = np.concatenate((data['transl'][0], data['global_orient'][0], data['betas'][0],
                                                      data['body_pose'][0],
                                                      data['left_hand_pose'][0], data['right_hand_pose'][0]), axis=0) 
    # print(vec.shape)
    return torch.from_numpy(vec).reshape(1,-1)

def transform(vec,trans,model):
	torch_param = {}
	torch_param['betas'] = vec[:,6:16]
	output = model(return_verts=True, **torch_param)
	p = output.joints[0,0,:].detach().numpy()

	t_wc = trans[0:3,3]
	R_wc = trans[0:3,0:3] 		
	t_c = vec[0,0:3].numpy()
	R_c = batch_rodrigues(vec[:,3:6]).numpy()
	t_w = (R_wc@(p+t_c)+t_wc-p)
	R_w = R_wc@R_c
	vec[:,0:3] = torch.from_numpy(t_w)
	vec[:,3:6] = batch_rot2aa(torch.from_numpy(R_w))
	return vec

def gen_mesh_from_vec(vec,model):
	# vec = torch.from_numpy(vec).reshape(1,-1)
	torch_param = {}
	torch_param["transl"] = vec[:,0:3]
	torch_param["global_orient"] = vec[:,3:6]
	torch_param["betas"] = vec[:,6:16]
	torch_param["body_pose"] = vec[:,16:79]
	torch_param["left_hand_pose"] = vec[:,79:91]
	torch_param["right_hand_pose"] = vec[:,91:103]
	output = model(return_verts=True, **torch_param)
	return output

def rotation_matrix_to_cont_repr(x: Tensor) -> Tensor:
    assert len(x.shape) == 3, (
        f'Expects an array of size Bx3x3, but received {x.shape}')
    return x[:, :3, :2].reshape(-1,6)


def cont_repr_to_rotation_matrix(
    x: Tensor
) -> Tensor:
    ''' Converts tensor in continous representation to rotation matrices
    '''
    batch_size = x.shape[0]
    reshaped_input = x.view(-1, 3, 2)

    # Normalize the first vector
    b1 = F.normalize(reshaped_input[:, :, 0].clone(), dim=1)

    dot_prod = torch.sum(
        b1 * reshaped_input[:, :, 1].clone(), dim=1, keepdim=True)
    # Compute the second vector by finding the orthogonal complement to it
    b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=1)
    # Finish building the basis by taking the cross product
    b3 = torch.cross(b1, b2, dim=1)
    rot_mats = torch.stack([b1, b2, b3], dim=-1)
    # print(rot_mats.view(batch_size, 3, 3).shape)
    return rot_mats.view(batch_size, 3, 3)



def vec_to_cont(vec):
    # convert 'global ori' to continous representation
	new_vec = torch.zeros((vec.shape[0],106))
	new_vec[:,0:3] = vec[:,0:3]
	new_vec[:,9:106] = vec[:,6:103]
	new_vec[:,3:9] = rotation_matrix_to_cont_repr(batch_rodrigues(vec[:,3:6]))
	return new_vec
def vec_to_org(vec):
    # convert 'global ori' to original representation
	new_vec = torch.zeros((vec.shape[0],103))
	# print(vec.shape)
	new_vec[:,0:3] = vec[:,0:3]
	new_vec[:,6:103] = vec[:,9:106]
	new_vec[:,3:6] = batch_rot2aa(cont_repr_to_rotation_matrix(vec[:,3:9]))
	return new_vec