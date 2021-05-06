import smplx
import numpy as np
import json
import open3d as o3d
import torch
from pytorch3d import transforms
import math



from utils.convert_to_world import *



model = smplx.create('dataset/models', model_type='smplx',
						 gender='male', ext='npz',
						 num_pca_comps=12,
						 create_global_orient=True,
						 create_body_pose=True,
						 create_betas=True,
						 create_left_hand_pose=True,
						 create_right_hand_pose=True,
						 create_expression=True,
						 create_jaw_pose=True,
						 create_leye_pose=True,
						 create_reye_pose=True,
						 create_transl=True
						 )

arr1 = np.load("Gt.npy")
arr2 = np.load("Pred.npy")

arr1 = vec_to_org(torch.from_numpy(arr1))
arr2 = vec_to_org(torch.from_numpy(arr2))

for i in range(arr1.shape[0]):
		body1 = o3d.geometry.TriangleMesh()
		body2 = o3d.geometry.TriangleMesh()

		vec = arr1[i:i+1,:]
		output = gen_mesh_from_vec(vec,model)
		vertices = output.vertices.detach().cpu().numpy().squeeze()
		body1.vertices = o3d.utility.Vector3dVector(vertices)
		body1.triangles = o3d.utility.Vector3iVector(model.faces)
		body1.vertex_normals = o3d.utility.Vector3dVector([])
		body1.triangle_normals = o3d.utility.Vector3dVector([])


		vec = arr2[i:i+1,:]
		output = gen_mesh_from_vec(vec,model)
		vertices = output.vertices.detach().cpu().numpy().squeeze()
		body2.vertices = o3d.utility.Vector3dVector(vertices)
		body2.triangles = o3d.utility.Vector3iVector(model.faces)
		body2.vertex_normals = o3d.utility.Vector3dVector([])
		body2.triangle_normals = o3d.utility.Vector3dVector([])

		o3d.visualization.draw_geometries([body1,body2])