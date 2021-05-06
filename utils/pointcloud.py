import numpy as np
import trimesh

def bounding_box(points,cen):
	min_x,min_y,min_z = cen-1
	max_x,max_y,max_z = cen+1
	bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
	bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)
	bound_z = np.logical_and(points[:, 2] > min_z, points[:, 2] < max_z)
	bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)
	return points[bb_filter,:]


def sample(data, num_sample=4096):
	idx = np.random.choice(data.shape[0], num_sample,replace=True)
	return data[idx]

def sample_and_crop(data,pos,num_sample=4096):
	data = bounding_box(data,pos)
	# if data.shape[0] <1000:
	# 	return  False
	data = sample(data,num_sample)
	max_room_x = max(data[:,0])
	max_room_y = max(data[:,1])
	max_room_z = max(data[:,2])
	new_data = np.ones((data.shape[0],9))
	new_data[:,:6] = data
	data = new_data
	data[:,6:] =  data[:,:3]-data[:,:3].min(axis=0)
	data[:,6:] = data[:,6:]/data[:,6:].max(axis=0)
	return data

# a = trimesh.load("./MPH1Library.ply")
# data = np.zeros((a.vertices.shape[0],6))
# data[:,:3] = np.asarray(a.vertices)
# data[:,3:6] = a.visual.vertex_colors[:,:3]/255


def load_ply(file):
	a = trimesh.load(file)
	# print(a)
	data = np.zeros((a.vertices.shape[0],6))
	data[:,:3] = np.asarray(a.vertices)
	data[:,3:6] = a.visual.vertex_colors[:,:3]/255
	return data
