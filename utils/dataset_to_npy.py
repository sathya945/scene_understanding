import os
import os.path as osp
import torch
import pickle
import smplx

from typing import NewType, List, Union


from convert_to_world import *
import glob

def main(args,fitting_dir):
	recording_name = os.path.abspath(fitting_dir).split("/")[-1]
	fitting_dir = osp.join(fitting_dir, 'results')
	scene_name = recording_name.split("_")[0]
	print("scene_name")
	base_dir = args.base_dir
	cam2world_dir = osp.join(base_dir, 'cam2world')
	scene_dir = osp.join(base_dir, 'scenes')


	female_subjects_ids = [162, 3452, 159, 3403]
	subject_id = int(recording_name.split('_')[1])
	if subject_id in female_subjects_ids:
	    gender = 'female'
	else:
	    gender = 'male'

	with open(os.path.join(cam2world_dir, scene_name + '.json'), 'r') as f:
		trans = np.array(json.load(f))

	model = smplx.create(args.model_folder, model_type='smplx',
                         gender=gender, ext='npz',
                         num_pca_comps=args.num_pca_comps,
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
	lis = sorted(os.listdir(fitting_dir))

	arr = torch.zeros((len(lis),103))
	for i,img_name in enumerate(lis):
		try:
			vec = get_vec(osp.join(fitting_dir, img_name, '000.pkl'))
		except:
			arr[i,:] = arr[i-1,:]
			continue
		trans_vec = transform(vec,trans,model)
		# print(trans_vec.shape)
		arr[i,:] = trans_vec[0,:]
	arr  = vec_to_cont(arr)
	np.save("../dataset/PROXD_cleaned/"+recording_name, arr.numpy())
	print("recording_name compiled all dataset")


class Args:
    base_dir    = "/ssd_scratch/cvit/vivek/PROX_Dataset"
    model_folder="/ssd_scratch/cvit/vivek/PROX_Dataset/models"
    num_pca_comps = 12
    gender = 'male'

args = Args()


# fitting_dir = "../dataset/PROX_Dataset/PROXD/N3OpenArea_00157_01"
for i in glob.glob("/ssd_scratch/cvit/vivek/PROX_Dataset/PROXD/*"):
	print(i)
	main(args,i)



# def func(file):
# 	a = np.load(file)
# 	np.save(file,a[:,:-3])
# for i in glob.glob("./*"):
# 	func(i)