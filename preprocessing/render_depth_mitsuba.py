import argparse
import array
import numpy as np
#import Imath
import os
import open3d
from open3d import *
from itertools import combinations
import random
import cv2
import torch
import random
import time
#import imageio
import h5py
import argparse, sys, os

import projections.cameras_norm as cameras_norm
import projections.operations as linux_operations
import projections.global_variables as global_variables
import projections.point_cloud_operations as point_cloud_operations

from render_setting import *
import render_setting

class Reconstruction_Renderer_2:
    def __init__(self, args):
        self.gpu_ids = args.gpu_ids
        self.reso = args.reso

        self.intrinsics = cameras_norm.get_fixed_intrinsics()

        self.is_render_and_combine = args.render_and_combine

        grid_size=int(global_variables.GRID_SIZE)
        grid_pooling = torch.nn.MaxPool2d((grid_size, grid_size), stride=(grid_size, grid_size))
        self.grid_pooling  = linux_operations.init_net(grid_pooling, init_type='normal', init_gain=0.02, gpu_ids = [self.gpu_ids])

        self.infinite_depth = (torch.ones((self.reso*grid_size, self.reso*grid_size), dtype=torch.float)*(-20)).cuda(self.gpu_ids)
                
        self.num_fixed_views = args.num_views

        if args.isTrain:
            self.num_random_views = 8
        else:
            self.num_random_views = 1

        self.render_GT_views = args.render_GT_views
        
        self.setup_dir(args)
        
    def read_random_pose(self, chass_name, model_name):

        pose_file = self.pose_dir + '/%s/rgb/%s/random_poses.txt' % (chass_name, model_name)
        with open(pose_file) as file:
            pose_list_str = file.read().splitlines()
            
        pose_list=[]
        for s in pose_list_str:
            pose_s = s.split(',')
            pose = np.array([float(pose_s[0]), float(pose_s[1]), float(pose_s[2])])
            pose_list.append(cameras_norm.lookAt(pose))

        return pose_list

    def render_obj_coord_gt_depths_mitusba(self): # render obj coordinates, gt depths.
            
        height = self.reso
        width = self.reso

        with open(self.model_list_dir) as file:
            model_list = file.read().splitlines()

        #POOLING 
        #CONSTS.
        tensor_0=torch.cuda.FloatTensor(1).zero_().cuda(self.gpu_ids)
        tensor_1=(torch.cuda.FloatTensor(1).zero_()+1).cuda(self.gpu_ids)
    
        tensor_0_cpu = torch.FloatTensor(1).zero_()
        tensor_1_cpu = torch.FloatTensor(1).zero_()+1

        total_time = 0

        #
        model_id_start = 0

        min_dep = tensor_1 * 100
        max_dep = -tensor_1 * 100

        t=0
        #operations.
        render_gt_views = linux_operations.Renderer_VIEWS_FROM_POINT_CLOUD(self.gpu_ids, self.reso)#

        cnt = 0

        pose_list = cameras_norm.get_fixed_8_views()
        for i in range(len(model_list)):
           print(model_list[i])
           if model_list[i].find('\\')!=-1:
               str = model_list[i].split('\\')
           else:
               str = model_list[i].split('/')
           model_name = str[1]
           class_name = str[0]

           if not os.path.exists(self.full_pc_dir +'/%s_%s.pcd' % (class_name,model_name)):              
              continue

           intermediate_depth_dir =  self.out_intermediate_depth_dir + '/%s' % class_name
           obj_cood_dir = self.out_obj_cood_dir + '/%s' % class_name
           gt_fixed_depths_dir  =  self.out_gt_fixed_depths_dir + '/%s' % class_name
           coord_AB_dir = self.out_coord_AB_dir + '/%s' % class_name        

           if False:
               new_model_name = model_name
               if self.num_random_views>1:
                    new_model_name = '%s_7' % (model_name)
               else:
                   new_model_name = '%s_0' % (model_name)
               #print(coord_AB_dir+'/%s_%s_cood.exr' % (class_name, new_model_name))
               if os.path.exists(coord_AB_dir+'/%s_%s_coord.exr' % (class_name, new_model_name)):                   
                   print('finish')
                   continue
               
           os.makedirs(intermediate_depth_dir, exist_ok=True)
           os.makedirs(obj_cood_dir, exist_ok=True)
           os.makedirs(gt_fixed_depths_dir, exist_ok=True)
           os.makedirs(coord_AB_dir, exist_ok=True)
                               
           points = torch.from_numpy(np.asarray(point_cloud_operations.read_pcd(self.full_pc_dir +'/%s_%s.pcd' % (class_name,model_name)))).float().cuda(self.gpu_ids)
                        

           gt_shape = (points).clone()
                                 
           #render intermediate gt depth, and obj coord.           
           org_model_name = model_name
           new_model_name = model_name

           random_pose_list = self.read_random_pose(class_name, model_name)

           views = self.num_random_views
           fg = True
           for k in range(views):#self.num_random_views
                
                new_model_name = '%s_%d' % (org_model_name, k)
                
                #pose
                 
                proj_gpu = linux_operations.pcd2depth_with_refinement_dp(points, random_pose_list[k], self.intrinsics, self.infinite_depth, self.grid_pooling, self.gpu_ids, self.reso, global_variables.NOISE_SCALE)

                intermediate_depth_map = torch.where(proj_gpu>10, tensor_0, proj_gpu)
               
                single_pc, x_coord, y_coord, f = linux_operations.depth2pcd_torch_cuda_p_x_y_with_error_info(intermediate_depth_map, self.intrinsics, random_pose_list[k], self.gpu_ids)
                                            
                exr_coord = (-global_variables.EXR_COORD_BACKGROUND_VALUE) * torch.ones((256,256, 3), dtype=torch.float).cuda(self.gpu_ids)
                exr_coord[x_coord, y_coord] = single_pc + global_variables.EXR_COORD_BACKGROUND_SHIFT
                
                cv2.imwrite(coord_AB_dir+'/%s_%s_coord.exr' % (class_name, new_model_name), exr_coord.cpu().numpy())
                
                if OUTPUT_COORD_IMG:
                    im_cood = torch.zeros((256,256, 3), dtype=torch.uint8).cuda(args.gpu_ids)
                    pc_pixel = cameras_norm.normalize_pc_to_pix(single_pc)
                    im_cood[x_coord, x_coord]=pc_pixel
                    #im_cood = torch.flip(im_cood,[0])
                    ic = im_cood.cpu().numpy()
                    cv2.imwrite(intermediate_depth_dir+'/%s_%s_coord.png' % (class_name, new_model_name), ic) 
               
                if OUTPUT_INTERMEDIATE_DEPTH_IMG: #output intermediate depth images
                    im_ab1 = cameras_norm.normalize_depth_min_max(proj_gpu, global_variables.REAL_DEPTH_MIN, global_variables.REAL_DEPTH_MAX)*255               
                    cv2.imwrite(intermediate_depth_dir+'/%s_%s_depth.png' % (class_name, new_model_name), im_ab1) #.type(torch.uint8).cpu().numpy()

                    im_ab1 = cameras_norm.normalize_depth_(proj_gpu)*255               
                    cv2.imwrite(intermediate_depth_dir+'/%s_%s_depth_mm.png' % (class_name, new_model_name), im_ab1) #.type(torch.uint8).cpu().numpy()

                if OUTPUT_COORD_RENDERED_DEPTH_IMG: #output tested renderd depths.
                    self.render_views_from_coord(exr_cood, obj_cood_dir, prefix = '%s_%s' % (class_name, new_model_name))

           if self.render_GT_views:
              render_gt_views.render_save(points=gt_shape, out_dir = gt_fixed_depths_dir, prefix = '%s_%s' % (class_name, model_name), Refine_Depth=True)

        print(cnt)

    def setup_dir(self, args):
       
        if args.isTrain==1:
            test_train_subdir = 'train_dataset'
        else:
            test_train_subdir = 'test_dataset'


        self.full_pc_dir = args.sampled_pc_dir

        self.data_pipeline_base_dir = args.base_dir 

        self.model_list_dir =  self.data_pipeline_base_dir + '/model_list/%s' % args.model_file

        self.out_dataset_dir =  self.data_pipeline_base_dir + '/depth/%s' % test_train_subdir

        self.pose_dir = args.base_dir + '/pose'       

        #output dir
        self.out_intermediate_depth_dir = self.out_dataset_dir + '/intermediate_depth' #+ '/%s' % class_name
        self.out_obj_cood_dir = self.out_dataset_dir + '/obj_cood' #+ '/%s' % class_name
        self.out_gt_fixed_depths_dir = self.out_dataset_dir + '/gt_fixed_depths' #+ '/%s' % class_name
        self.out_coord_AB_dir = self.out_dataset_dir + '/coord_AB' #+ '/%s' % class_name

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--isTrain', type=int, default=0)
    parser.add_argument('--model_file', type=str, default='03001627_test1.txt')    
    parser.add_argument('--gpu_ids', type=int, default=0)
    parser.add_argument('--reso', type=int, default=256)
    parser.add_argument('--render_GT_views', type=int, default=1)

    parser.add_argument('--render_and_combine', type=int, default=0) #read rgba dir, render depth, mask, and make training data.

    parser.add_argument('--sampled_pc_dir', type=str, default = render_setting.SAMPLED_POINT_CLOUD_DIR)

    parser.add_argument('--base_dir', type=str, default = render_setting.DATASET_BASE_DIR)
        
    parser.add_argument('--num_views', type=int, default=8)
    
    args = parser.parse_args()

    rr = Reconstruction_Renderer_2(args)

    rr.render_obj_coord_gt_depths_mitusba()


