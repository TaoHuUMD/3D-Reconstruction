import argparse
import array
import numpy as np
import os
from itertools import combinations
import random
import cv2
import torch
import random
import time
#import imageio
import h5py
import argparse, sys, os

from . import global_variables
from . import cameras_norm
from . import point_cloud_operations

def pcd2depth_torch_xyd_upsample(points, intrinsics, pose, gpu_id, size=256):
    inv_pose = torch.from_numpy(np.linalg.inv(pose)).float().cuda(gpu_id)
    x=torch.ones((1, points.shape[0]))
    tmp_p = torch.cat([torch.transpose(points,0,1), torch.ones((1, points.shape[0])).cuda(gpu_id)], 0)
        
    r_camera_points = inv_pose @ tmp_p

    inv_K = np.linalg.inv(intrinsics)
    inv_K[2, 2] = -1
    r_inv_K = torch.from_numpy(np.linalg.inv(inv_K)).float().cuda(gpu_id)
    
    r_dr=r_inv_K@r_camera_points[:3, :]

    dp=r_dr[2]
    d_x=r_dr[1]/dp
    d_y=r_dr[0]/dp

    d_x = ((size-1-d_x)*global_variables.GRID_SIZE).type(torch.int64)
    d_y = (d_y*global_variables.GRID_SIZE).type(torch.int64)
    
    return d_x,d_y,dp

def pcd2depth_with_refinement_dp(points, pose, intrinsics, infinite_depth, grid_pooling, gpu_id, reso, NOISE_SCALE):
    
    tensor_0=torch.cuda.FloatTensor(1).zero_().cuda(gpu_id)
    tensor_1=(torch.cuda.FloatTensor(1).zero_()+1).cuda(gpu_id)
    
    tensor_0_cpu = torch.FloatTensor(1).zero_()
    tensor_1_cpu = torch.FloatTensor(1).zero_()+1

    x, y, dp=pcd2depth_torch_xyd_upsample(points,intrinsics,pose, gpu_id, reso)
    
    if global_variables.USE_MITSUBA_RENDER_PARA:
        dp=-dp

    projection = infinite_depth + 0
    projection[x, y] = -dp
    
    projection = torch.abs(grid_pooling(projection.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0))           

    #delete noisy points appearing in the middle of the shape,
    #first remove outlier points
    proj_cpu = projection.cpu().numpy() #.data #cpu()
           
    #extract internal nodes
    id = np.argwhere(proj_cpu<10)
    cx = id[:,0]
    cy = id[:,1]

    l = cx-1
    id = np.argwhere(proj_cpu[l,cy]<10)
    cy = cy[id[:,0]]
    cx = cx[id[:,0]]

    r = cx + 1
    id = np.argwhere(proj_cpu[r,cy]<10)
    cy = cy[id[:,0]]
    cx = cx[id[:,0]]

    u = cy - 1
    id = np.argwhere(proj_cpu[cx, u]<10)
    cy = cy[id[:,0]]
    cx = cx[id[:,0]]

    d = cy + 1
    id = np.argwhere(proj_cpu[cx, d]<10)
    cy = cy[id[:,0]]
    cx = cx[id[:,0]]

    #consider noisy points appearing inside the shape                   
    proj_gpu = torch.from_numpy(proj_cpu).cuda(gpu_id)
         
    
    t=0
    
    if True:
        lu = proj_gpu[cx-1, cy-1]
        u  = proj_gpu[cx, cy-1]
        ru = proj_gpu[cx+1, cy-1]
        r = proj_gpu[cx+1, cy]
        rd = proj_gpu[cx+1, cy+1]
        d = proj_gpu[cx, cy+1]
        ld = proj_gpu[cx-1, cy+1]
        l = proj_gpu[cx-1, cy]

        m=proj_gpu[cx, cy]

        same_num = torch.where(m > lu + NOISE_SCALE, tensor_0, tensor_1)
        diff_sum = torch.where(m > lu + NOISE_SCALE, lu, tensor_0)

        same_num += torch.where(m > u + NOISE_SCALE, tensor_0, tensor_1)
        diff_sum += torch.where(m > u + NOISE_SCALE, u, tensor_0)

        same_num += torch.where(m > ru + NOISE_SCALE, tensor_0, tensor_1)
        diff_sum += torch.where(m > ru + NOISE_SCALE, ru, tensor_0)

        same_num += torch.where(m > r + NOISE_SCALE, tensor_0, tensor_1)
        diff_sum += torch.where(m > r + NOISE_SCALE, r, tensor_0)

        same_num += torch.where(m > rd + NOISE_SCALE, tensor_0, tensor_1)
        diff_sum += torch.where(m > rd + NOISE_SCALE, rd, tensor_0)

        same_num += torch.where(m > d + NOISE_SCALE, tensor_0, tensor_1)
        diff_sum += torch.where(m > d + NOISE_SCALE, d, tensor_0)

        same_num += torch.where(m > ld + NOISE_SCALE, tensor_0, tensor_1)
        diff_sum += torch.where(m > ld + NOISE_SCALE, ld, tensor_0)

        same_num += torch.where(m > l + NOISE_SCALE, tensor_0, tensor_1)
        diff_sum += torch.where(m > l + NOISE_SCALE, l, tensor_0)           

        proj_gpu[cx, cy] = torch.where(same_num >=3, m, diff_sum/(8-same_num))

    return proj_gpu

def pcd2depth_texture_without_refinement_filter(points, textures, pose, intrinsics, infinite_depth, infinite_depth3c, grid_pooling, gpuids, reso):

    x, y, dp = pcd2depth_torch_xyd_upsample(points,intrinsics,pose, gpuids, reso)

    if global_variables.USE_MITSUBA_RENDER_PARA:
       dp=-dp

    projection = infinite_depth + 0

    p_index = np.array(range(x.cpu().shape[0]))

    #remove wrong points.
    id = np.argwhere(x.cpu() >= 0)
    #print(id.shape, id[0].shape)
    x = x[id[0]]
    y = y[id[0]]
    dp = dp[id[0]]
    p_index = p_index[id[0]]

    id = np.argwhere(y.cpu() >= 0)
    x = x[id[0]]
    y = y[id[0]]
    dp = dp[id[0]]
    p_index = p_index[id[0]]

    id = np.argwhere(x.cpu() < global_variables.GRID_SIZE*reso)
    x = x[id[0]]
    y = y[id[0]]
    dp = dp[id[0]]
    p_index = p_index[id[0]]

    id = np.argwhere(y.cpu() < global_variables.GRID_SIZE*reso)
    x = x[id[0]]
    y = y[id[0]]
    dp = dp[id[0]]
    p_index = p_index[id[0]]

    projection[x, y] = -dp                                 

    depth_pooling, index = grid_pooling(projection.unsqueeze(0).unsqueeze(0))
    depth_map = torch.abs(depth_pooling.squeeze(0).squeeze(0))

    texture_pooling = infinite_depth3c
    
    texture_pooling[x,y] = textures[p_index]

    #tex 3 channel
    texture_map0 = retrieve_elements_from_indices(texture_pooling[:,:,0].unsqueeze(0).unsqueeze(0), index).view(reso,reso)
    texture_map1 = retrieve_elements_from_indices(texture_pooling[:,:,1].unsqueeze(0).unsqueeze(0), index).view(reso,reso)
    texture_map2 = retrieve_elements_from_indices(texture_pooling[:,:,2].unsqueeze(0).unsqueeze(0), index).view(reso,reso)

    texture_map = torch.cat([texture_map0, texture_map1, texture_map2], 1)
    texture_map = torch.abs(texture_map)

    return depth_map, texture_map


def retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=2)
    output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)) #.view_as(indices)
    return output

def pcd2depth_without_refinement_filter(points, pose, intrinsics, infinite_depth, grid_pooling, gpuids, reso):

    x, y, dp = pcd2depth_torch_xyd_upsample(points,intrinsics,pose, gpuids, reso)           
    projection = infinite_depth + 0

    #remove wrong points.
    id = np.argwhere(x.cpu() >= 0)
    x = x[id[0]]
    y = y[id[0]]
    dp = dp[id[0]]

    id = np.argwhere(y.cpu() >= 0)
    x = x[id[0]]
    y = y[id[0]]
    dp = dp[id[0]]

    id = np.argwhere(x.cpu() < global_variables.GRID_SIZE*reso)
    x = x[id[0]]
    y = y[id[0]]
    dp = dp[id[0]]

    id = np.argwhere(y.cpu() < global_variables.GRID_SIZE*reso)
    x = x[id[0]]
    y = y[id[0]]
    dp = dp[id[0]]

    projection[x, y] = -dp                                 
    projection = torch.abs(grid_pooling(projection.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0))
    return projection,x,y

 
def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        print (gpu_ids)
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    return net

def depth2pcd_torch_cuda_p_x_y_with_error_info(depth_org, intrinsics, pose, gpu_id):
    depth=depth_org.clone()
    inv_K = torch.from_numpy(np.linalg.inv(intrinsics)).float().cuda(gpu_id)
    inv_K[2, 2] = -1

    depth = torch.flip(depth, [0])

    tensor_0=torch.cuda.FloatTensor(1).zero_().cuda(gpu_id)

    if global_variables.USE_MITSUBA_RENDER_PARA:
        depth = -depth
        z=torch.nonzero(torch.where(depth < tensor_0, depth, tensor_0))
    else:
        z=torch.nonzero(torch.where(depth > tensor_0, depth, tensor_0))
    y = z[:,0]
    x = z[:,1]
    
    left = torch.stack([x, y, torch.ones_like(x)],0).float()
    right = depth[y, x]

    e=left*right
    if e.size()[1]<30:
        return 0,0,0,True #failed completion, forced direct fusion

    dr=e

    camera_points = inv_K @ e
    
    pose_mat = torch.from_numpy(pose).float().cuda(gpu_id)

    tmp= pose_mat @ torch.cat([camera_points, torch.ones([1, camera_points.shape[1]], device = torch.device('cuda:%d' % gpu_id))], 0)

    points = torch.transpose(tmp,0,1)[:, :3]

    return points, y, x, False



class Renderer_VIEWS_FROM_POINT_CLOUD:

    def __init__(self, gpu_ids, reso):        
        self.gpu_ids = gpu_ids
        self.reso = reso
        self.fixed_intrinsics = cameras_norm.get_fixed_intrinsics()
        self.pose_list = cameras_norm.get_fixed_8_views()

        grid_size=int(global_variables.GRID_SIZE)
        grid_pooling = torch.nn.MaxPool2d((grid_size, grid_size), stride=(grid_size, grid_size))
        self.grid_pooling  = init_net(grid_pooling, init_type='normal', init_gain=0.02, gpu_ids = [self.gpu_ids])

        grid_pooling_index = torch.nn.MaxPool2d((grid_size, grid_size), stride=(grid_size, grid_size), return_indices=True)
        self.grid_pooling_index = init_net(grid_pooling_index, init_type='normal', init_gain=0.02, gpu_ids = [self.gpu_ids])

        self.infinite_depth = (torch.ones((self.reso*grid_size, self.reso*grid_size), dtype=torch.float)*(-20)).cuda(self.gpu_ids)

        self.infinite_depth_3c = (torch.ones((self.reso*grid_size, self.reso*grid_size, 3), dtype=torch.float)*(-20)).cuda(self.gpu_ids)
   

    def projection2exr(self, projection):#depth
        tensor_1=(torch.cuda.FloatTensor(1).zero_()-1).cuda(self.gpu_ids)
        #projection tensor to exr img scaled to (0-0.96). 1.0 is infinite.
        exr_img = ((projection - global_variables.REAL_DEPTH_MIN)*global_variables.TRAIN_SCALE_DEPTH/global_variables.REAL_SCALE_DEPTH + global_variables.TRAIN_DEPTH_MIN)

        exr_img = torch.where(exr_img>global_variables.DEPTH_BOUND, tensor_1*global_variables.EXR_DEPTH_BACKGROUND_VALUE, exr_img)
        return exr_img

    def texture2exr(self, texture_map):
        tensor_1=(torch.cuda.FloatTensor(1).zero_()-1).cuda(self.gpu_ids)
        exr_img = torch.where(texture_map>global_variables.DEPTH_BOUND, tensor_1*global_variables.EXR_RGB_BACKGROUND_VALUE, texture_map)
        return exr_img
          
    def pix2depth(self, im):
        tensor_0 = torch.FloatTensor(1).zero_()
        float_im = im.float()
        d= torch.where(im==255, tensor_0, (float_im/255 - global_variables.TRAIN_DEPTH_MIN) * global_variables.REAL_SCALE_DEPTH / global_variables.TRAIN_SCALE_DEPTH + global_variables.REAL_DEPTH_MIN)
        return d

    def render2depth(self, points, isRefine=True):

        depth_list = []

        for j in range(len(self.pose_list)):

            if isRefine:
                proj_gpu,x,y = pcd2depth_with_refinement_dp(points, self.pose_list[j], self.fixed_intrinsics, self.infinite_depth, self.grid_pooling, self.gpu_ids, self.reso, global_variables.NOISE_SCALE)
            else:
                proj_gpu, x, y = pcd2depth_without_refinement_filter(points, self.pose_list[j], self.fixed_intrinsics, self.infinite_depth, self.grid_pooling, self.gpu_ids, self.reso)
           
            exr_img = self.projection2exr(proj_gpu)
    
            depth_list.append(exr_img.cpu().numpy())

        return depth_list


    def depth_texture_2_exr(self, depth, texture):

        tensor_1=(torch.cuda.FloatTensor(1).zero_()-1).cuda(self.gpu_ids)
        #projection tensor to exr img scaled to (0-0.96). 1.0 is infinite.
        exr_depth = ((depth - global_variables.REAL_DEPTH_MIN)*global_variables.TRAIN_SCALE_DEPTH/global_variables.REAL_SCALE_DEPTH + global_variables.TRAIN_DEPTH_MIN)

        depth_map = torch.where(exr_depth>global_variables.DEPTH_BOUND, tensor_1, exr_depth)

        texture_map0 = torch.where(exr_depth>global_variables.DEPTH_BOUND, tensor_1*global_variables.EXR_DEPTH_BACKGROUND_VALUE, texture[:,0:256])
        texture_map1 = torch.where(exr_depth>global_variables.DEPTH_BOUND, tensor_1*global_variables.EXR_RGB_BACKGROUND_VALUE, texture[:,256:512])
        texture_map2 = torch.where(exr_depth>global_variables.DEPTH_BOUND, tensor_1*global_variables.EXR_RGB_BACKGROUND_VALUE, texture[:,512:768])

        return depth_map, torch.cat([texture_map0, texture_map1, texture_map2], 1)

    def render_depth_texture(self, points, textures, isRefine=True):

        depth_texture_list = []

        for j in range(len(self.pose_list)):

            if isRefine:
                proj_gpu,x,y = pcd2depth_with_refinement_dp(points, self.pose_list[j], self.fixed_intrinsics, self.infinite_depth, self.grid_pooling, self.gpu_ids, self.reso, global_variables.NOISE_SCALE)
            else:
                depth_map, texture_map = pcd2depth_texture_without_refinement_filter(points, textures, self.pose_list[j], self.fixed_intrinsics, self.infinite_depth, self.infinite_depth_3c, self.grid_pooling_index, self.gpu_ids, self.reso)
           
            depth_map, texture_map = self.depth_texture_2_exr(depth_map, texture_map)
            
            texture_depth = torch.cat([texture_map, depth_map], 1)

            depth_texture_list.append(texture_depth.cpu().numpy())

        return depth_texture_list

    def render_save(self, points, out_dir, prefix, Refine_Depth = False, OUT_PNG_AFTER_REFINE=False, OUT_EXR_PROEJCTED_POINTS=False):

        depth_list = []

        for j in range(len(self.pose_list)):

            if Refine_Depth:
                proj_gpu = pcd2depth_with_refinement_dp(points, self.pose_list[j], self.fixed_intrinsics, self.infinite_depth, self.grid_pooling, self.gpu_ids, self.reso, global_variables.NOISE_SCALE)
                  
            if global_variables.USE_MITSUBA_RENDER_PARA:                
                proj_gpu = torch.flip(proj_gpu,[0])
            
            exr_depth = self.projection2exr(proj_gpu)
    
            exr_file_name = out_dir + '/%s_%d.exr' % (prefix, j)             
            cv2.imwrite(exr_file_name, exr_depth.cpu().numpy())
            
            if OUT_EXR_PROEJCTED_POINTS:

                exr_single_pc = (depth2pcd_torch_cuda(exr_projection,intrinsics,pose_list[j], args.gpu_ids)).cpu()
                if j==0:
                    exr_reprojected_points = exr_single_pc
                else:
                    exr_reprojected_points = np.vstack((exr_single_pc, exr_reprojected_points))

            if OUT_PNG_AFTER_REFINE:
                
                im_ab1 = cameras_norm.normalize_depth_(proj_gpu)
                cv2.imwrite(outdepth_dir+'/%s_%s_%d.png' % (class_name, model_name, j) , im_ab1)
                               
                png_single_pc = (depth2pcd_torch_cuda(pix2depth(im_ab1).cuda(args.gpu_ids),intrinsics,pose_list[j], args.gpu_ids)).cpu()
                if j==0:
                    png_reprojected_points = png_single_pc
                else:
                    png_reprojected_points = np.vstack((png_single_pc, png_reprojected_points))


def render_views_from_coord_save(exr_coord, out_dir, prefix, gpu_ids, reso):

    im = exr_coord.cpu().numpy()

    x,y = np.where(np.any(im > np.array([EXR_COORD_BACKGROUND_VALUE/2, EXR_COORD_BACKGROUND_VALUE/2, EXR_COORD_BACKGROUND_VALUE/2]), -1))#
    
    points = torch.from_numpy(im[x,y]).cuda(gpu_ids)

    render = operations.Renderer_VIEWS_FROM_POINT_CLOUD(gpu_ids, reso)     
    depth_list = render.render2file(points, out_dir, prefix)

def get_model_name(model_name):
    id = model_name.find('X')
    if id!=-1:
        return model_name[0:id]
    
    return model_name


def make_depth_pairs_from_coord(opt, exr_coord, exr_rgb, gt_dir, out_dir, model_name, model_id, use_rgb_mask = 0, gpu_ids=0, reso=256):


    im = exr_coord #.cpu().numpy()
    rgb = exr_rgb #.cpu().numpy()

    model_name = get_model_name(model_name)
    if os.path.exists(out_dir + '/%s_%s_group.exr' % (model_name, model_id)):        
        return 0

    if use_rgb_mask: #org, no mask
        x,y = np.where(np.any(rgb > np.array([-0.5, -0.5, -0.5]), -1))#        
    else:
        x,y =  np.where(np.all(im > np.array([-0.2, -0.2, -0.2]), -1)) #>-0.2, valid points.

    points = torch.from_numpy(im[x,y] - global_variables.EXR_COORD_BACKGROUND_SHIFT).cuda(gpu_ids)

    render = Renderer_VIEWS_FROM_POINT_CLOUD(gpu_ids, reso)     
    depth_list = render.render2depth(points, isRefine=False)

    if not os.path.exists(out_dir):
       os.makedirs(out_dir)

    cmb=[]
    no_file = False
    for i in range(global_variables.FIXED_VIEW_NUM):

        if not opt.test_pix3d and not os.path.exists(gt_dir + '/%s_%d.exr' % (model_name, i)):
            no_file = True
            return 0

        if opt.test_pix3d:
            #gt = depth_list[i]
            gt = np.flip(depth_list[i],[0])
        else:
            #gt = cv2.imread(gt_dir + '/%s_%d.exr' % (model_name, i), cv2.IMREAD_UNCHANGED).astype(np.float32)
            gt = cv2.imread(gt_dir + '/%s_%d.exr' % (model_name, i), cv2.IMREAD_UNCHANGED).astype(np.float32)[:,256*3:256*4]        

        depth_pair = np.concatenate([np.flip(depth_list[i],[0]), gt], 1)#.cpu().numpy()

        if not i:
            comb = depth_pair
        else: 
            comb = np.concatenate((comb, depth_pair),0)

    new_name = '%s_%s_group.exr' % (model_name, model_id)
   
    cv2.imwrite(out_dir + '/%s' % new_name, comb)

def make_depth_texture_pairs_from_coord(opt, exr_coord, exr_rgb, gt_dir, out_dir, model_name, model_id, use_rgb_mask = 0, gpu_ids=0, reso=256):

    im = exr_coord #.cpu().numpy()
    rgb = exr_rgb #.cpu().numpy()

    model_name = get_model_name(model_name)
    
    if use_rgb_mask: #org, no mask
        x,y = np.where(np.any(rgb > np.array([-0.5, -0.5, -0.5]), -1))#        
    else:
        x,y =  np.where(np.all(im > np.array([-0.2, -0.2, -0.2]), -1)) #>-0.2, valid points.

    points = torch.from_numpy(im[x,y] - global_variables.EXR_COORD_BACKGROUND_SHIFT).cuda(gpu_ids)
    texture = torch.from_numpy(rgb[x,y]).cuda(gpu_ids)


    render = Renderer_VIEWS_FROM_POINT_CLOUD(gpu_ids, reso)     

    depth_texture_list = render.render_depth_texture(points, texture, isRefine=False)

    if not os.path.exists(out_dir):
       os.makedirs(out_dir)

    cmb=[]
    no_file = False

    for i in range(global_variables.FIXED_VIEW_NUM):

        if (not opt.test_pix3d) and not os.path.exists(gt_dir + '/%s_%d.exr' % (model_name, i)):
            no_file = True
            print(gt_dir + '/%s_%d.exr' % (model_name, i))
            return 0

        if opt.test_pix3d:
            gt = np.flip(depth_texture_list[i],[0])
        else:
            gt = cv2.imread(gt_dir + '/%s_%d.exr' % (model_name, i), cv2.IMREAD_UNCHANGED).astype(np.float32)        

        
        depth_pair = np.concatenate([np.flip(depth_texture_list[i],[0]), gt], 1)#.cpu().numpy() 

        if not i:            
            comb = depth_pair
        else: 
            comb = np.concatenate((comb, depth_pair),0)

    new_name = '%s_%s_tg.exr' % (model_name, model_id)

    cv2.imwrite(out_dir + '/%s' % new_name, comb)
 
def visualize_depth_texture(path, comb):

    cv2.imwrite(path+'.depth.png', comb[:,256*3:256*4]*255)
    print(np.expand_dims(comb[:,0:256], 2).shape)
    t= np.concatenate((np.expand_dims(comb[:,0:256], 2), np.expand_dims(comb[:,256*1:256*2], 2), np.expand_dims(comb[:,256*2:256*3], 2)),2)
    cv2.imwrite(path+'.texture.png', t*255)
    td = np.concatenate((t,np.expand_dims(comb[:,256*3:256*4],2)),2)
    cv2.imwrite(path+'.depth_texture.png', td*255)    


def restore_depth(gen_data, gpu_id):
    tensor_0=(torch.cuda.FloatTensor(1).zero_()).cuda(gpu_id)

    gen_data = torch.where(gen_data<0, tensor_0, (gen_data-global_variables.TRAIN_DEPTH_MIN)*global_variables.REAL_SCALE_DEPTH / global_variables.TRAIN_SCALE_DEPTH + global_variables.REAL_DEPTH_MIN)
    return gen_data 

class Depth_Fusion_PostProcessing():
    def __init__(self, gpu_id, reso, view_num, vote_number = 0, base_path=''):
        self.gpu_id = gpu_id
        self.reso = reso
        self.view_num = view_num

        self.base_path = base_path
        self.vote_number = vote_number

        self.fixed_intrinsics = cameras_norm.get_fixed_intrinsics()
        self.pose_list = cameras_norm.get_fixed_8_views()

    def fusion(self, model_name):
        self.in_depth_dir = os.path.join(self.base_path, 'results/%s/test_latest__pix/images/' % model_name)
        self.out_pc_dir = os.path.join(self.base_path, 'results/point_cloud/%s/' % model_name)
        if not self.vote_number:
            self.out_pc_dir = self.out_pc_dir + '/direct'
        else:
            self.out_pc_dir = self.out_pc_dir + '/%d' % self.vote_number
        
        if not os.path.exists(self.out_pc_dir):
            os.makedirs(self.out_pc_dir)

        files = ['%s_%s' % (name.split('_')[0], name.split('_')[1]) for name in os.listdir(self.in_depth_dir)]
        model_list = list(dict.fromkeys(files))
        tensor_0 = torch.cuda.FloatTensor(1).zero_().cuda(self.gpu_id)


        model_list.sort()

        new_model_list=[]
        stop_num=0
        for k in range(len(model_list)):
            model = model_list[k]
            cnt=0
            for i in range(self.view_num):                
                pix_file_name = self.in_depth_dir + '/%s_%d_fake_B.exr' % (model, i)                
                if os.path.exists(pix_file_name):
                    cnt+=1
            if  cnt!=8:
                print(model,  cnt)
                stop_num +=1
            else:
                new_model_list.append(model)

        cats = ['%s' % (name.split('_')[0]) for name in os.listdir(self.in_depth_dir)]
        cats = dict.fromkeys(cats)

        print('total cats ', len(cats))
        print('total models ', len(model_list))        

        fg=True

        num_processed_models=0

        for model in new_model_list:            
            out_pcd_name = self.out_pc_dir + '/%s_%s.pcd' % (model.split('_')[0], model.split('_')[1])

            if os.path.exists(out_pcd_name):
                if (point_cloud_operations.read_pcd(out_pcd_name).shape[0])>2048:
                    print('already exists')
                    #continue
            else:
                print(model)
                fg=False

            num_processed_models +=1

            forced_direct_fusion_flag = False # bad completion, avoid compution error

            reprojected_points_list=[]
            direct_fusion = []
            depth_list = []
            cmb_cnt = 0
            for i in range(self.view_num):
                dm_file_name = self.in_depth_dir + '/%s_fake_B_%d.exr' % (model, i)
                pix_file_name = self.in_depth_dir + '/%s_%d_fake_B.exr' % (model, i)
                

                if os.path.exists(dm_file_name):
                    depth = cv2.imread(dm_file_name, cv2.IMREAD_UNCHANGED).astype(np.float32)
                else:
                    #print(pix_file_name)
                    depth = cv2.imread(pix_file_name, cv2.IMREAD_UNCHANGED).astype(np.float32)

                depth = torch.from_numpy(depth).float().cuda(self.gpu_id)
                
                depth = restore_depth(depth, self.gpu_id)                
                depth = torch.flip(depth,[0])                
                
                single_pc, x_cood, y_cood, f = depth2pcd_torch_cuda_p_x_y_with_error_info(depth, self.fixed_intrinsics, self.pose_list[i], self.gpu_id)


                if global_variables.REFINED_WITH_BOX: #remove invalid points.
                    
                    bound_min = -0.52
                    bound_max = 0.52

                    #p_index = np.array(range(single_pc.cpu().shape[0]))                    
                    id = np.argwhere(single_pc[:,0].cpu() >= bound_min)
                    single_pc = single_pc[id[0]]
                    id = np.argwhere(single_pc[:,0].cpu() <= bound_max)
                    single_pc = single_pc[id[0]]

                    id = np.argwhere(single_pc[:,1].cpu() >= bound_min)
                    single_pc = single_pc[id[0]]
                    id = np.argwhere(single_pc[:,1].cpu() <= bound_max)
                    single_pc = single_pc[id[0]]


                    id = np.argwhere(single_pc[:,2].cpu() >= bound_min)
                    single_pc = single_pc[id[0]]
                    id = np.argwhere(single_pc[:,2].cpu() <= bound_max)
                    single_pc = single_pc[id[0]]



                if f: #no valid points
                    forced_direct_fusion_flag = True
                    continue

                single_pc=single_pc.cpu().numpy()

                reprojected_points_list.append(single_pc)
                depth_list.append(depth.cpu().numpy())

                if not self.vote_number:
                    if cmb_cnt==0:
                       direct_fusion = single_pc
                    else:
                       direct_fusion = np.vstack((single_pc, direct_fusion))
                cmb_cnt += 1
            
            class_name = model.split('_')[0]
            model_name = model.split('_')[1].split('.')[0]
            id = model_name.find('X')
            if id!=-1:
                model_name = model_name[0:id]

            if forced_direct_fusion_flag:
                valid_list=[]
                for k in range(len(reprojected_points_list)):
                    if k==0:
                        valid_list = reprojected_points_list[k]
                    else:
                        valid_list = np.vstack((reprojected_points_list[k], valid_list))
                point_cloud_operations.save_pcd(self.out_pc_dir + '/%s_%s.pcd' % (class_name, model_name), valid_list, isRefine = True)                        
            elif (not self.vote_number):
                point_cloud_operations.save_pcd(self.out_pc_dir + '/%s_%s.pcd' % (class_name, model_name), direct_fusion, isRefine = True)
            elif self.vote_number:
                self.voting(self.out_pc_dir + '/%s_%s.pcd' % (class_name, model_name), reprojected_points_list, depth_list)

        print('fused: ', num_processed_models)

    def bfs(self, grid, start):
        import collections

        queue = collections.deque([[start]])
        seen = set([start])

        width=256
        height=256

        while queue:
            path = queue.popleft()
            x, y = path[-1]
            if np.max(grid[y][x])>0.4 and np.min(grid[y][x])>0: #and np.min(grid[y][x])<0.6 
                return grid[y][x]
            for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
                if 0 <= x2 < width and 0 <= y2 < height and (x2, y2) not in seen:
                    queue.append(path + [(x2, y2)])
                    seen.add((x2, y2))

    def fusion_texture_depth_mix_mode(self, model_name, texture_model_name, withColors=True):

        self.in_depth_dir = os.path.join(self.base_path, './results/%s/test_latest__pix/images/' % model_name)
        
        self.in_texture_dir = os.path.join(self.base_path, './results/%s/test_latest__pix/images/' % texture_model_name)

        self.out_pc_dir = os.path.join(self.base_path, './results/point_cloud/%s_Sd_t/' % model_name)
        
        if not self.vote_number:
            self.out_pc_dir = self.out_pc_dir + '/direct'
        else:
            self.out_pc_dir = self.out_pc_dir + '/%d' % self.vote_number
        
        if not os.path.exists(self.out_pc_dir):
            os.makedirs(self.out_pc_dir)
        
        model_names = ['%s_%s' % (name.split('_')[0], name.split('_')[1]) for name in os.listdir(self.in_depth_dir)]
        valid_names = dict.fromkeys(model_names) 
        model_list = sorted(valid_names)
             
        tensor_0 = torch.cuda.FloatTensor(1).zero_().cuda(self.gpu_id)

        new_model_list=[]
        stop_num=0
        for k in range(len(model_list)):
            model = model_list[k].split('.')[0]
            cnt=0
            for i in range(self.view_num):                
                pix_file_name = self.in_depth_dir + '/%s_%d_fake_B.exr' % (model, i)
                tex = self.in_texture_dir + '/%s_%d_fake_B.exr' % (model, i)                
                if os.path.exists(pix_file_name) and os.path.exists(tex):
                    cnt+=1
            if  cnt!=8:
                print(model, model_list[k], cnt)
                stop_num +=1
            else:
                new_model_list.append(model)

        cats = ['%s' % (name.split('_')[0]) for name in os.listdir(self.in_depth_dir)]
        cats = dict.fromkeys(cats)

        print('total cats ', len(cats))
        print('total models ', len(model_list))        

        fg=True

        num_processed_models=0

        for model in new_model_list:            
            out_pcd_name = self.out_pc_dir + '/%s_%s.pcd' % (model.split('_')[0], model.split('_')[1])

            if os.path.exists(out_pcd_name):
                if (point_cloud_operations.read_pcd(out_pcd_name).shape[0])>2048:
                    print('already exists')                    
            else:
                print(model)
                fg=False

            num_processed_models +=1

            forced_direct_fusion_flag = False # bad completion, avoid compution error

            reprojected_points_list=[]
            reprojected_colors_list=[]

            direct_fusion = []
            depth_list = []
            
            cmb_cnt = 0
            for i in range(self.view_num):
                
                #fake_B: rgb_depth
                depth_file_name = self.in_depth_dir + '/%s_%d_fake_B.exr' % (model, i)                                

                tex_file_name = self.in_texture_dir + '/%s_%d_fake_B.exr' % (model, i)

                rgb = cv2.imread(tex_file_name, cv2.IMREAD_UNCHANGED).astype(np.float32)
                rgb  =  rgb[:, 0:self.reso*3]

                depth = cv2.imread(depth_file_name, cv2.IMREAD_UNCHANGED).astype(np.float32)
                
                rgb = np.concatenate((np.expand_dims(rgb[:,0:256], 2), np.expand_dims(rgb[:,256*1:256*2], 2), np.expand_dims(rgb[:,256*2:256*3], 2)),2)

                depth = torch.from_numpy(depth).float().cuda(self.gpu_id)
                depth = restore_depth(depth, self.gpu_id)
                depth = torch.flip(depth,[0])                

                single_pc, x_coord, y_coord, f = depth2pcd_torch_cuda_p_x_y_with_error_info(depth, self.fixed_intrinsics, self.pose_list[i], self.gpu_id)
                
                #search nearest neighbor.
                xx = x_coord.cpu().numpy()
                yy = y_coord.cpu().numpy()
                
                rgb_cp = np.copy(rgb)

                leng = xx.shape[0]
                for l in range(leng):
                    xi = xx[l]
                    yi = yy[l]
                    if np.min(rgb[xi-1][yi])>0 and np.min(rgb[xi+1][yi])>0 and np.min(rgb[xi][yi-1])>0 and np.min(rgb[xi][yi+1])>0:
                        continue
                    rgb_cp[xi][yi] = self.bfs(rgb,(xi,yi))

                colors = np.roll(rgb_cp[xx, yy][:,0:3],1,1)

                if f: #no valid points
                    forced_direct_fusion_flag = True
                    continue

                single_pc=single_pc.cpu().numpy()

                reprojected_points_list.append(single_pc)
                reprojected_colors_list.append(colors)
                depth_list.append(depth.cpu().numpy())
        
            class_name = model.split('_')[0]
            model_name = model.split('_')[1].split('.')[0]
            id = model_name.find('X')
            if id!=-1:
                model_name = model_name[0:id]

            if forced_direct_fusion_flag or (not self.vote_number):
                valid_list=[]
                color_list=[]

                valid_list = self.fusion_list(reprojected_points_list) #np.array(reprojected_points_list).reshape(-1,3)
                if withColors:
                    color_list = self.fusion_list(reprojected_colors_list) #np.array(reprojected_colors_list).reshape(-1,3)
                
                point_cloud_operations.save_pcd_textures(self.out_pc_dir + '/%s_%s.pcd' % (class_name, model_name), valid_list, color_list, isRefine = True)                                    
            elif self.vote_number:
                
                self.voting(self.out_pc_dir + '/%s_%s.pcd' % (class_name, model_name), reprojected_points_list, depth_list, reprojected_colors_list)

            
        print('fused: ', num_processed_models)

    def fusion_texture_depth(self, model_name, withColors=True):
        self.in_depth_dir = os.path.join(self.base_path, './results/%s/test_latest__pix/images/' % model_name)
        self.out_pc_dir = os.path.join(self.base_path, './results/point_cloud/%s/' % model_name)
        if not self.vote_number:
            self.out_pc_dir = self.out_pc_dir + '/direct'
        else:
            self.out_pc_dir = self.out_pc_dir + '/%d' % self.vote_number
        
        if not os.path.exists(self.out_pc_dir):
            os.makedirs(self.out_pc_dir)

        files = ['%s_%s' % (name.split('_')[0], name.split('_')[1]) for name in os.listdir(self.in_depth_dir)]
        model_list = list(dict.fromkeys(files))
        tensor_0 = torch.cuda.FloatTensor(1).zero_().cuda(self.gpu_id)

        model_list.sort()

        new_model_list=[]
        stop_num=0
        for k in range(len(model_list)):
            model = model_list[k]
            cnt=0
            for i in range(self.view_num):                
                pix_file_name = self.in_depth_dir + '/%s_%d_fake_B.exr' % (model, i)                
                if os.path.exists(pix_file_name):
                    cnt+=1
            if  cnt!=8:
                print(model, model_list[k+1], cnt)
                stop_num +=1
            else:
                new_model_list.append(model)

        cats = ['%s' % (name.split('_')[0]) for name in os.listdir(self.in_depth_dir)]
        cats = dict.fromkeys(cats)

        print('total cats ', len(cats))
        print('total models ', len(model_list))        

        fg=True

        num_processed_models=0

        for model in new_model_list:            
            #model = model[0:model.find('X')]
            out_pcd_name = self.out_pc_dir + '/%s_%s.pcd' % (model.split('_')[0], model.split('_')[1])

            if os.path.exists(out_pcd_name):
                if (point_cloud_operations.read_pcd(out_pcd_name).shape[0])>2048:
                    print('already exists')
                    #continue
            else:
                print(model)
                fg=False

            num_processed_models +=1

            forced_direct_fusion_flag = False # bad completion, avoid compution error

            reprojected_points_list=[]
            reprojected_colors_list=[]

            direct_fusion = []
            depth_list = []
            
            cmb_cnt = 0
            for i in range(self.view_num):
                
                #fake_B: rgb_depth
                file_name = self.in_depth_dir + '/%s_%d_fake_B.exr' % (model, i)                                
                rgbd = cv2.imread(file_name, cv2.IMREAD_UNCHANGED).astype(np.float32)

                depth = rgbd[:,self.reso*3:self.reso*4]
                rgb  =  rgbd[:, 0:self.reso*3]

                rgb = np.concatenate((np.expand_dims(rgb[:,0:256], 2), np.expand_dims(rgb[:,256*1:256*2], 2), np.expand_dims(rgb[:,256*2:256*3], 2)),2)

                depth = torch.from_numpy(depth).float().cuda(self.gpu_id)
                depth = restore_depth(depth, self.gpu_id)
                depth = torch.flip(depth,[0])                

                single_pc, x_coord, y_coord, f = depth2pcd_torch_cuda_p_x_y_with_error_info(depth, self.fixed_intrinsics, self.pose_list[i], self.gpu_id)
                colors = np.roll(rgb[x_coord.cpu().numpy(),y_coord.cpu().numpy()][:,0:3],1,1)

                if f: #no valid points
                    forced_direct_fusion_flag = True
                    continue

                single_pc=single_pc.cpu().numpy()

                reprojected_points_list.append(single_pc)
                reprojected_colors_list.append(colors)
                depth_list.append(depth.cpu().numpy())
        
            class_name = model.split('_')[0]
            model_name = model.split('_')[1].split('.')[0]
            id = model_name.find('X')
            if id!=-1:
                model_name = model_name[0:id]

            if forced_direct_fusion_flag or (not self.vote_number):
                valid_list=[]
                color_list=[]

                valid_list = self.fusion_list(reprojected_points_list) #np.array(reprojected_points_list).reshape(-1,3)
                if withColors:
                    color_list = self.fusion_list(reprojected_colors_list) #np.array(reprojected_colors_list).reshape(-1,3)
                
                point_cloud_operations.save_pcd_textures(self.out_pc_dir + '/%s_%s.pcd' % (class_name, model_name), valid_list, color_list, isRefine = True)                                    
            elif self.vote_number:
                
                self.voting(self.out_pc_dir + '/%s_%s.pcd' % (class_name, model_name), reprojected_points_list, depth_list, reprojected_colors_list)

        print('fused: ', num_processed_models)
        
        

    def pcd2depth_xy(self, points, intrinsics, pose, size=256):

        inv_pose = np.linalg.inv(pose)
        tmp_p = np.concatenate([points.T, np.ones((1, points.T.shape[1]))], 0)
        r_camera_points = np.dot(inv_pose,tmp_p)

        inv_K = np.linalg.inv(intrinsics)
        inv_K[2, 2] = -1
        r_inv_K = np.linalg.inv(inv_K)
        
        r_dr=np.dot(r_inv_K,r_camera_points[:3, :])

        dp=r_dr[2]
        d_x=r_dr[1]/dp
        d_y=r_dr[0]/dp

        r_x=(d_y).astype(int)
        r_y=(d_x).astype(int)

        d_x= (size-1-d_x+0.5).astype(int)
        d_y=(d_y+0.5).astype(int)

        return d_x,d_y

    def voting(self, out_file, points_list, depth_list, colors_list=[]):
        
        #vote
        withColors = len(colors_list)
        valid_list=[]
        valid_colors_list=[]
        for i in range(self.view_num):
            pn=points_list[i].shape[0]
            vote_sum=np.ones((pn,))

            for j in range(self.view_num):
                if j==i:
                    continue

                x,y = self.pcd2depth_xy(points_list[i], self.fixed_intrinsics, self.pose_list[j], size=256)

                x0_id = np.where(x<0)
                vote_sum[x0_id] +=1

                x255_id = np.where(x>255)
                vote_sum[x255_id] +=1

                y0_id = np.where(y<0)
                vote_sum[y0_id] +=1

                y255_id = np.where(y>255)
                vote_sum[y255_id] +=1

                #remove repeated.
                for k in range(len(x0_id[0])):
                    if x0_id[0].size<1: break
                    xd=x0_id[0][k]            
                    if y[xd] > 255: vote_sum[xd]-=1
                    if y[xd] < 0: vote_sum[xd]-=1

                for k in range(len(x255_id[0])):
                    if x255_id[0].size<1: break
                    xd=x255_id[0][k]
                    if y[xd] > 255: vote_sum[xd]-=1
                    if y[xd] < 0: vote_sum[xd]-=1
                   
                #valid coordinate.
                org_id = np.array(range(pn))
                id= np.where(x>=0)
                x=x[id]
                y=y[id]
                org_id=org_id[id]

                id= np.where(x<=255)
                x=x[id]
                y=y[id]
                org_id=org_id[id]

                id= np.where(y>=0)
                x=x[id]
                y=y[id]
                org_id=org_id[id]

                id= np.where(y<=255)
                x=x[id]
                y=y[id]
                org_id=org_id[id]

                id = np.where(depth_list[j][x, y]>0)
                org_id=org_id[id]
                vote_sum[org_id] += 1

            if i==0:
                valid_list = points_list[i][np.where(vote_sum>=self.vote_number)]
                if withColors:
                    valid_colors_list = colors_list[i][np.where(vote_sum>=self.vote_number)]
            else:
                valid_list = np.vstack((points_list[i][np.where(vote_sum>=self.vote_number)], valid_list))
                if withColors:
                    valid_colors_list = np.vstack((colors_list[i][np.where(vote_sum>=self.vote_number)], valid_colors_list))                         
        if len(valid_list)< 2048: #use direct fusion
            valid_list = self.fusion_list(points_list)#np.array(points_list).reshape(-1,3)
            if withColors:
                valid_colors_list = self.fusion_list(points_list) #np.array(colors_list).reshape(-1,3)                        
            point_cloud_operations.save_pcd_textures(out_file, valid_list, valid_colors_list, isRefine = True)
        elif withColors:
            point_cloud_operations.save_pcd_textures(out_file, valid_list, valid_colors_list, isRefine = True)
        else:
            point_cloud_operations.save_pcd(out_file, valid_list, isRefine = True)
    
    def fusion_list(self, points_list):
        valid_list = []
        for k in range(len(points_list)):
            if k==0:
                valid_list = points_list[k]
            else:
                valid_list = np.vstack((valid_list, points_list[k]))
        return valid_list