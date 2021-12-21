#calculate CD.

import tensorflow as tf
import argparse, sys, os
import open3d
#import projections.point_cloud_operations as point_cloud_operations
import csv
import numpy as np
import json
from collections import OrderedDict
import .icp as icp

#from projections.point_cloud_operations import read_pcd, save_pcd
    
import projections.point_cloud_operations as point_cloud_operations
from pc_distance import tf_nndistance
import config

def chamfer(pcd1, pcd2):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = tf.reduce_mean(tf.sqrt(dist1))
    dist2 = tf.reduce_mean(tf.sqrt(dist2))
    #print(dist1,  dist2)
    return dist1,  dist2
    #return (dist1 + dist2) / 2

def save_list(filename, lst):
    cnt=0
    with open(filename, 'w') as filehandle:
        for listitem in lst:
            cnt+=1
            if cnt==len(lst):
                filehandle.write('%s' % listitem)
            else:
                filehandle.write('%s\n' % listitem) 

def rotate_x(points):
    y = np.copy(points[:,1])#.clone()
    points[:,1] = points[:,2]
    points[:,2]= -y        
    return points

def rotate_x_(points):
    z = np.copy(points[:,2])#.clone()
    points[:,2] = points[:,1]
    points[:,1] = -z        
    return points

def rotate_y_(points):
    z = np.copy(points[:,2])#.clone()
    points[:,2] = -points[:,0]
    points[:,0] = z        
    return points

def transform_lmnet(lm_pos1):
    lm_pos = np.copy(lm_pos1)
    lm_pos = rotate_y_(lm_pos)
    lm_pos  = rotate_x(lm_pos)
    lm_pos  = rotate_y_(lm_pos)    
    return lm_pos    
    
def icp_transform(model_name, gen_pc, gt_pc):
    if model_name=='lmnet' or model_name =='mvs'  or model_name =='psgn':
        T, _, _ = icp.icp(gt_pc, gen_pc[:1024,:], tolerance=1e-10, max_iterations=1000)    
        gen_pc=np.matmul(gen_pc, T[:3,:3])-T[:3, 3]
        return gen_pc
    if model_name=='ours':
        if gen_pc.shape[0]>6000:
            num=6000
        else:
            num = gen_pc.shape[0]
        rand_indices = np.random.permutation(gen_pc.shape[0])[:num]
        rand_indices2 = np.random.permutation(40000)[:num]
        T, _, _ = icp.icp(gt_pc[rand_indices2], gen_pc[rand_indices], tolerance=1e-10, max_iterations=1000)    
        gen_pc=np.matmul(gen_pc, T[:3,:3])-T[:3, 3]
        return gen_pc

def calc_cd_pix3d(generated_dir, gt_dir, gt4_dir, model_name, use_icp):

    base_dir = generated_dir + '/../'
    model_list = os.listdir(generated_dir)

    results_file = base_dir + '/icp%d_1024.csv' % (use_icp)
    
    csv_file1 = open(results_file, 'w')
    writer_1024 = csv.writer(csv_file1) 
    writer_1024.writerow(['name', 'id', 'pred->gt', 'gt->pred', 'cd'])

    results_file =base_dir + '/icp%d_40k.csv' % (use_icp)
    csv_file4 = open(results_file, 'w')
    writer_4k = csv.writer(csv_file4) 
    writer_4k.writerow(['name', 'id', 'pred->gt', 'gt->pred', 'cd'])
    
    cd1_1_per_cat = {}
    cd1_2_per_cat = {}
    cd4_1_per_cat = {}
    cd4_2_per_cat = {}

    avg_dist1_1 = 0.0
    avg_dist1_2 = 0.0
    avg_dist4_1 = 0.0
    avg_dist4_2 = 0.0

    #cd session
    gt = tf.placeholder(tf.float32, (1, None, 3))
    output = tf.placeholder(tf.float32, (1, None, 3))
    cd_op = chamfer(output, gt)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    
    icp_generated_dir = generated_dir+'_icp'
    os.makedirs(icp_generated_dir, exist_ok=True)

    #sess = tf.Session(); 
    cnt=0

    for i in range(len(model_list)):
        
        #model in model_list:
        class_name = model_list[i].split('_')[0]
        id = model_list[i].split('_')[1].split('.')[0]

        gen_pc = point_cloud_operations.read_pcd(generated_dir + '/%s_%s.pcd' % (class_name, id)) 

        name = gt_dir + '/%s_%s.pcd' % (class_name, id)
        if not(os.path.exists(name) and os.path.exists(gt4_dir + '/%s_%s.pcd' % (class_name, id))):
            continue

        #if os.path.exists(icp_generated_dir+ '/%s_%s.pcd' % (class_name, id)):
        #    continue

        gt_pc = point_cloud_operations.read_pcd(name)
        gt4_pc = point_cloud_operations.read_pcd(gt4_dir + '/%s_%s.pcd' % (class_name, id))


        cnt+=1

        print('/%s_%s.pcd' % (class_name, id)) 

        if model_name =='mvs':
            gen_pc = rotate_x(gen_pc)
            gen_pc = rotate_y_(gen_pc)
        elif model_name =='ours':
            #t=0
            gen_pc = rotate_y_(gen_pc)
        elif model_name =='lmnet':
            gen_pc = transform_lmnet(gen_pc)
        
        if use_icp:
            if model_name=='ours':                
                gen_pc = icp_transform(model_name, gen_pc, gt4_pc)    
            else:
                gen_pc = icp_transform(model_name, gen_pc, gt_pc)    

        point_cloud_operations.save_pcd(icp_generated_dir+'/%s_%s.pcd' % (class_name, id), gen_pc, isRefine=False)
        #continue

        cd = sess.run([cd_op], feed_dict={output: [gen_pc], gt: [gt_pc]})        
        print('cd', cd[0])
        dist1=cd[0][0]*100
        dist2=cd[0][1]*100
        writer_1024.writerow([class_name, id, dist1, dist2, (dist1+dist2)])

        cd = sess.run([cd_op], feed_dict={output: [gen_pc], gt: [gt4_pc]})        
        print('cd', cd[0])
        dist3=cd[0][0]*100
        dist4=cd[0][1]*100
        writer_4k.writerow([class_name, id, dist1, dist2, (dist1+dist2)])

        if not cd1_1_per_cat.get(class_name):
            cd1_1_per_cat[class_name] = []
            cd1_2_per_cat[class_name] = []
            cd4_1_per_cat[class_name] = []
            cd4_2_per_cat[class_name] = []

        cd1_1_per_cat[class_name].append(dist1)
        cd1_2_per_cat[class_name].append(dist2)
        cd4_1_per_cat[class_name].append(dist3)
        cd4_2_per_cat[class_name].append(dist4)

        avg_dist1_1 += dist1
        avg_dist1_2 += dist2
        avg_dist4_1 += dist3
        avg_dist4_2 += dist4

    avg_dist1_1 /= cnt
    avg_dist1_2 /= cnt
    avg_dist4_1 /= cnt
    avg_dist4_2 /= cnt


    cdlist=list(cd1_1_per_cat.keys())
    cdlist.sort()

    for class_name in cdlist:        
        writer_1024.writerow([class_name, class_name, '%.2f' % np.mean(cd1_1_per_cat[class_name]), '%.2f' % np.mean(cd1_2_per_cat[class_name]), '%.2f' % (np.mean(cd1_1_per_cat[class_name]) + np.mean(cd1_2_per_cat[class_name]))])
        writer_4k.writerow([class_name, class_name, '%.2f' % np.mean(cd4_1_per_cat[class_name]), '%.2f' % np.mean(cd4_2_per_cat[class_name]), '%.2f' % (np.mean(cd4_1_per_cat[class_name]) + np.mean(cd4_2_per_cat[class_name]))])    
    
    writer_1024.writerow(['all', 'all', '%.2f' % avg_dist1_1, '%.2f' % avg_dist1_2, '%.2f' % (avg_dist1_2 + avg_dist1_1)])
    writer_4k.writerow(['all', 'all', '%.2f' % avg_dist4_1, '%.2f' % avg_dist4_2, '%.2f' % ( avg_dist4_1+ avg_dist4_2)])    

    csv_file1.close()
    csv_file4.close()
    sess.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pcd_dir', type = str, required=True)
    parser.add_argument('--model_name', type = str, required=True)
    parser.add_argument('--use_icp', action='store_true')
    parser.add_argument('--ground_truth_dir', type = str, default='') 
    args = parser.parse_args()

    gt_dir = os.path.join(args.ground_truth_dir,'1024')
    gt4_dir = os.path.join(args.ground_truth_dir,'40k')

    generated_dir = args.pcd_dir
    print(gt_dir)
    
    calc_cd_pix3d(generated_dir, gt_dir, gt4_dir,  args.model_name, args.use_icp)

    