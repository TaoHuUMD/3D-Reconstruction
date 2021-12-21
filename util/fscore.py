import os
import open3d
import argparse
import csv
import numpy as np
from .metrics import rotate_x, rotate_x_, rotate_y_, transform_lmnet
import projections.point_cloud_operations as point_cloud_operations
from .metrics import icp_transform

def calculate_fscore(gt: open3d.geometry.PointCloud, pr: open3d.geometry.PointCloud, th: float=0.01):
    '''Calculates the F-score between two point clouds with the corresponding threshold value.'''
    # code is from https://github.com/ThibaultGROUEIX/AtlasNet/
    d1 = open3d.compute_point_cloud_to_point_cloud_distance(gt, pr)
    d2 = open3d.compute_point_cloud_to_point_cloud_distance(pr, gt)
    
    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(len(d2))
        precision = float(sum(d < th for d in d1)) / float(len(d1))

        if recall+precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0

    return fscore, precision, recall


def calc_fscore(generated_dir, gt_dir, out_path, model_name, kind_name, use_icp):

    CUBE_SIDE_LEN = 1.0

    open3d.set_verbosity_level(open3d.utility.VerbosityLevel.Error)

    
    #threshold_list = [CUBE_SIDE_LEN/200, CUBE_SIDE_LEN/100,
    #               CUBE_SIDE_LEN/50, CUBE_SIDE_LEN/20,
    #                CUBE_SIDE_LEN/10, CUBE_SIDE_LEN/5]
    #threshold_list = [CUBE_SIDE_LEN*2/100, CUBE_SIDE_LEN*0.8/100, CUBE_SIDE_LEN*0.5/100]
    threshold_list = [CUBE_SIDE_LEN*0.5/100, CUBE_SIDE_LEN*0.8/100, CUBE_SIDE_LEN/100, CUBE_SIDE_LEN*2/100]


    base_dir = generated_dir + '/../'
    model_list = os.listdir(generated_dir)

    for th in threshold_list:    

        os.makedirs(os.path.join(out_path, '%s_%s_%f' % (model_name, kind_name, th)), exist_ok=True)
        results_file = os.path.join(out_path, '%s_%s_%f' % (model_name, kind_name, th), "fscore.csv")
        
        csv_file1 = open(results_file, 'w')
        file_writer = csv.writer(csv_file1) 
        file_writer.writerow(['name', 'id', 'fscore', 'precision', 'recall'])


        cnt=0

        for i in range(len(model_list)):
            
            #model in model_list:
            class_name = model_list[i].split('_')[0]
            id = model_list[i].split('_')[1].split('.')[0]

            gen_pc = point_cloud_operations.read_pcd(generated_dir + '/%s_%s.pcd' % (class_name, id)) 

            #print(gen_pc.size)

            name = gt_dir + '/%s_%s.pcd' % (class_name, id)
            if not(os.path.exists(name)):
                continue
                

            if use_icp:
                print('icp')
                if model_name =='mvs':
                    gen_pc = rotate_x(gen_pc)
                    gen_pc = rotate_y_(gen_pc)
                elif model_name =='ours':
                    gen_pc = rotate_y_(gen_pc) 
                elif model_name =='lmnet':
                    gen_pc = transform_lmnet(gen_pc)
                elif model_name=='psgn':
                    gen_pc = transform_lmnet(gen_pc)

                gt_pc = point_cloud_operations.read_pcd(name)
                gen_pc = icp_transform(model_name, gen_pc, gt_pc)


            gt_pc = open3d.read_point_cloud(name)

            cnt+=1

            print('/**************%s_%s.pcd' % (class_name, id)) 

            
                    
            pcd = open3d.PointCloud()
            pcd.points = open3d.Vector3dVector(gen_pc)
            gen_pc = pcd

            #for th in threshold_list:
            f, p, r = calculate_fscore(gt_pc, gen_pc, th=th)        
            file_writer.writerow([class_name, id, f, p, r])

            print(f,p,r)

            
        csv_file1.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pcd_dir', type = str, required=True)
    parser.add_argument('--test_kind', type = str, required=True)
    parser.add_argument('--model_name', type = str, required=True)
    parser.add_argument('--use_icp', action='store_true')    
    parser.add_argument('--ground_truth_dir', type = str, default='dataset/pix3d/ground_truth')  
    parser.add_argument('--out_path', type = str, default='results/pix3d/fscore') 
    
    args = parser.parse_args()

    gt_dir = args.ground_truth_dir


    generated_dir = args.pcd_dir
    
    #use 
    calc_fscore(generated_dir, gt_dir, args.out_path, args.model_name, args.test_kind, args.use_icp)

    