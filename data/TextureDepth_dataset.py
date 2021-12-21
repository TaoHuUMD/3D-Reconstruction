import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
from config import *
import numpy as np
import imageio
import cv2

class TextureDepthDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.flag)
        self.AB_paths = sorted(make_dataset(self.dir_AB))

    def __getitem__(self, index):

        AB_path = self.AB_paths[index]

        group = cv2.imread(AB_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        
        h,w = group.shape
       
        w2 = int(w / 2)
         
        A = group[:, 0:int(w/2)]      
        B = group[:, int(w/2):w]

        if self.opt.only_depth:
            if w2==256:
                at = A
                bt = B
            else:
                at = A[:,256*3:256*4]
                bt = B[:,256*3:256*4]
            A = transforms.ToTensor()(at)
            B = transforms.ToTensor()(bt)
        else:
            at = np.concatenate((np.expand_dims(A[:,0:256], 2), np.expand_dims(A[:,256*1:256*2], 2), np.expand_dims(A[:,256*2:256*3], 2), np.expand_dims(A[:,256*3:256*4], 2)),2)
            bt = np.concatenate((np.expand_dims(B[:,0:256], 2), np.expand_dims(B[:,256*1:256*2], 2), np.expand_dims(B[:,256*2:256*3], 2), np.expand_dims(B[:,256*3:256*4], 2)),2)
            A = torch.from_numpy(at)
            B = torch.from_numpy(bt)
        
        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        view_id = torch.tensor(range(0,self.opt.view_num))
        
        return {'A': A, 'B': B,
                    'A_paths': AB_path, 'B_paths': AB_path, 'View_id': view_id}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'TextureDepthDataset'