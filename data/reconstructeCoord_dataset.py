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
import ntpath

class ReconstructeCoordDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.flag)
        self.AB_paths = sorted(make_dataset(self.dir_AB))

        dataset_size = len(self.AB_paths)
        
        if self.opt.phase == 'test':
            self.test_start_index = int(opt.data_start*dataset_size/opt.total_threads)
            self.test_end_index   =  int(opt.data_end*dataset_size/opt.total_threads)
        
    def __getitem__(self, index):

        AB_path = self.AB_paths[index]
        
        if self.opt.phase == 'test':
            if index < self.test_start_index or index> self.test_end_index:
                return {'A': 0, 'B': 0, 'A_paths': AB_path}
        
        AB = cv2.imread(AB_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        if self.opt.input_nc>1:
            h, w, c = AB.shape
        else:
            h, w = AB.shape

        if self.opt.phase == 'test' and self.opt.test_pix3d:
            A = AB
            B = AB
        else:
            A = AB[:, 0:int(w/2)]      
            B = AB[:, int(w/2):w]

        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        
        if (not self.opt.no_flip) and random.random() < 0.5: #flip, data augmentation            
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        return {'A': A, 'B': B,
                'A_paths': AB_path}
                  
    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'ReconstructeCoordDataset'



