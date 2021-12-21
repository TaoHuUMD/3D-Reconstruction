import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
from config import *
from . import networks_rec_shape
from .networks_rec_shape import *
import os
import cv2

import projections.global_variables as global_variables

class RecTextureDepthModel(BaseModel):
    def name(self):
        return 'RecTextureDepth'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        if RESOLUTION == 256:
            parser.set_defaults(norm='batch', netG='unet_256')
        elif RESOLUTION == 128:
            parser.set_defaults(norm='batch', netG='unet_128')
        
        #to change
        parser.set_defaults(dataset_mode='RecTextureDepth')
        parser.set_defaults(input_nc=1)
        parser.set_defaults(output_nc=1)
        
        if is_train:
            parser.set_defaults(pool_size=0, no_lsgan=True)
            parser.add_argument('--lambda_L1', type=float, default= 100, help='weight for L1 loss') #parser.weight_l_loss

        return parser
    
    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter\\buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    

    def train_pix2pix(self, opt):

        self.netG = networks_rec_shape.define_Unet_Skip(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.resolution, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, True)
      
        self.set_requires_grad(self.netG, True)

        self.params_G = list(self.netG.parameters())
           
    def forward_pix2pix(self):        
        self.fake_B = self.netG(self.real_A_calc)

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        
        self.opt = opt
        self.isLossFunc=opt.isLossFunc
        
        self.loss_names = ['G_L1']#

        # specify the images you want to save\\display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks

        self.model_names = ['G']
        self.train_pix2pix(opt)

        # optimizer.
        self.optimizers = []
     
        if self.isTrain:                
            use_sigmoid = opt.no_lsgan

            self.optimizer_G = torch.optim.Adam(self.params_G,
                                            lr=opt.lr, betas=(opt.beta1, 0.999))            
            self.optimizers.append(self.optimizer_G)
            

        self.criterionL1_no_reduce = torch.nn.L1Loss(reduction='none').to(self.device)
        self.criterionL1 = torch.nn.L1Loss().to(self.device)
        self.criterionL2 = torch.nn.MSELoss().to(self.device)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            
                   
    def set_input(self, input):

        AtoB = self.opt.direction == 'AtoB'
        
        self.real_A_calc = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        
        tensor1 = (torch.cuda.FloatTensor(1).zero_()+1).cuda(self.device)

        self.real_A_calc = torch.where(self.real_A_calc > tensor1*global_variables.DEPTH_BOUND, -global_variables.EXR_DEPTH_BACKGROUND_VALUE*tensor1, self.real_A_calc)

        #flat views
        self.real_B = self.real_B.view(self.real_B.shape[0]*self.opt.view_num, self.opt.resolution, self.opt.resolution, self.opt.input_nc)
        self.real_B = self.real_B.permute(0,3,1,2)

        self.real_A = self.real_A_calc.view(self.real_A_calc.shape[0]*self.opt.view_num, self.opt.resolution, self.opt.resolution, self.opt.input_nc)
        self.real_A = self.real_A.permute(0,3,1,2)        
        
    def forward(self):
        self.forward_pix2pix()


    def backward_G(self):
        
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.weight_l_loss                    
        self.loss_G = self.loss_G_L1
        
        self.loss_G.backward()


    def optimize_parameters(self):
        
        self.forward()
        
        # update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
