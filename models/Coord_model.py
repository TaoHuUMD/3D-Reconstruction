import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
from config import *

class CoordModel(BaseModel):
    def name(self):
        return 'CoordModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        if RESOLUTION == 256:
            parser.set_defaults(norm='batch', netG='unet_256')
        elif RESOLUTION == 128:
            parser.set_defaults(norm='batch', netG='unet_128')
        
        #to change
        if is_train:
            parser.set_defaults(pool_size=0, no_lsgan=True)
            parser.add_argument('--lambda_L1', type=float, default= 100, help='weight for L1 loss') #parser.weight_l_loss
        return parser

    def initialize(self, opt):

        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain        
        
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        #self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'D_real_pred_max', 'D_real_pred_min', 'D_fake_pred_max', 'D_fake_pred_min']
        
        if self.isTrain and opt.use_gan:
            self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'D_real_pred_max', 'D_real_pred_min', 'D_fake_pred_max', 'D_fake_pred_min']
        else:
            self.loss_names = ['G_L1']

        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.use_gan = opt.use_gan
        if self.isTrain and opt.use_gan:
            self.model_names = ['G', 'D']
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        else:  # during test time, only load Gs
            self.model_names = ['G']

        # load\\define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
           
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionL1_no_reduce = torch.nn.L1Loss(reduction='none')
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            if opt.use_gan:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                    lr=opt.lr*opt.d_lr_rate, betas=(opt.beta1, 0.999))                
                self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)        
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.set_requires_grad(self.netG, True)
        self.fake_B = self.netG(self.real_A)
    
    def backward_G(self):
        # First, G(A) should fake the discriminator
        
        #print(self.fake_B[0][0])
        #print(self.real_B[0][0])        
        self.loss_G_L1  = self.criterionL1(self.fake_B, self.real_B) * self.opt.weight_l_loss

        if self.use_gan:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            self.loss_G = self.loss_G_L1 + self.loss_G_GAN
        else:
            self.loss_G = self.loss_G_L1


        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        if self.use_gan:
            # update D
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
            self.set_requires_grad(self.netD, False)

        # update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

