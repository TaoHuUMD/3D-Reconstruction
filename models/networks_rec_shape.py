import torch
import torch.nn as nn
import functools
from config import *
from models.networks import *


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
# one object one calling

#kernel from 32.
def Unet_Gene_32(input_nc, output_nc, num_downs, ngf=32,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
    # construct unet structure
    unet_block = Unet_Down_Up(ngf * 16, ngf * 16, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
    
    for i in range(num_downs - 6):
        unet_block = Unet_Down_Up(ngf * 16, ngf * 16, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
    unet_block = Unet_Down_Up(ngf * 8, ngf * 16, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
    unet_block = Unet_Down_Up(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
    unet_block = Unet_Down_Up(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
    unet_block = Unet_Down_Up(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
    unet_block = Unet_Down_Up(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

    return Unet_One(unet_block.down), Unet_One(unet_block.up)

def Unet_Gene(input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
    # construct unet structure
    unet_block = Unet_Down_Up(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
    
    for i in range(num_downs - 5):
        unet_block = Unet_Down_Up(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
    unet_block = Unet_Down_Up(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
    unet_block = Unet_Down_Up(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
    unet_block = Unet_Down_Up(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
    unet_block = Unet_Down_Up(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

    return Unet_One(unet_block.down), Unet_One(unet_block.up)


def define_Unet_Skip(input_nc, output_nc, ngf, netG, resolution = 256,  norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], is_train_pix2pix=False):

    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        if KERNEL_NUM == 32:            
            unet = UnetGenerator_32_Skip(input_nc, output_nc, 7, 32, norm_layer=norm_layer, use_dropout=use_dropout, is_train_pix2pix = is_train_pix2pix)
        elif KERNEL_NUM == 64:  
            unet = UnetGenerator_64_Skip(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, is_train_pix2pix = is_train_pix2pix)
    elif netG == 'unet_256':
        unet = UnetGenerator_64_Skip_8(input_nc, output_nc, 8, ngf, resolution = resolution, norm_layer=norm_layer, use_dropout=use_dropout,gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    return init_net(unet, init_type, init_gain, gpu_ids)

class UnetGenerator_32_Skip(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs,  ngf=32, resolution=256,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, is_train_pix2pix=False):
        super(UnetGenerator_32_Skip, self).__init__()

        unet_block = UnetSkipConnectionBlock(ngf * 16, ngf * 16, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, is_train_pix2pix = is_train_pix2pix)
    
        for i in range(num_downs - 6):
            unet_block = UnetSkipConnectionBlock(ngf * 16, ngf * 16, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 16, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, resolution = resolution, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        # construct unet structure
        self.model = unet_block

    def forward(self, input):
        x=self.model(input)
        return x

class UnetGenerator_64_Skip_8(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs,  ngf=64, resolution=256,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, is_train_pix2pix=False,gpu_ids=[]):
        super(UnetGenerator_64_Skip_8, self).__init__()

        # construct unet structure     
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True,gpu_ids=gpu_ids)

        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout,gpu_ids=gpu_ids)
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, add_view_pooling_before_x = True, input_shape=(8,8),gpu_ids=gpu_ids)#
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout,  add_view_pooling_after_x = True ,gpu_ids=gpu_ids)#

        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,gpu_ids=gpu_ids) #1.5
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer,gpu_ids=gpu_ids)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer,gpu_ids=gpu_ids)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, resolution = resolution, submodule=unet_block, outermost=True, norm_layer=norm_layer,gpu_ids=gpu_ids)

        # construct unet structure
        self.model = unet_block

    def load_parameters(self, memory_id, view_id):
        self.model.load_parameters(memory_id, view_id)

    def forward(self, input):
        #input[0]=self.model(input)        
        return self.model(input)
# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,  input_nc=None, resolution=256,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, is_train_pix2pix=False, add_view_pooling_before_x = False, input_shape=(2,2), add_view_pooling_after_x = False, gpu_ids=[]):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.use_static_memory = False

        self.add_view_pooling_before_x=add_view_pooling_before_x

        self.add_view_pooling_after_x = add_view_pooling_after_x

        self.inner_nc = inner_nc
        self.out_c = outer_nc
        self.input_c = input_nc

        self.height=input_shape[0]
        self.width=input_shape[1]

        self.VIEW_NUM=8

        inner_nc = inner_nc
        height=input_shape[0]
        width=input_shape[1]

        self.is_train_pix2pix = is_train_pix2pix #pooling memory

        self.submodule =  submodule

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            self.resolution = resolution
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:

            if self.add_view_pooling_before_x and CHANGE_INPUT:                
                downconv = nn.Conv2d(input_nc*2, inner_nc, kernel_size=4,
                        stride=2, padding=1, bias=use_bias)
   
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, bias=use_bias)

            self.down = [downrelu, downconv]
            self.up = [uprelu, upconv, upnorm]

            model = self.down + self.up
                        
            if self.add_view_pooling_before_x and CHANGE_INPUT:                                
                self.add_pooling_layer(inner_nc, height, width, gpu_ids=gpu_ids)
                model = self.down + self.up

            self.down = nn.Sequential(*self.down)
            self.up = nn.Sequential(*self.up)
                 
        else:

            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                            kernel_size=4, stride=2,
                            padding=1, bias=use_bias)
            
            if self.add_view_pooling_before_x and CHANGE_INPUT:                
                downconv = nn.Conv2d(input_nc*2, inner_nc, kernel_size=4,
                            stride=2, padding=1, bias=use_bias)

            if self.add_view_pooling_after_x:                
                upconv = nn.ConvTranspose2d(inner_nc * 3, outer_nc,
                            kernel_size=4, stride=2,
                            padding=1, bias=use_bias)
           
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

            if self.add_view_pooling_before_x and CHANGE_INPUT:
                self.add_pooling_layer(inner_nc, height, width, gpu_ids=gpu_ids)
                            
        self.model = nn.Sequential(*model)
    
    def add_pooling_layer(self,inner_nc, height, width, gpu_ids=[]):
       
                
        if USE_AVG_POOLING:
            self.pooling0 = nn.AvgPool2d((self.VIEW_NUM, 1), stride=(self.VIEW_NUM,1))
            self.pooling1 = nn.AvgPool2d((self.VIEW_NUM, 1), stride=(self.VIEW_NUM,1))           
            self.pooling2 = nn.AvgPool2d((self.VIEW_NUM, 1), stride=(self.VIEW_NUM,1))           
            self.pooling3 = nn.AvgPool2d((self.VIEW_NUM, 1), stride=(self.VIEW_NUM,1))
            self.pooling4 = nn.AvgPool2d((self.VIEW_NUM, 1), stride=(self.VIEW_NUM,1))
            self.pooling5 = nn.AvgPool2d((self.VIEW_NUM, 1), stride=(self.VIEW_NUM,1))           
            self.pooling6 = nn.AvgPool2d((self.VIEW_NUM, 1), stride=(self.VIEW_NUM,1))           
            self.pooling7 = nn.AvgPool2d((self.VIEW_NUM, 1), stride=(self.VIEW_NUM,1))           
        elif USE_MAX_POOLING:
            self.pooling0 = nn.MaxPool2d((self.VIEW_NUM, 1), stride=(self.VIEW_NUM,1 ))
            self.pooling1 = nn.MaxPool2d((self.VIEW_NUM, 1), stride=(self.VIEW_NUM,1 ))
            self.pooling2 = nn.MaxPool2d((self.VIEW_NUM, 1), stride=(self.VIEW_NUM,1 ))
            self.pooling3 = nn.MaxPool2d((self.VIEW_NUM, 1), stride=(self.VIEW_NUM,1 ))
            self.pooling4 = nn.MaxPool2d((self.VIEW_NUM, 1), stride=(self.VIEW_NUM,1 ))
            self.pooling5 = nn.MaxPool2d((self.VIEW_NUM, 1), stride=(self.VIEW_NUM,1 ))
            self.pooling6 = nn.MaxPool2d((self.VIEW_NUM, 1), stride=(self.VIEW_NUM,1 ))
            self.pooling7 = nn.MaxPool2d((self.VIEW_NUM, 1), stride=(self.VIEW_NUM,1 ))
            
        if len(gpu_ids):
            self.pooling0  = init_net(self.pooling0, init_type='normal', init_gain=0.02, gpu_ids=[gpu_ids[0]])        
        if len(gpu_ids)>1:
            self.pooling1  = init_net(self.pooling1, init_type='normal', init_gain=0.02, gpu_ids=[gpu_ids[1]])
        if len(gpu_ids)>2:
            self.pooling2  = init_net(self.pooling2, init_type='normal', init_gain=0.02, gpu_ids=[gpu_ids[2]])
        if len(gpu_ids)>3:
            self.pooling3  = init_net(self.pooling3, init_type='normal', init_gain=0.02, gpu_ids=[gpu_ids[3]])
        if len(gpu_ids)>4:
            self.pooling4  = init_net(self.pooling4, init_type='normal', init_gain=0.02, gpu_ids=[gpu_ids[4]])        
        if len(gpu_ids)>5:
            self.pooling5  = init_net(self.pooling5, init_type='normal', init_gain=0.02, gpu_ids=[gpu_ids[5]])
        if len(gpu_ids)>6:
            self.pooling6  = init_net(self.pooling6, init_type='normal', init_gain=0.02, gpu_ids=[gpu_ids[6]])
        if len(gpu_ids)>7:
            self.pooling7  = init_net(self.pooling7, init_type='normal', init_gain=0.02, gpu_ids=[gpu_ids[7]])

    def load_parameters(self, model_id, view_id):

        if self.innermost:
            self.model_id = model_id
            self.view_id = view_id
        else:
            self.model_id = model_id
            self.view_id = view_id
            self.submodule.load_parameters(model_id, view_id)

    def transform_view_id(self, org_id):
        #1，3，5=0，1，2
        #1，3，5，6，7=0，1，2，3，4
        if self.VIEW_NUM==8: 
            return org_id
        elif self.VIEW_NUM==3: 
            return org_id % 3
        elif self.VIEW_NUM==5:
            if org_id>=5:
                return org_id-3
            elif org_id==1:
                return 0
            elif org_id==3:
                return 1

    def calc_view_pooling(self, x):

        batch=x.shape[0]
        channel = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]
    
        batch_code = []

        group_num = int(batch/self.VIEW_NUM)
        batch_code=[]
        for i in range(group_num):

            start_index = i*self.VIEW_NUM
            end_index = (i+1)*self.VIEW_NUM

            block = x[start_index:end_index,:,:,:]#.clone().detach()

            if x.get_device()==0:
                dy_memory_code = self.pooling0(block.view(self.VIEW_NUM, channel*height*width).unsqueeze(0).unsqueeze(0)).view(1, channel, height, width) #8*(512*2*2)
            elif x.get_device()==1:
                dy_memory_code = self.pooling1(block.view(self.VIEW_NUM, channel*height*width).unsqueeze(0).unsqueeze(0)).view(1, channel, height, width) #8*(512*2*2)
            elif x.get_device()==2:
                dy_memory_code = self.pooling2(block.view(self.VIEW_NUM, channel*height*width).unsqueeze(0).unsqueeze(0)).view(1, channel, height, width) #8*(512*2*2)
            elif x.get_device()==3:
                dy_memory_code = self.pooling3(block.view(self.VIEW_NUM, channel*height*width).unsqueeze(0).unsqueeze(0)).view(1, channel, height, width) #8*(512*2*2)
            elif x.get_device()==4:
                dy_memory_code = self.pooling4(block.view(self.VIEW_NUM, channel*height*width).unsqueeze(0).unsqueeze(0)).view(1, channel, height, width) #8*(512*2*2)
            elif x.get_device()==5:
                dy_memory_code = self.pooling5(block.view(self.VIEW_NUM, channel*height*width).unsqueeze(0).unsqueeze(0)).view(1, channel, height, width) #8*(512*2*2)
            elif x.get_device()==6:
                dy_memory_code = self.pooling6(block.view(self.VIEW_NUM, channel*height*width).unsqueeze(0).unsqueeze(0)).view(1, channel, height, width) #8*(512*2*2)
            elif x.get_device()==7:
                dy_memory_code = self.pooling7(block.view(self.VIEW_NUM, channel*height*width).unsqueeze(0).unsqueeze(0)).view(1, channel, height, width) #8*(512*2*2)
        
            if not i:
                batch_code = dy_memory_code.repeat(self.VIEW_NUM,1,1,1)
            else:
                batch_code = torch.cat([batch_code, dy_memory_code.repeat(self.VIEW_NUM,1,1,1)],0)            
                            
        return batch_code

    def forward(self, x):
        if self.innermost:
                                
            if self.add_view_pooling_before_x and CHANGE_INPUT: # add views. # expand channel in next upconv.
                # do pooling. input 512*(2*2)                                
                view_pooling = self.calc_view_pooling(x)  #1, 512, 1, 1
                x_in = torch.cat([x, view_pooling], 1)
                return torch.cat([x_in, self.model(x_in)], 1)
         
            else:
                code = self.down (x)
                res=self.up(code)                
                return torch.cat([x, res], 1)
        elif self.outermost:
            x = x.view(x.shape[0]*self.VIEW_NUM, self.input_c, self.resolution, self.resolution)
            return self.model(x)
        else:
             if self.add_view_pooling_before_x and CHANGE_INPUT: # add views. # expand channel in next upconv.
                view_pooling = self.calc_view_pooling(x)  #1, 512, 1, 1
                                    
                x_in = torch.cat([x, view_pooling], 1)
                                                          
                return torch.cat([x_in, self.model(x_in)], 1)
             elif self.add_view_pooling_before_x and COMBINE_POOLING_INPUT:
                view_pooling = self.calc_view_pooling(x)  #1, 512, 4, 4

                conv_input = torch.cat([x, view_pooling], 1)
                real_input = self.pix_net(conv_input)
                
                return torch.cat([real_input, self.org_net(real_input)], 1)
             else:
                return torch.cat([x, self.model(x)], 1)

   

class UnetGenerator_Down_Up(nn.Module):

    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

    def forward(self, input):
        return self.model(input)



class Unet_One(nn.Module):
     def __init__(self, model):
        super(Unet_One, self).__init__()
        self.model = model

     def forward(self, input):
         return self.model(input)

