import numpy as np
import os
import sys
import ntpath
import time
from . import util
from . import html
from scipy.misc import imresize
from config import *
import cv2
import imageio
import torch
import torchvision.transforms as transforms

import projections.operations as operations

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

def get_depth_channel(img):
    dep = np.ones((img.shape[0], img.shape[1], 3), dtype = 'uint8')

    for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                dep[i][j][0]=dep[i][j][1]= dep[i][j][2]= img[i][j][2]
    return dep

def remove_model_id_name(fileName):
    s = fileName.split('_')
    #[2]='1'
    if len(s)==5:
        return 'none'
    return '%s_%s_%s' % (s[0], s[1], s[3])


def save_reconstruction_coord(diff_web, visuals, image_path, opt, aspect_ratio=1.0, width=256):
    
    diff_image_dir = diff_web.web_dir
    image_dir = diff_image_dir

    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    diff_web.add_header(name)

    ims, txts, links = [], [], []

    count=0

    for label, im_data in visuals.items():
        
        new_file_name = name
        image_name = '%s_%s.png' % (new_file_name, label)
        exr_name = '%s_%s.exr' % (new_file_name, label)

        if label=='fake_B':
            fake_B_exr = im_data[0].permute(1, 2, 0).cpu().detach().numpy()
            count+=1
        elif label=='real_A':
            real_A_exr = im_data[0].permute(1, 2, 0).cpu().detach().numpy()
        elif label=='real_B':
            real_B_exr = im_data[0].permute(1, 2, 0).cpu().detach().numpy()
            count+=1

        save_path = os.path.join(diff_image_dir,'images')
        png_save_path = os.path.join(save_path, image_name)
        exr_save_path = os.path.join(save_path, exr_name)
        
        os.makedirs(save_path, exist_ok=True)

        if opt.output_nc==1:
            if opt.save_results:
                cv2.imwrite(exr_save_path, im_data.cpu().detach().numpy()[0][0])
        else:
            
            if opt.save_results:
                exr_im = im_data[0].permute(1, 2, 0).cpu().detach().numpy()
                cv2.imwrite(png_save_path, exr_im*255)
                if label != 'real_B' and label != 'real_A':                             
                    cv2.imwrite(exr_save_path, exr_im)
    
        ims.append(image_name)
        if count==2:
            exr_loss = np.sum(np.abs(fake_B_exr - real_B_exr))/(real_B_exr.size)
            txts.append('%s %f' % (label, exr_loss))
        else: txts.append(label)
        links.append(image_name)
    diff_web.add_images(ims, txts, links, width=width)

    if opt.make_coord_depth:

        if os.path.exists(opt.data_dir + '/%s/%s' % (opt.gt_depth_dir, name.split('_')[0])):
            gt_depth_dir = os.path.join(opt.data_dir, '%s/%s' % (opt.gt_depth_dir, name.split('_')[0])) #+ '/../%s' % opt.gt_depth_dir
        else:
            gt_depth_dir = os.path.join(opt.data_dir, '%s' % (opt.gt_depth_dir))
       
        out_dir = os.path.join(opt.dataroot) + '/../%s_%s/%s' % (opt.coord_data_prefix, opt.name, opt.flag)
        
        s = name.split('_')
        model_name = '%s_%s'%(s[0], s[1])
        model_id = '%s'%(s[2])

        if opt.texture:
            operations.make_depth_texture_pairs_from_coord(opt, fake_B_exr, real_A_exr, gt_depth_dir, out_dir, model_name, model_id, use_rgb_mask=opt.rgb_mask)
        else:
            operations.make_depth_pairs_from_coord(opt, fake_B_exr, real_A_exr, gt_depth_dir, out_dir, model_name, model_id, use_rgb_mask=opt.rgb_mask)

    return exr_loss


def save_reconstruction_depth_dm(diff_web, visuals, image_path, opt, aspect_ratio=1.0, width=256):
    
    diff_image_dir = diff_web.web_dir
    image_dir = diff_image_dir

    util.mkdir(os.path.join(diff_image_dir,'images'))

    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    diff_web.add_header(name)

    count=0

    for label, im_data_group in visuals.items():
        if label=='fake_B':
            fake_B_group = im_data_group            
        elif label=='real_A':
            real_A_group = im_data_group
        elif label=='real_B':
            real_B_group = im_data_group
    
    if not opt.save_results:
        return 0

    fake_B_group = fake_B_group.cpu().detach().numpy()            
    real_A_group = real_A_group.cpu().detach().numpy()
    real_B_group = real_B_group.cpu().detach().numpy()            

    print('fake_B_group.shape', fake_B_group.shape)

    for j in range(fake_B_group.shape[0]):

        ims, txts, links = [], [], []
        
        real_a = real_A_group[j]
        fake_b = fake_B_group[j]
        real_b = real_B_group[j]
        
        new_file_name = '%s_%s' % (name.split('_')[0], name.split('_')[1])
        if new_file_name.find('X') !=-1:
            new_file_name = new_file_name[0:new_file_name.find('X')]

        real_a_image_name = '%s_%d_real_A.png' % (new_file_name,  j)
        fake_b_image_name = '%s_%d_fake_B.png' % (new_file_name,  j)
        real_b_image_name = '%s_%d_real_B.png' % (new_file_name,  j)
        
        real_a_exr_name = '%s_%d_real_A.exr' % (new_file_name, j)        
        fake_b_exr_name = '%s_%d_fake_B.exr' % (new_file_name, j)
        real_b_exr_name = '%s_%d_real_B.exr' % (new_file_name, j)

        
        real_a_png_save_path = os.path.join(diff_image_dir,'images', real_a_image_name)
        fake_b_png_save_path = os.path.join(diff_image_dir,'images', fake_b_image_name)
        real_b_png_save_path = os.path.join(diff_image_dir,'images', real_b_image_name)

        real_a_exr_save_path = os.path.join(diff_image_dir,'images', real_a_exr_name)
        fake_b_exr_save_path = os.path.join(diff_image_dir,'images', fake_b_exr_name)
        real_b_exr_save_path = os.path.join(diff_image_dir,'images', real_b_exr_name)        
    
        if opt.output_nc==1 and opt.save_results:                

            cv2.imwrite(real_a_png_save_path, real_a[0]*255)
            cv2.imwrite(fake_b_png_save_path, fake_b[0]*255)
            cv2.imwrite(real_b_png_save_path, real_b[0]*255)            
            
            cv2.imwrite(fake_b_exr_save_path, fake_b[0])
                
        ims.append(real_a_image_name)
        ims.append(fake_b_image_name)
        ims.append(real_b_image_name)

        links.append(real_a_image_name)
        links.append(fake_b_image_name)
        links.append(real_b_image_name)

        txts.append('real_a')
        txts.append('fake_b')

        exr_loss = np.sum(np.abs(fake_B_group[j] - real_B_group[j]))/(fake_B_group[j].size)
        txts.append('real_b %f' % (exr_loss))    
        diff_web.add_images(ims, txts, links, width=width)



    print(fake_B_group.shape, fake_B_group.size)
    exr_loss = np.sum(np.abs(fake_B_group - real_B_group))/(fake_B_group.size)
    return exr_loss


def save_reconstruction_depth_pix(diff_web, visuals, image_path, opt, aspect_ratio=1.0, width=256):
    
    diff_image_dir = diff_web.web_dir
    image_dir = diff_image_dir

    util.mkdir(os.path.join(diff_image_dir,'images'))

    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    diff_web.add_header(name)

    count=0

    for label, im_data_group in visuals.items():
        if label=='fake_B':
            fake_B_group = im_data_group            
        elif label=='real_A':
            real_A_group = im_data_group
        elif label=='real_B':
            real_B_group = im_data_group
    
    if not opt.save_results:
        return 0

    fake_B_group = fake_B_group.cpu().detach().numpy()[0][0]            
    real_A_group = real_A_group.cpu().detach().numpy()[0][0]
    real_B_group = real_B_group.cpu().detach().numpy()[0][0]      

    num = int(fake_B_group.shape[0]/256)
    print(num)
    for j in range(num):

        ims, txts, links = [], [], []
        
        real_a = real_A_group[j*256:(j+1)*256,:]
        fake_b = fake_B_group[j*256:(j+1)*256,:]
        real_b = real_B_group[j*256:(j+1)*256,:]
        
        new_file_name = '%s_%s' % (name.split('_')[0], name.split('_')[1])
        if new_file_name.find('X') !=-1:
            new_file_name = new_file_name[0:new_file_name.find('X')]

        real_a_image_name = '%s_%d_real_A.png' % (new_file_name,  j)
        fake_b_image_name = '%s_%d_fake_B.png' % (new_file_name,  j)
        real_b_image_name = '%s_%d_real_B.png' % (new_file_name,  j)
        
        real_a_exr_name = '%s_%d_real_A.exr' % (new_file_name, j)        
        fake_b_exr_name = '%s_%d_fake_B.exr' % (new_file_name, j)
        real_b_exr_name = '%s_%d_real_B.exr' % (new_file_name, j)

        
        real_a_png_save_path = os.path.join(diff_image_dir,'images', real_a_image_name)
        fake_b_png_save_path = os.path.join(diff_image_dir,'images', fake_b_image_name)
        real_b_png_save_path = os.path.join(diff_image_dir,'images', real_b_image_name)

        real_a_exr_save_path = os.path.join(diff_image_dir,'images', real_a_exr_name)
        fake_b_exr_save_path = os.path.join(diff_image_dir,'images', fake_b_exr_name)
        real_b_exr_save_path = os.path.join(diff_image_dir,'images', real_b_exr_name)
                            
        if opt.output_nc==1 and opt.save_results:                

            cv2.imwrite(real_a_png_save_path, real_a*255)
            cv2.imwrite(fake_b_png_save_path, fake_b*255)
            cv2.imwrite(real_b_png_save_path, real_b*255)            
            
            cv2.imwrite(fake_b_exr_save_path, fake_b)
                
        ims.append(real_a_image_name)
        ims.append(fake_b_image_name)
        ims.append(real_b_image_name)

        links.append(real_a_image_name)
        links.append(fake_b_image_name)
        links.append(real_b_image_name)

        txts.append('real_a')
        txts.append('fake_b')

        exr_loss = np.sum(np.abs(fake_B_group[j] - real_B_group[j]))/(fake_B_group[j].size)
        txts.append('real_b %f' % (exr_loss))    
        diff_web.add_images(ims, txts, links, width=width)



    print(fake_B_group.shape, fake_B_group.size)
    exr_loss = np.sum(np.abs(fake_B_group - real_B_group))/(fake_B_group.size)
    return exr_loss



def save_reconstruction_depth_texture(diff_web, visuals, image_path, opt, aspect_ratio=1.0, width=256):
    
    diff_image_dir = diff_web.web_dir
    image_dir = diff_image_dir

    util.mkdir(os.path.join(diff_image_dir,'images'))

    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    diff_web.add_header(name)

    count=0

    for label, im_data_group in visuals.items():
        if label=='fake_B':
            fake_B_group = im_data_group            
        elif label=='real_A':
            real_A_group = im_data_group
        elif label=='real_B':
            real_B_group = im_data_group

    fake_B_group = fake_B_group.permute(0, 2, 3, 1).cpu().detach().numpy()            
    real_A_group = real_A_group.permute(0, 2, 3, 1).cpu().detach().numpy()
    real_B_group = real_B_group.permute(0, 2, 3, 1).cpu().detach().numpy()            

    for j in range(fake_B_group.shape[0]):
        ims, txts, links = [], [], []
        
        real_a = real_A_group[j]
        fake_b = fake_B_group[j]
        real_b = real_B_group[j]
        
        new_file_name = '%s_%s' % (name.split('_')[0], name.split('_')[1])
        if new_file_name.find('X') !=-1:
            new_file_name = new_file_name[0:new_file_name.find('X')]

        real_a_image_name = '%s_%d_real_A.png' % (new_file_name,  j)
        fake_b_image_name = '%s_%d_fake_B.png' % (new_file_name,  j)
        real_b_image_name = '%s_%d_real_B.png' % (new_file_name,  j)
        
        real_a_exr_name = '%s_%d_real_A.exr' % (new_file_name, j)        
        fake_b_exr_name = '%s_%d_fake_B.exr' % (new_file_name, j)
        real_b_exr_name = '%s_%d_real_B.exr' % (new_file_name, j)

        
        real_a_png_save_path = os.path.join(diff_image_dir,'images', real_a_image_name)
        fake_b_png_save_path = os.path.join(diff_image_dir,'images', fake_b_image_name)
        real_b_png_save_path = os.path.join(diff_image_dir,'images', real_b_image_name)

        real_a_exr_save_path = os.path.join(diff_image_dir,'images', real_a_exr_name)
        fake_b_exr_save_path = os.path.join(diff_image_dir,'images', fake_b_exr_name)
        real_b_exr_save_path = os.path.join(diff_image_dir,'images', real_b_exr_name)
                            
    
        if opt.save_results: # 4 channel.

            cv2.imwrite(real_a_png_save_path, real_a[:,:,0:3]*255)
            cv2.imwrite(fake_b_png_save_path, fake_b[:,:,0:3]*255)
            cv2.imwrite(real_b_png_save_path, real_b[:,:,0:3]*255)

            if True:
                fake_b = np.concatenate((fake_b[:,:,0], fake_b[:,:,1], fake_b[:,:,2], fake_b[:,:,3]), 1)
                cv2.imwrite(fake_b_exr_save_path, fake_b)            
                
        ims.append(real_a_image_name)
        ims.append(fake_b_image_name)
        ims.append(real_b_image_name)

        links.append(real_a_image_name)
        links.append(fake_b_image_name)
        links.append(real_b_image_name)

        txts.append('real_a')
        txts.append('fake_b')
        #txts.append('real_b')

        exr_loss = np.sum(np.abs(fake_B_group[j] - real_B_group[j]))/(fake_B_group[j].size)
        txts.append('real_b %f' % (exr_loss))    
        diff_web.add_images(ims, txts, links, width=width)


    print(fake_B_group.shape, fake_B_group.size)
    exr_loss = np.sum(np.abs(fake_B_group - real_B_group))/(fake_B_group.size)
    return exr_loss

# save image to the disk
def save_images(diff_web, visuals, image_path, opt, aspect_ratio=1.0, width=256):
    
    if opt.model =='Coord':
        return save_reconstruction_coord(diff_web, visuals, image_path, opt,  aspect_ratio=1.0, width=256)
    elif opt.model =='RecShapeMemory': #dynamic memory
        return save_reconstruction_depth_dm(diff_web, visuals, image_path, opt,  aspect_ratio=1.0, width=256)
    elif opt.model =='Depth': #pix2pix
        return save_reconstruction_depth_pix( diff_web, visuals, image_path, opt, aspect_ratio=1.0, width=256)        
    elif opt.model == 'RecTextureDepth' or opt.model == 'PixDepthTexture':
        return save_reconstruction_depth_texture(diff_web, visuals, image_path, opt,  aspect_ratio=1.0, width=256)
    
    
class Visualizer():
    def __init__(self, opt):
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.opt = opt
        self.saved = False
        if self.display_id > 0:
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env, raise_exceptions=True)

        if self.use_html:            
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web_pix')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    def throw_visdom_connection_error(self):
        print('\n\nCould not connect to Visdom server (https://github.com/facebookresearch/visdom) for displaying training progress.\nYou can suppress connection to Visdom using the option --display_id -1. To install visdom, run \n$ pip install visdom\n, and start the server by \n$ python -m visdom.server.\n\n')
        exit(1)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, save_result):
        if self.display_id > 0:  # show images in the browser
            ncols = self.ncols
            if ncols > 0:
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.throw_visdom_connection_error()

            else:
                idx = 1
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1

        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # losses: dictionary of error labels and values
    def plot_current_losses(self, epoch, counter_ratio, opt, losses):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.throw_visdom_connection_error()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, i, losses, t, t_data):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

        # losses: same format as |losses| of plot_current_losses
    def print_avg_losses(self, msg):
        
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % msg)
