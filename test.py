import os
import open3d
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
from config import *
from util.util import remove
import ntpath

if __name__ == '__main__':
    opt = TestOptions().parse()

    opt.results_dir = os.path.join(opt.data_dir, 'results')

    opt.dataroot = os.path.join(opt.data_dir, opt.dataroot)
    opt.checkpoints_dir = os.path.join(opt.data_dir, 'checkpoints')

    # hard-code some parameters for test
    opt.phase = 'test'
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    # create a website
    


    if opt.flag=='test':
        opt.rgb_mask = 1

    if opt.save_model_name=='':
        opt.save_model_name = opt.name

    web_dir = os.path.join(opt.results_dir, opt.save_model_name, '%s_%s' % (opt.flag, opt.epoch))
        
    diff_web = html.HTML(web_dir +'_' + '_pix', 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.flag, opt.epoch))
    

    # test in eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval()

    cnt = 0
    loss_sum = 0.0

    dataset_size = len(data_loader)
    start = int(opt.data_start*dataset_size/opt.total_threads)
    end   =  int(opt.data_end*dataset_size/opt.total_threads)
 
    if True:
        for i, data in enumerate(dataset):

            if i<start or i>end:
                continue            

            if opt.phase == 'test' and opt.make_coord_depth:

                image_path = data['A_paths']
                             
                if image_path[0] =='finish':
                    print('**yes, finish')             
                    continue
                
            cnt += 1 
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()

            if i % 5 == 0:
                print('processing (%04d)-th image... %s' % (i, img_path))
            loss_sum += save_images(diff_web, visuals, img_path, opt, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
            
    print(loss_sum/cnt)
    
    diff_web.add_header('avg_loss %f' % (loss_sum/cnt))
    diff_web.save()