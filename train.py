import time
import open3d
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from config import *
import os
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    opt = TrainOptions().parse()

    opt.phase = 'train'
    opt.dataroot = os.path.join(opt.data_dir, opt.dataroot)


    opt.checkpoints_dir = os.path.join(opt.data_dir, 'checkpoints')

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    writer = SummaryWriter(opt.checkpoints_dir+'/%s' % opt.name)

    model = create_model(opt)

    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    if opt.isLoad >0:
        model.load_networks('%s' % opt.load_file_name)
   
    if dataset_size>10000:
        opt.save_epoch_freq = 1    

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):

        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        avg_loss = 0.0
        cntt = 0
        print(opt.lr)
        for i, data in enumerate(dataset):
            
            iter_start_time = time.time()
            
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            
            #continue
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                print(opt.name)
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                
                avg_loss += losses['G_L1']
                cntt +=1

                writer.add_scalar('loss_iter/train(%s)' % opt.name, losses['G_L1'], total_steps)                


                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            iter_data_time = time.time()

        t = (time.time() - iter_start_time)
        print('one epoch time %f' % t)

        if epoch % opt.save_epoch_freq == 0 or epoch == opt.niter + opt.niter_decay:
            real_epoch = epoch + opt.epoch_from_last
            print('saving the model at the end of epoch %d, iters %d' % (real_epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch + opt.epoch_from_last)
            
            rm_dir = os.path.join(opt.checkpoints_dir, opt.name)
            file_name = '%s/%d_net_G.pth' % (rm_dir, real_epoch-opt.save_epoch_freq*2)
            if ( (real_epoch-opt.save_epoch_freq*2) % 50 ==0):
                continue
            if os.path.exists(file_name):
                os.remove(file_name)
        
        writer.add_scalar('loss_epoch/train(%s)' % opt.name, avg_loss/cntt, total_steps)

        print('End of epoch %d \\ %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
    writer.close()
