from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')#5000
        parser.add_argument('--save_epoch_freq', type=int, default=50, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        #parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate') #100
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.00002, help='initial learning rate for adam') #
        parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]\\[opt.name]\\web\\')
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--isLoad', type=int, default=0, help='continue train')
        parser.add_argument('--load_file_name', type=str, default='0', help='load_file name')

        parser.add_argument('--use_gan', type=int, default=0, help='whether use discriminator')
        parser.add_argument('--flag', type=str, default='train', help='train, val, test, etc') #load dataset
        parser.add_argument('--weight_l_loss',  type=float, default = 1, help='10, 100, 200')
        parser.add_argument('--d_lr_rate', type=float, help='use with gan')

        parser.add_argument('--epoch_from_last', type=int, default=0, help='start point from last training, in disk')

        parser.add_argument('--consistent_results_dir', default='consistent_results', help='save consistency results')



        #consistent
        parser.add_argument('--vote_input', type = int, default=1, help='whether consider input in voting')
        parser.add_argument('--org_trained_model', type = str, help='org loaded models')

        self.isTrain = True
        return parser
