from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        #parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        #  Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.add_argument('--lr', type=float, default=0.00002, help='initial learning rate for adam') #
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        
        parser.add_argument('--flag', type=str, default='test', help='load train or test dir')

        parser.add_argument('--gt_depth_dir', type=str, default='', help='ground truth depth dir, /test_gt_fixed_depths or /train_gt_fixed_depths')
        parser.add_argument('--coord_depth_dir', type=str, help='depth dir from coord, /test_coord_depth or /train_coord_depth')

        
        parser.add_argument('--make_coord_depth', type=int, default = 0, help='whether render depths from coord (coord) ')


        parser.add_argument('--save_results', type=int, default = 0, help='whether save final_results (depth completion)')

        parser.add_argument('--consistency_data_prefix', type=str, help='name of consistency_data_prefix')

        parser.add_argument('--coord_data_prefix', type=str, help='name of coord_data_prefix')

        parser.add_argument('--save_model_name', type=str, default="", help='name of actual')

        parser.add_argument('--rgb_mask', type=int, default=0, help='name of actual')

        parser.add_argument('--texture', type=int, default=0, help='name of actual')

        parser.add_argument('--data_start', type=int, default=0, help='start of dataset')
        parser.add_argument('--data_end', type=int, default=6, help='end of dataset')
        parser.add_argument('--total_threads', type=int, default=6, help='total parts')



        parser.set_defaults(model='test')

        # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(loadSize=parser.get_default('fineSize'))

        self.isTrain = False
        return parser
