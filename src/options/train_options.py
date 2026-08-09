from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # self.parser.add_argument('--display_freq', type=int, default=1000, help='frequency of showing training results on screen') # 10240000
        # self.parser.add_argument('--print_freq', type=int, default=1000, help='frequency of showing training results on console') # 10240000
        self.parser.add_argument('--save_images_loss_freq', type=int, default=1000, help='frequency of saving images and loss')
        self.parser.add_argument('--save_latest_freq', type=int, default=200, help='frequency of saving the latest results')
        self.parser.add_argument('--save_metrics_freq', type=int, default=200, help='frequency of saving metrics results') # 50000
        self.parser.add_argument('--metrics_dir', type=str, default='/mimer/NOBACKUP/groups/snic2022-5-277/piacente/OUTPUT_RETE', help='metrics are saved here')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1000, help='frequency of saving checkpoints at the end of epochs (pesi)')

        self.parser.add_argument('--patience', type=int, default=100, help='patience for early stopping - number of no improvement consecutive epochs )')
        #self.parser.add_argument('--warm_up_epochs', type=int, default=200, help= 'epoca da cui parte early stopping')

        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        #self.parser.add_argument('--district', type=list, default=['body','head'], help='head, body, arms, legs') # in default c'era arms
        #self.parser.add_argument('--initial_district', type=str, default='head', help= 'distretto iniziale')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model') # latest
        self.parser.add_argument('--niter', type=int, default=100000, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100000, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--identity', type=float, default=0.0, help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--random_district', action='store_true', help='distretti in ordine randomico')
        self.parser.add_argument('--warm_up_variabile', action='store_true', help='epoche di warm up da cui far partire early stopping')
        #self.parser.add_argument('--grouped_district', action='store_true', help='continue training: load the latest model')
        # self.parser.add_argument('--included_districts', type=list, default=['adrenal_gland','thyroid','gallbladder','bladder','kidney','trachea','pancreas','spleen','brain','stomach','lung','liver','arms','legs'], help='included districts') # SORTED
        # self.parser.add_argument('--included_districts', type=list,default=['adrenal_gland','kidney','pancreas','arms','thyroid','spleen','liver','lung','stomach','brain','trachea','gallbladder','legs','bladder'],help='included districts') # RANDOM
        self.parser.add_argument('--included_districts', type=list, default=['adrenal_gland', 'gallbladder', 'thyroid', 'bladder', 'trachea', 'kidney','spleen', 'pancreas', 'stomach', 'brain', 'lung', 'liver', 'arms','legs'],help='included districts')  # SORTED+GROUPED / SORTED+GROUPED+WARMUP

        self.parser.add_argument('--switch_counter', type=int, default=1,help='switch counter')

        args, _ = self.parser.parse_known_args()
        if args.warm_up_variabile and args.grouped_district:
            self.parser.add_argument('--warm_up_epochs', type=list, default=[200, 200, 203, 205, 209, 216, 221, 227, 261, 263, 400], help='epoca da cui parte early stopping') # [200, 200, 203, 205, 209, 216, 221, 227, 261, 263, 400]
        else:
            self.parser.add_argument('--warm_up_epochs', type=int, default=200, help='epoca da cui parte early stopping')

        self.isTrain = True
