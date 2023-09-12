import argparse
import tasks


def modify_command_options(opts):
    if opts.dataset == 'voc':
        opts.num_classes = 21
    elif opts.dataset == 'coco':
        opts.num_classes = 80
    elif opts.dataset == 'cub':
        opts.num_classes = 200

    if not opts.visualize:
        opts.sample_num = 0
    
    if opts.backbone is None:
        opts.backbone = 'resnet101'

    if opts.dataset == "coco-voc":
        opts.backbone = 'wider_resnet38_a2'
        opts.output_stride = 8
        opts.crop_size = 448
        opts.crop_size_val = 512

    opts.no_overlap = not opts.overlap
    opts.pooling = opts.crop_size // opts.output_stride

    return opts

def modify_fss_command_options(opts):
    # modify argparse arguments for the WSCI Few-Shot Segmentation experiments
    if opts.backbone is None:
        opts.backbone = 'resnet101'

    if opts.batch_size == -1:
        if opts.step == 0:
            opts.batch_size = 12
        else:
            opts.batch_size = 10
    
    opts.no_overlap = not opts.overlap
    opts.pooling = opts.crop_size // opts.output_stride
    opts.crop_size_test = 500 if opts.dataset == 'voc' else 640
    opts.test_batch_size = 1

    return opts

def get_argparser():
    parser = argparse.ArgumentParser()

    # Performance Options
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--num_workers", type=int, default=1,
                        help='number of workers (default: 1)')
    parser.add_argument("--device", type=int, default=None,
                        help='Device ID')
    parser.add_argument("--launcher", 
                        choices=['pytorch', 'slurm'],
                        default='pytorch',
                        help='job launcher')

    # Datset Options
    parser.add_argument("--data_root", type=str, default='data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc', help='Name of dataset')
    parser.add_argument("--weakly", default=False, action='store_true')
    parser.add_argument("--num_classes", type=int, default=None, help="num classes (default: None)")

    # Train Options
    parser.add_argument("--epochs", type=int, default=30,
                        help="epoch number (default: 30)")

    parser.add_argument("--batch_size", type=int, default=24,
                        help='batch size (default: 24)')
    parser.add_argument("--crop_size", type=int, default=512,
                        help="crop size (default: 512)")
    parser.add_argument("--crop_size_val", type=int, default=512,
                        help="crop size (default: 512)")

    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help='momentum for SGD (default: 0.9)')
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')

    parser.add_argument("--lr_policy", type=str, default='poly',
                        choices=['poly', 'step', 'none', 'warmup', 'one_cycle'],
                        help="lr schedule policy (default: poly)")
    parser.add_argument("--lr_decay_step", type=int, default=5000,
                        help="decay step for stepLR (default: 5000)")
    parser.add_argument("--lr_decay_factor", type=float, default=0.1,
                        help="decay factor for stepLR (default: 0.1)")
    parser.add_argument("--lr_power", type=float, default=0.9,
                        help="power for polyLR (default: 0.9)")
    parser.add_argument("--bce", default=False, action='store_true',
                        help="Whether to use BCE or not (default: no)")

    # Validation Options
    parser.add_argument("--val_on_trainset", action='store_true', default=False,
                        help="enable validation on train set (default: False)")
    parser.add_argument("--crop_val", action='store_false', default=True,
                        help='do crop for validation (default: True)')

    # Logging Options
    parser.add_argument("--logdir", type=str, default='./logs',
                        help="path to Log directory (default: ./logs)")
    parser.add_argument("--name", type=str, default='Experiment',
                        help="name of the experiment - to append to log directory (default: Experiment)")
    parser.add_argument("--sample_num", type=int, default=8,
                        help='number of samples for visualization (default: 0)')
    parser.add_argument("--debug", action='store_true', default=False,
                        help="verbose option")
    parser.add_argument("--visualize", action='store_false', default=True,
                        help="visualization on tensorboard (def: Yes)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=5,
                        help="epoch interval for eval (default: 1)")
    parser.add_argument("--visualize_images", action='store_true', default=False,
                        help="visualization images during inference only")

    # Model Options
    parser.add_argument("--backbone", type=str, default='resnet101',
                        choices=['resnet50', 'resnet101', 'wider_resnet38_a2'],
                        help='backbone for the body (def: resnet50)')
    parser.add_argument("--output_stride", type=int, default=16,
                        choices=[8, 16], help='stride for the backbone (def: 16)')
    parser.add_argument("--no_pretrained", action='store_true', default=False,
                        help='Wheather to use pretrained or not (def: True)')
    parser.add_argument("--norm_act", type=str, default="iabn_sync",
                        help='Which BN to use (def: abn_sync')
    parser.add_argument("--pooling", type=int, default=32,
                        help='pooling in ASPP for the validation phase (def: 32)')

    # Test and Checkpoint options
    parser.add_argument("--test", action='store_true', default=False,
                        help="Whether to train or test only (def: train and test)")
    parser.add_argument("--ckpt", default=None, type=str,
                        help="path to trained model. Leave it None if you want to retrain your model")
    parser.add_argument("--continue_ckpt", default=False, action='store_true',
                        help="Restart from the ckpt. Named taken automatically from method name.")
    parser.add_argument("--ckpt_interval", type=int, default=1,
                        help="epoch interval for saving model (default: 1)")
    parser.add_argument("--ckpt_root", default=None, type=str,
                        help="path to the root of the checkpoints directory")

    # Parameters for Knowledge Distillation of ILTSS (https://arxiv.org/abs/1907.13372)
    parser.add_argument("--freeze", action='store_true', default=False,
                        help="Use this to freeze the feature extractor in incremental steps")
    parser.add_argument("--loss_de", type=float, default=0.,  # Distillation on Encoder
                        help="Set this hyperparameter to a value greater than "
                             "0 to enable distillation on Encoder (L2)")
    parser.add_argument("--loss_kd", type=float, default=0.,  # Distillation on Output
                        help="Set this hyperparameter to a value greater than "
                             "0 to enable Knowlesge Distillation (Soft-CrossEntropy)")

    # Arguments for ICaRL (from https://arxiv.org/abs/1611.07725)
    parser.add_argument("--icarl", default=False, action='store_true',
                        help="If enable ICaRL or not (def is not)")
    parser.add_argument("--icarl_importance", type=float, default=1.,
                        help="the regularization importance in ICaRL (def is 1.)")
    parser.add_argument("--icarl_disjoint", action='store_true', default=False,
                        help="Which version of icarl is to use (def: combined)")
    parser.add_argument("--icarl_bkg", type=float, default=-1,
                        help="Background interpolation (1 is new gt)")

    # METHODS
    parser.add_argument("--init_balanced", default=False, action='store_true',
                        help="Enable Background-based initialization for new classes")
    parser.add_argument("--unkd", default=False, action='store_true',
                        help="Enable Unbiased Knowledge Distillation instead of Knowledge Distillation")
    parser.add_argument("--unce", default=False, action='store_true',
                        help="Enable Unbiased Cross Entropy instead of CrossEntropy")

    # Incremental parameters
    parser.add_argument("--task", type=str, default="19-1", choices=tasks.get_task_list(),
                        help="Task to be executed (default: 19-1)")
    parser.add_argument("--step", type=int, default=0,
                        help="The incremental step in execution (default: 0)")
    parser.add_argument("--no_mask", action='store_true', default=False,
                        help="Use this to not mask the old classes in new training set") # equivalent to --masking in the FSS paper
    parser.add_argument("--overlap", action='store_true', default=False,
                        help="Use this to not use the new classes in the old training set")
    parser.add_argument("--step_ckpt", default=None, type=str,
                        help="path to trained model at previous step. Leave it None if you want to use def path")
    parser.add_argument('--curr_step_ckpt', default=None, type=str,
                        help='path to the trained model at the current step. Only used if the user wants to just run inference \
                        using pretrained checkpoints.')
    parser.add_argument("--replay", action='store_true', default=False,
                        help="Use this to replay all the old class training images")
    parser.add_argument("--replay_size", type=int, default=100,
                        help="number of old class images from 0:(t-1) to be replayed \
                        during an incremental step t")
    parser.add_argument('--semantic_similarity', action='store_true', default=False,
                        help='semantic similarity loss')
    parser.add_argument('--similarity_type', default='transformer', choices=['transformer', 'glove', 'tree'],
                        help='kinds of semantic similarity')
    parser.add_argument('--lambda_sem', type=float, default=1,
                        help='semantic similarity loss weight')
    parser.add_argument('--tau', type=float, default=5,
                        help='dampening term in the semantic similarity metric')

    # FSS parameters
    parser.add_argument('--nshot', type=int, default=5,
                        help='If step>0 the shot to use for FSS (default=5)')
    parser.add_argument('--ishot', type=int, default=0,
                        help='First index where to sample shots')
    parser.add_argument('--input_mix', default='novel', choices=['novel', 'both'],
                        help='Which class to use for FSS')
    parser.add_argument("--iter", type=int, default=None,
                        help="iteration number (default: None)\n THIS OVERWRITE --EPOCHS! Its a FSS parameter.")
    
    # external data set params
    parser.add_argument("--external_dataset", action='store_true', default=False,
                       help='use an external data set (relevant ImageNet classes) as an substitute of memory replay')
    parser.add_argument("--external_size", type=int, default=100,
                        help='number of sample from the external data set images to be replayed in an incremental step')
    parser.add_argument("--external_rootdir", type=str, default='/path/to/dataroot/',
                        help='path to the root dir of external data set (ImageNet)')
    parser.add_argument("--external_mode", type=str, choices=['old', 'new', 'both'], default='old',
                        help='how to use the external images. Old means replay old class images,\
                        new means use external images for the new classes only and both means both of the above')

    # Weakly supervised Pars
    parser.add_argument("--pseudo", default=None, type=str,
                        help="Pseudo labels for steps>0")
    parser.add_argument("--pl_ckpt", default=None, type=str,
                        help="path to pseudolabeler")
    parser.add_argument("--alpha", default=0.5, type=float,
                        help="The parameter to hard-ify the soft-labels. Def is 1.")
    parser.add_argument("--pos_w", type=float, default=1.,
                        help="Positive weight")
    parser.add_argument("--affinity", action='store_true', default=False,
                        help="Use affinity on CAM")
    parser.add_argument("--pseudo_ep", default=5, type=int,
                        help="When to start pseudolabeling (Default is 5)")
    parser.add_argument("--lr_pseudo", default=0.01, type=float,
                        help="learning rate pseudolabeler")
    parser.add_argument("--lr_head", default=10., type=float,
                        help="learning rate pseudolabeler")
    parser.add_argument("--cam", default="ngwp", type=str, choices=['ngwp', 'att', 'none', 'sem-ngwp'],
                        help="CAM model used")
    parser.add_argument("--ss_dist", action='store_true', default=False,
                        help="Dist on bkg prior")
    parser.add_argument("--l_seg", type=float, default=1)
    parser.add_argument("--pl_threshold", type=float, default=0.7,
                        help="threshold value for pseudo-labels (default: 0.7)")
    parser.add_argument("--ws_bkg", action="store_true", default=False,
                        help="Use this to include the bkg class in the weakly supervised cam loss")

    return parser
