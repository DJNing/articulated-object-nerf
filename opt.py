import argparse
import json

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="config file for runing")
    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff', 'llff_nocs', 'google_scanned', 'objectron', 'srn', 'srn_multi', 'objectron_multi', 'nocs_bckg', 'llff_nsff', 'co3d', 'pd', 'pd_multi_obj', 'pd_multi', 'pd_multi_ae', 'srn_multi_ae', 'pd_multi_obj_ae', 'pd_multi_obj_ae_nocs', 'pd_multi_obj_ae_cv', 'sapien', 'sapien_multi', 'sapien_part'],
                        help='which dataset to train/val')
    parser.add_argument('--output_path', type=str, default='./results', help='dir to save the training results.')
    parser.add_argument('--save_path', type=str,
                        default='vanilla',
                        help='save results during eval')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[640, 480],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--white_back', default=False, action="store_true",
                        help='try for synthetic scenes like blender')
    parser.add_argument('--spheric_poses', default=True, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')
    parser.add_argument('--emb_dim', type=int, default=2458,
                        help='Total number of different objects in a category')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='dim of latent each for shape and appearance')
    parser.add_argument('--N_emb_xyz', type=int, default=10,
                        help='number of frequencies in xyz positional encoding')
    parser.add_argument('--N_emb_dir', type=int, default=4,
                        help='number of frequencies in dir positional encoding')
    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=64,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--perturb', type=float, default=1.0,
                        help='factor to perturb depth sampling points')
    parser.add_argument('--noise_std', type=float, default=1.0,
                        help='std dev of noise added to regularize sigma')

    parser.add_argument('--crop_img', default=False, action="store_true",
                        help='initially crop the image or not')
    parser.add_argument('--use_image_encoder', default=False, action="store_true",
                        help='initially crop the image or not')
    parser.add_argument('--latent_code_path', type=str, default=None,
                        help='which category to use')
    parser.add_argument('--encoder_type', type=str, default='resnet',
                        help='which category to use')
    parser.add_argument('--finetune_lpips', default=False, action="store_true",
                        help='whether to finetune with lpips loss and patched dataloader')

    # params for SRN multicat training

    parser.add_argument('--splits', type=str, default=None,
                        help='which category to use')
                        
    parser.add_argument('--run_eval', default=False, action="store_true")

    parser.add_argument('--do_generate', default=False, action="store_true")
    parser.add_argument('--val_splits', type=str, default=None,
                        help='which category to use')
    parser.add_argument('--cat', type=str, default=None,
                        help='which category to use')
    parser.add_argument('--use_tcnn', default=False, action="store_true")

    parser.add_argument('--model_type', type=str, default='geometry',
                        help='which model to use i.e. geometry or render for refnerf')
    parser.add_argument('--train_opacity_rgb', default=False, action="store_true",
                        help='whether to train both opacity and rgb for voxel model')
                

    # params for latent codes:
    # 
    parser.add_argument('--N_max_objs', type=int, default=151,
                        help='maximum number of object instances in the dataset')
    
    #onl for nerfmvs
    parser.add_argument('--nv', type=int, default=3,
                        help='maximum number of object instances in the dataset')
    
    parser.add_argument('--num_nocs_ch', type=int, default=256,
                        help='maximum number of object instances in the dataset')
    parser.add_argument('--N_obj_code_length', type=int, default=128,
                        help='size of latent vector')
    ## params for Nerf Model 
    #(Scene branch)
    parser.add_argument('--D', type=int, default=8)
    parser.add_argument('--W', type=int, default=256)
    parser.add_argument('--N_freq_xyz', type=int, default=10)
    parser.add_argument('--N_freq_dir', type=int, default=4)
    parser.add_argument('--skips', type=list, default=[4])

    ## params for Nerf Model 
    #(Obj branch)
    parser.add_argument('--inst_D', type=int, default=4)
    parser.add_argument('--inst_W', type=int, default=128)
    parser.add_argument('--inst_skips', type=list, default=[2])
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    # parser.add_argument('--chunk', type=int, default= 16*64,
    #                     help='chunk size to split the input to avoid OOM')
    parser.add_argument('--chunk', type=int, default= 16*240,
                        help='chunk size to split the input to avoid OOM')
    # parser.add_argument('--chunk', type=int, default= 32*1024,
    #                     help='chunk size to split the input to avoid OOM')
    parser.add_argument('--num_epochs', type=int, default=80,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')

    parser.add_argument('--run_max_steps', type=int, default=100000,
                        help='number of gpus')

    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--is_optimize',  type=str, default=None,
                        help='whether to finetune the network after training on prior data')
    parser.add_argument('--prefix', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'],
                        help='the prefixes to ignore in the checkpoint state dict')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained model weight to load (do not load optimizers, etc)')

    
    #### Loss params
    parser.add_argument('--color_loss_weight', type=float, default=1.0)
    parser.add_argument('--depth_loss_weight', type=float, default=0.1)
    parser.add_argument('--opacity_loss_weight', type=float, default=10.0)
    parser.add_argument('--instance_color_loss_weight', type=float, default=1.0)
    parser.add_argument('--instance_depth_loss_weight', type=float, default=1.0)

    #### object-nerf optimizer params
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type',
                        choices=['sgd', 'adam', 'radam', 'ranger'])
    # parser.add_argument('--lr', type=float, default=1.0e-3,
    #                     help='learning rate')
    parser.add_argument('--lr', type=float, default=1.0e-3,
                        help='learning rate')
    parser.add_argument('--iters', type=int, default=30000,
                        help='iters')
    # parser.add_argument('--lr', type=float, default=1.0e-4,
    #                     help='learning rate')
    parser.add_argument('--latent_lr', type=float, default=1.0e-3,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='poly',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    parser.add_argument('--lr_scheduler_latent', type=str, default='poly',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')

    #### nerf_pl configs
    # parser.add_argument('--optimizer', type=str, default='adam',
    #                     help='optimizer type',
    #                     choices=['sgd', 'adam', 'radam', 'ranger'])
    # parser.add_argument('--lr', type=float, default=5e-4,
    #                     help='learning rate')
    # parser.add_argument('--momentum', type=float, default=0.9,
    #                     help='learning rate momentum')
    # parser.add_argument('--weight_decay', type=float, default=0,
    #                     help='weight decay')
    # parser.add_argument('--lr_scheduler', type=str, default='steplr',
    #                     help='scheduler type',
    #                     choices=['steplr', 'cosine', 'poly'])
    # #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    # parser.add_argument('--warmup_multiplier', type=float, default=1.0,
    #                     help='lr is multiplied by this factor after --warmup_epochs')
    # parser.add_argument('--warmup_epochs', type=int, default=0,
    #                     help='Gradually warm-up(increasing) learning rate in optimizer')
    ###########################
    #### params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int, default=[20],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='learning rate decay amount')
    ###########################
    #### params for poly ####
    parser.add_argument('--poly_exp', type=float, default=0.99,
                        help='exponent for polynomial learning rate decay')
    # parser.add_argument('--poly_exp', type=float, default=2,
    #                     help='exponent for polynomial learning rate decay')
    ###########################

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')

    parser.add_argument('--render_name', type=str, default=None,
                        help='render directory name')  

    parser.add_argument('--exp_type', type=str, default='vanilla',
                        help='experiment type --choose from vanilla, pixel_nerf, pixel_nerf_sphere, groundplanar, triplanar')

    ###########################

    # deformation mlp params
    parser.add_argument('--deform_layer_num', type=int, default=8, help="number of layers for the deformation mlp")
    parser.add_argument('--deform_layer_width', type=int, default=512, help="width for hidden layers")
    parser.add_argument('--deform_input_dim', type=int, default=12+2+1, help="input dim for deformation MLP, normally 3x4, + part_num + articulation length")
    parser.add_argument('--deform_output_dim', type=int, default=12, help="output dim for deformation mlp, normally 3x3 rotation + 3x1 translation")
    parser.add_argument('--part_num', type=int, default=2, help="number of parts for the object")
    parser.add_argument('--ray_batch', type=int, default=2048, help="number of rays used during segmentation training.")
    parser.add_argument('--freeze_nerf', type=bool, default=True, help="whether to freeze the nerf model during segmentation training")
    parser.add_argument('--nerf_ckpt', type=str, default=None, help="ckpt path for the pretrained nerf model")
    parser.add_argument('--forward_chunk', type=int, default=16*240, help="chunk size for inference, distinguish from chunk parameter above")
    parser.add_argument('--seg_mode', type=str, default='v3', help="how to set the segmentation")
    parser.add_argument('--use_part_condition', type=bool, default=True, help="whether to use conditioned segmentation")
    parser.add_argument('--use_seg_mask', type=bool, default=False, help="whether to use seg-mask rendering")
    parser.add_argument('--render_seg', type=bool, default=False, help="whether to render seg as color")
    parser.add_argument('--record_hard_sample', type=bool, default=False, help="whether to render seg as color")
    parser.add_argument('--one_hot_loss', type=bool, default=False, help="whether to use one-hot loss for sample classification")
    parser.add_argument('--one_hot_activation', type=bool, default=False, help="whether to use one-hot activation for sample classification, pred[pred>0.5] = 1, pred[pred<=0.5] = 0")
    parser.add_argument('--num_cpu', type=int, default=2, help="number of cpus to use")
    parser.add_argument('--res_raw', type=bool, default=False, help="whether to concatenate raw positions to segmentation head")
    parser.add_argument('--include_bg', type=bool, default=True, help="whether to include background during segmentation")
    parser.add_argument('--lr_final', type=float, default=1e-5, help="mininum learning rate during training")
    parser.add_argument('--use_opa_loss', type=bool, default=False, help="whether to use opacity loss")
    parser.add_argument('--use_seg_module', type=bool, default=False, help="whether to use segmentation module")
    parser.add_argument('--use_late_pose', type=bool, default=False, help="whether to use pose after the initial segmentation")
    parser.add_argument('--use_dist_reg', type=bool, default=False, help="whether to use local smoothness loss")
    parser.add_argument('--use_bg_reg', type=bool, default=False, help="whether to use background regularization loss")
    parser.add_argument('--composite_rendering', type=bool, default=False, help="whether to use composite_rendering")
    parser.add_argument('--rgb_activation', type=bool, default=False, help="whether to use rgb_activation")
    parser.add_argument('--scan_density', type=bool, default=False, help="whether to scan the nerf space to save density for visualization")
    parser.add_argument('--grid_num', type=int, default=256, help="number of grids used for density scan")
    parser.add_argument('--near', type=int, default=2, help="near plane")
    parser.add_argument('--far', type=int, default=9, help="far plane")
    return parser

def get_opts():

    parser = get_parser()

    args = parser.parse_args()
    # Load and parse the JSON configuration file
    with open(args.config, "r") as config_file:
        config_data = json.load(config_file)
        
    # Update the args namespace with loaded JSON data
    for key, value in config_data.items():
        setattr(args, key, value)
        
    return args

