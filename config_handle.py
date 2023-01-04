import argparse
import configparser

def get_config_strgqn(config):
    # Fill the parameters
    args = lambda: None
    # Model Parameters
    args.w = config.getint('model', 'w')
    args.v = (config.getint('model', 'v_h'), config.getint('model', 'v_w'))
    args.c = config.getint('model', 'c')
    args.ch = config.getint('model', 'ch')
    args.down_size = config.getint('model', 'down_size')
    args.draw_layers = config.getint('model', 'draw_layers')
    args.share_core = config.getboolean('model', 'share_core')
    
    if config.has_option('model', 'vsize'):
        args.vsize = config.getint('model', 'vsize')
    else:
        args.vsize = 7

    if config.has_option('model', 'stocahstic_unit'):
        args.stochastic_unit = config.getboolean('model', 'stocahstic_unit')
    else:
        args.stochastic_unit = True
    
    if config.has_option('model', 'loss_type'):
        args.loss_type = config.get('model', 'loss_type')
    else:
        args.loss_type = "MSE"
    
    # Experimental Parameters
    args.data_path = config.get('exp', 'data_path')
    args.frac_train = config.getfloat('exp', 'frac_train')
    args.frac_test = config.getfloat('exp', 'frac_test')
    args.max_obs_size = config.get('exp', 'max_obs_size')
    args.total_steps = config.getint('exp', 'total_steps')
    args.kl_scale = config.getfloat('exp', 'kl_scale')
    
    if config.has_option('exp', 'convert_bgr'):
        args.convert_rgb = config.getboolean('exp', 'convert_bgr')
    else:
        args.convert_rgb = True
    
    if config.has_option('exp', 'distort_type'):
        args.distort_type = config.get('exp', 'distort_type')
        if args.distort_type == "None":
            args.distort_type = None
    else:
        args.distort_type = None
    
    if config.has_option('exp', 'view_trans'):
        args.view_trans = config.getboolean('exp', 'view_trans')
    else:
        args.view_trans = True

    return args

def get_config_gqn(config):
    # Fill the parameters
    args = lambda: None

    # Model Parameters
    args.c = config.getint('model', 'c')
    args.ch = config.getint('model', 'ch')
    args.down_size = config.getint('model', 'down_size')
    args.draw_layers = config.getint('model', 'draw_layers')
    args.share_core = config.getboolean('model', 'share_core')
    if config.has_option('model', 'vsize'):
        args.vsize = config.getint('model', 'vsize')
    else:
        args.vsize = 7

    if config.has_option('model', 'stocahstic_unit'):
        args.stochastic_unit = config.getboolean('model', 'stocahstic_unit')
    else:
        args.stochastic_unit = True
    
    if config.has_option('model', 'view_trans'):
        args.view_trans = config.getboolean('model', 'view_trans')
    else:
        args.view_trans = True
    
    if config.has_option('model', 'loss_type'):
        args.loss_type = config.get('model', 'loss_type')
    else:
        args.loss_type = "MSE"

    # Experimental Parameters
    args.data_path = config.get('exp', 'data_path')
    args.frac_train = config.getfloat('exp', 'frac_train')
    args.frac_test = config.getfloat('exp', 'frac_test')
    args.max_obs_size = config.get('exp', 'max_obs_size')
    args.total_steps = config.getint('exp', 'total_steps')
    args.kl_scale = config.getfloat('exp', 'kl_scale')
    
    if config.has_option('exp', 'convert_bgr'):
        args.convert_rgb = config.get('exp', 'convert_bgr')
    else:
        args.convert_rgb = True
    
    if config.has_option('exp', 'distort_type'):
        args.distort_type = config.get('exp', 'distort_type')
        if args.distort_type == "None":
            args.distort_type = None
    else:
        args.distort_type = None
    
    if config.has_option('exp', 'view_trans'):
        args.view_trans = config.getboolean('exp', 'view_trans')
    else:
        args.view_trans = True

    return args

def load_eval_config(exp_path):
    config_file = exp_path + "config.conf"
    config = configparser.ConfigParser()
    config.read(config_file)
    args = get_config(config)
    args.img_size = (args.v[0]*args.down_size, args.v[1]*args.down_size)
    return args