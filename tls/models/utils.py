import logging
import os

import gin
import numpy as np
import torch


def save_model(model, optimizer, epoch, save_file):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def load_model_state(filepath, model, optimizer=None):
    state = torch.load(filepath)
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    logging.info('Loaded model and optimizer')


def save_config_file(log_dir):
    with open(os.path.join(log_dir, 'train_config.gin'), 'w') as f:
        f.write(gin.operative_config_str())


def get_bindings_and_params(args):
    gin_bindings = []
    log_dir = args.logdir
    if args.num_class:
        num_class = args.num_class
        gin_bindings += ['NUM_CLASSES = ' + str(num_class)]

    if args.horizon:
        if args.rs:
            horizon = args.horizon[np.random.randint(len(args.horizon))]
        else:
            horizon = args.horizon[0]
        gin_bindings += ['HORIZON  = ' + str(horizon)]
        log_dir = log_dir + '_horizon_' + str(horizon)

    if args.objective_type:
        if args.rs:
            objective_type = args.objective_type[np.random.randint(len(args.objective_type))]
        else:
            objective_type = args.objective_type[0]
        gin_bindings += ['OBJ_TYPE = ' + '"' + str(objective_type) + '"']
        log_dir = log_dir + '_objective-type_' + str(objective_type)

    if args.embedding_depth:
        if args.rs:
            embedding_depth = args.embedding_depth[np.random.randint(len(args.embedding_depth))]
        else:
            embedding_depth = args.embedding_depth[0]
        gin_bindings += ['EMB_DEPTH = ' + str(embedding_depth)]

        log_dir = log_dir + '_emb_' + str(embedding_depth)

    if args.l_smooth:
        if args.rs:
            l_smooth = args.l_smooth[np.random.randint(len(args.l_smooth))]
        else:
            l_smooth = args.l_smooth[0]
        gin_bindings += ['L_SMOOTH = ' + str(l_smooth)]

        log_dir = log_dir + '_l_smooth_' + str(l_smooth)

    if args.up:
        if args.rs:
            up = args.up[np.random.randint(len(args.up))]
        else:
            up = args.up[0]
        gin_bindings += ['UP = ' + str(up)]

        log_dir = log_dir + '_up-weight_' + str(up)

    if args.gamma:
        if args.rs:
            gamma = args.gamma[np.random.randint(len(args.gamma))]
        else:
            gamma = args.gamma[0]
        gin_bindings += ['GAMMA = ' + str(gamma)]

        log_dir = log_dir + '_gamma_' + str(gamma)

    if args.ad:
        if args.rs:
            ad = args.ad[np.random.randint(len(args.ad))]
        else:
            ad = args.ad[0]
        gin_bindings += ['AD = ' + str(ad)]

        log_dir = log_dir + '_ad_' + str(ad)

    if args.reg:
        if args.rs:
            reg = args.reg[np.random.randint(len(args.reg))]
            reg_weight = args.reg_weight[np.random.randint(len(args.reg_weight))]
        else:
            reg_weight = args.reg_weight[0]
            reg = args.reg[0]
        gin_bindings += ['REG_WEIGHT  = ' + str(reg_weight)]
        gin_bindings += ['REG_TYPE = ' + '"' + str(reg) + '"']

        log_dir = log_dir + '_' + str(reg) + '-reg_' + str(reg_weight)

    if args.batch_size:
        if args.rs:
            batch_size = args.batch_size[np.random.randint(len(args.batch_size))]
        else:
            batch_size = args.batch_size[0]
        gin_bindings += ['BS = ' + str(batch_size)]
        log_dir = log_dir + '_bs_' + str(batch_size)

    if args.lr:
        if args.rs:
            lr = args.lr[np.random.randint(len(args.lr))]
        else:
            lr = args.lr[0]
        gin_bindings += ['LR = ' + str(lr)]
        log_dir = log_dir + '_lr_' + str(lr)

    if args.lr_decay:
        if args.rs:
            lr_decay = args.lr_decay[np.random.randint(len(args.lr_decay))]
        else:
            lr_decay = args.lr_decay[0]
        gin_bindings += ['LR_DECAY = ' + str(lr_decay)]
        log_dir = log_dir + '_lr-decay_' + str(lr_decay)

    if args.maxlen:
        maxlen = args.maxlen
        gin_bindings += ['MAXLEN = ' + str(maxlen)]
        log_dir = log_dir + '_maxlen_' + str(maxlen)

    if args.smooth_type:
        if args.rs:
            smooth_type = args.smooth_type[np.random.randint(len(args.smooth_type))]
        else:
            smooth_type = args.smooth_type[0]
        gin_bindings += ['SMOOTH  = ' + str(True)]
        gin_bindings += ['SMOOTHING_FN  = ' + '@' + str(smooth_type)]

        log_dir = log_dir + '_smoothing_' + str(smooth_type)

    if args.h_min:
        if args.rs:
            h_min = args.h_min[np.random.randint(len(args.h_min))]
        else:
            h_min = args.h_min[0]
        if len(h_min) == 1:
            h_min = h_min[0]
            h_min_string = str(h_min)
        else:
            h_min_string = str(h_min[0])
        gin_bindings += ['H_MIN = ' + str(h_min)]
        log_dir = log_dir + '_h-min_' + str(h_min_string)

    if args.h_max:
        if args.rs:
            h_max = args.h_max[np.random.randint(len(args.h_max))]
        else:
            h_max = args.h_max[0]
        if len(h_max) == 1:
            h_max = h_max[0]
            h_max_string = str(h_max)
        else:
            h_max_string = str(h_max[0])
        gin_bindings += ['H_MAX = ' + str(h_max)]
        log_dir = log_dir + '_h-max_' + str(h_max_string)

    if args.h_true:
        if args.rs:
            h_true = args.h_true[np.random.randint(len(args.h_true))]
        else:
            h_true = args.h_true[0]
        if len(h_true) == 1:
            h_true = h_true[0]
            h_true_string = str(h_true)
        else:
            h_true_string = str(h_true[0])
        gin_bindings += ['H_TRUE = ' + str(h_true)]
        log_dir = log_dir + '_h-true_' + str(h_true_string)

    if args.emb:
        if args.rs:
            emb = args.emb[np.random.randint(len(args.emb))]
        else:
            emb = args.emb[0]
        gin_bindings += ['EMB  = ' + str(emb)]
        log_dir = log_dir + '_emb_' + str(emb)

    if args.do:
        if args.rs:
            do = args.do[np.random.randint(len(args.do))]
        else:
            do = args.do[0]
        gin_bindings += ['DO  = ' + str(do)]
        log_dir = log_dir + '_do_' + str(do)

    if args.do_att:
        if args.rs:
            do_att = args.do_att[np.random.randint(len(args.do_att))]
        else:
            do_att = args.do_att[0]
        gin_bindings += ['DO_ATT  = ' + str(do_att)]
        log_dir = log_dir + '_do-att_' + str(do_att)

    if args.depth:
        if args.rs:
            depth = args.depth[np.random.randint(len(args.depth))]
        else:
            depth = args.depth[0]

        num_leaves = 2 ** depth
        gin_bindings += ['DEPTH  = ' + str(depth)]
        gin_bindings += ['NUM_LEAVES  = ' + str(num_leaves)]
        log_dir = log_dir + '_depth_' + str(depth)

    if args.heads:
        if args.rs:
            heads = args.heads[np.random.randint(len(args.heads))]
        else:
            heads = args.heads[0]
        gin_bindings += ['HEADS  = ' + str(heads)]
        log_dir = log_dir + '_heads_' + str(heads)

    if args.latent:
        if args.rs:
            latent = args.latent[np.random.randint(len(args.latent))]
        else:
            latent = args.latent[0]
        gin_bindings += ['LATENT  = ' + str(latent)]
        log_dir = log_dir + '_latent_' + str(latent)

    if args.hidden:
        if args.rs:
            hidden = args.hidden[np.random.randint(len(args.hidden))]
        else:
            hidden = args.hidden[0]
        gin_bindings += ['HIDDEN = ' + str(hidden)]
        log_dir = log_dir + '_hidden_' + str(hidden)

    if args.loss_weight:
        if args.rs:
            loss_weight = args.loss_weight[np.random.randint(len(args.loss_weight))]
        else:
            loss_weight = args.loss_weight[0]
        if loss_weight == "None":
            gin_bindings += ['LOSS_WEIGHT = ' + str(loss_weight)]
            log_dir = log_dir + '_loss-weight_no_weight'
        elif loss_weight == 'balanced' or loss_weight == 'balanced_patient':
            log_dir = log_dir + '_loss-weight_' + str(loss_weight)
            gin_bindings += ['LOSS_WEIGHT = ' + "'" + str(loss_weight) + "'"]
        elif isinstance(loss_weight, str):
            log_dir = log_dir + '_loss-weight_' + str(loss_weight)
            gin_bindings += ['LOSS_WEIGHT = ' + str(loss_weight)]

    if args.agg_type:
        if args.rs:
            agg_type = args.agg_type[np.random.randint(len(args.agg_type))]
        else:
            agg_type = args.agg_type[0]
        gin_bindings += ['AGG_TYPE = ' + "'" + str(agg_type) + "'"]
        log_dir = log_dir + '_agg_' + str(agg_type)

    if args.repeat_step:
        if args.rs:
            repeat_step = args.repeat_step[np.random.randint(len(args.repeat_step))]
        else:
            repeat_step = args.repeat_step[0]
        gin_bindings += ['REPEAT  = ' + str(repeat_step)]
        log_dir = log_dir + '_repeat-step_' + str(repeat_step)

    if args.min_value:
        if args.rs:
            min_value = args.min_value[np.random.randint(len(args.min_value))]
        else:
            min_value = args.min_value[0]
        gin_bindings += ['MIN_VALUE  = ' + str(min_value)]
        log_dir = log_dir + '_min-value_' + str(min_value)
    return gin_bindings, log_dir
