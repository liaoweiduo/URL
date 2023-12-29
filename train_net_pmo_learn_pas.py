"""
This code allows you to train a domain classifier.

Author: Weiduo Liao
Date: 2023.11.20
"""

import os
import sys
import pickle
import copy
from datetime import datetime

import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data.meta_dataset_reader import (MetaDatasetBatchReader, MetaDatasetEpisodeReader,
                                      TRAIN_METADATASET_NAMES)
from models.losses import cross_entropy_loss, prototype_loss, DistillKL
from models.model_utils import (CheckPointer, UniformStepLR,
                                CosineAnnealRestartLR, ExpDecayLR)
from models.model_helpers import get_model, get_optimizer, get_model_moe
from models.pa import apply_selection, pa_iterator
from utils import Accumulator, device, set_determ, check_dir
from config import args

from pmo_utils import (Pool, Mixer,
                       cal_hv_loss, cal_hv, draw_objs, draw_heatmap, available_setting, check_available, task_to_device)

from debug import Debugger

import warnings
warnings.filterwarnings('ignore')

def return_time():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

def train():
    # Set seed
    set_determ(seed=1234)

    debugger = Debugger(level='INFO')

    '''--------------------'''
    '''Initialization Phase'''
    '''--------------------'''
    # defining the summary writer
    writer = SummaryWriter(check_dir(os.path.join(args['out.dir'], 'summary'), False))

    def model_train(mode):
        # train mode
        mode.train()
        if mode.feature_extractor is not None:
            mode.feature_extractor.eval()        # to extract task features
        # pool.train()

    def model_eval(mode):
        # eval mode
        mode.eval()

    url = get_model(None, args, base_network_name='url', freeze_fe=True)
    # pmo model load from url
    pmo = get_model_moe(None, args, base_network_name='url')    # resnet18_moe
    if 'film' in args['train.cond_mode']:
        params = pmo.get_trainable_film_parameters()
    elif 'pa' in args['train.cond_mode']:
        params = pmo.get_trainable_pa_parameters()
    else:
        raise Exception(f"Un-implemented train.cond_mode {args['train.cond_mode']}")
    optimizer = get_optimizer(pmo, args, params=params)
    checkpointer = CheckPointer(args, pmo, optimizer=optimizer, save_all=True)

    assert args['model.dir'] != args['out.dir']

    start_iter, best_val_loss, best_val_acc = 0, 999999999, -1
    checkpointer.restore_model(ckpt='best', strict=False, optimizer=False)
    if args['train.lr_policy'] == "step":
        lr_manager = UniformStepLR(optimizer, args, start_iter)
    elif "exp_decay" in args['train.lr_policy']:
        lr_manager = ExpDecayLR(optimizer, args, start_iter)
    else:       # elif "cosine" in args['train.lr_policy']:
        lr_manager = CosineAnnealRestartLR(optimizer, args, 0)       # start_iter

    model_eval(pmo)
    model_eval(url)

    def init_train_log():
        log = {}
        # Tag: acc/loss
        log['mo_df'] = pd.DataFrame(columns=['Tag', 'Pop_id', 'Obj_id', 'Inner_id', 'Value'])
        # 'Inner_lr', 'Exp', 'Logit_scale',
        log['scaler_df'] = pd.DataFrame(columns=['Tag', 'Idx', 'Value'])

        return log

    epoch_log = init_train_log()
    val_log = init_train_log()

    pop_labels = [
        f"p{idx}" if idx < args['train.n_obj'] else f"m{idx - args['train.n_obj']}"
        for idx in range(args['train.n_mix'] + args['train.n_obj'])
    ]  # ['p0', 'p1', 'm0', 'm1']

    '''Load training data'''
    config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.allow_growth = False
    with (tf.compat.v1.Session(config=config)) as session:
        # initialize datasets and loaders
        trainsets = TRAIN_METADATASET_NAMES
        valsets = TRAIN_METADATASET_NAMES
        testsets = TRAIN_METADATASET_NAMES
        trainsets, valsets, testsets = args['data.train'], args['data.val'], args['data.test']
        print(f'Train on: {trainsets}.')    # "ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower"
        print(f'Val on: {valsets}.')
        # print(f'Test on: {testsets}.')

        print(f'devices: {device}.')

        train_loaders = dict()
        num_train_classes = dict()
        for t_indx, trainset in enumerate(trainsets):
            train_loaders[trainset] = MetaDatasetEpisodeReader(
                'train', [trainset], valsets, testsets, test_type=args['train.type'])
            num_train_classes[trainset] = train_loaders[trainset].num_classes('train')
        print(f'num_train_classes: {num_train_classes}')
        # {'ilsvrc_2012': 712, 'omniglot': 883, 'aircraft': 70, 'cu_birds': 140, 'dtd': 33, 'quickdraw': 241,
        #  'fungi': 994, 'vgg_flower': 71}

        val_loader = MetaDatasetEpisodeReader('val', trainsets, valsets, testsets, test_type=args['test.type'])

        '''-------------'''
        '''Training loop'''
        '''-------------'''
        max_iter = args['train.max_iter']
        print(f'\n>>>> {return_time()} Train start from {start_iter}.')
        for i in tqdm(range(start_iter, max_iter), ncols=100):      # every iter, load one task from all loaders
            print(f"\n>> {return_time()} Iter: {i}, collect training samples: ")

            '''obtain tasks from train_loaders and put to buffer/clusters'''
            # loading images and labels
            for t_indx, (name, train_loader) in enumerate(train_loaders.items()):
                sample = train_loader.get_train_task(session, d=device)
                context_images, target_images = sample['context_images'], sample['target_images']
                context_labels, target_labels = sample['context_labels'], sample['target_labels']

                '''use url with pa'''
                model = url
                with torch.no_grad():
                    context_features = model.embed(context_images)
                    target_features = model.embed(target_images)

                    task_features = pmo.embed(
                        torch.cat([context_images, target_images]))
                    selection, selection_info = pmo.selector(
                        task_features, gumbel=False, hard=False)

                selection_params = [torch.mm(selection.detach(), pmo.pas.flatten(1)).view(512, 512, 1, 1)]
                # detach from selection to prevent train the selector
                selected_context = apply_selection(context_features, selection_params)
                selected_target = apply_selection(target_features, selection_params)

                task_loss, stats_dict, _ = prototype_loss(
                    selected_context, context_labels, selected_target, target_labels,
                    distance=args['test.distance'])
                epoch_log['scaler_df'] = pd.concat([
                    epoch_log['scaler_df'], pd.DataFrame.from_records([
                        {'Tag': 'task/loss', 'Idx': 0, 'Value': stats_dict['loss']},
                        {'Tag': 'task/acc', 'Idx': 0, 'Value': stats_dict['acc']}])])

                optimizer.zero_grad()
                task_loss.backward()

                '''debug'''
                debugger.print_grad(pmo, key='pas', prefix=f'iter{i} after task_loss backward:\n')

                optimizer.step()

            lr_manager.step(i)

            '''log iter-wise params change'''
            writer.add_scalar('params/learning_rate', optimizer.param_groups[0]['lr'], i + 1)

            if (i + 1) % args['train.summary_freq'] == 0:
                print(f">> {return_time()} Iter: {i + 1}, train summary:")
                '''save train_log'''
                epoch_train_history = dict()
                if os.path.exists(os.path.join(args['out.dir'], 'summary', 'train_log.pickle')):
                    epoch_train_history = pickle.load(
                        open(os.path.join(args['out.dir'], 'summary', 'train_log.pickle'), 'rb'))
                epoch_train_history[i + 1] = epoch_log.copy()
                with open(os.path.join(args['out.dir'], 'summary', 'train_log.pickle'), 'wb') as f:
                    pickle.dump(epoch_train_history, f)

                debugger.write_scaler(epoch_log['scaler_df'], key='task/loss', i=i, writer=writer)
                debugger.write_scaler(epoch_log['scaler_df'], key='task/acc', i=i, writer=writer)

                epoch_log = init_train_log()

            if (i + 1) % args['train.eval_freq'] == 0 or i == 0:    # eval at init
                print(f"\n>> {return_time()} Iter: {i + 1}, evaluation:")

                # eval mode
                model_eval(pmo)
                model_eval(url)

                '''nvidia-smi'''
                print(os.system('nvidia-smi'))

                for j in tqdm(range(args['train.eval_size']), ncols=100):
                    '''obtain 1 task from all val_loader'''
                    for v_indx, valset in enumerate(valsets):
                        samples = val_loader.get_validation_task(session, valset, d=device)
                        context_images, target_images = samples['context_images'], samples['target_images']
                        context_labels, target_labels = samples['context_labels'], samples['target_labels']
                        context_gt_labels, target_gt_labels = samples['context_gt'], samples['target_gt']
                        domain = v_indx

                        '''eval task loss use url with pa'''
                        model = url
                        with torch.no_grad():
                            context_features = model.embed(context_images)
                            target_features = model.embed(target_images)
                            task_features = pmo.embed(torch.cat([context_images, target_images]))
                            selection, selection_info = pmo.selector(
                                task_features, gumbel=False, hard=False)

                            selection_params = [torch.mm(selection, pmo.pas.flatten(1)).view(512, 512, 1, 1)]

                            selected_context = apply_selection(context_features, selection_params)
                            selected_target = apply_selection(target_features, selection_params)

                        _, stats_dict, _ = prototype_loss(
                            selected_context, context_labels, selected_target, target_labels,
                            distance=args['test.distance'])
                        val_log['scaler_df'] = pd.concat([
                            val_log['scaler_df'], pd.DataFrame.from_records([
                                {'Tag': f'task_loss/{valset}', 'Idx': 0, 'Value': stats_dict['loss']},
                                {'Tag': f'task_acc/{valset}', 'Idx': 0, 'Value': stats_dict['acc']}])])

                '''task/loss'''
                avg_log = {'loss': [], 'acc': []}
                for v_indx, valset in enumerate(valsets):
                    avg_log['loss'].append(debugger.write_scaler(
                        val_log['scaler_df'], key=f'task_loss/{valset}',
                        i=i, writer=writer, prefix='val_'))
                    avg_log['acc'].append(debugger.write_scaler(
                        val_log['scaler_df'], key=f'task_acc/{valset}',
                        i=i, writer=writer, prefix='val_'))

                '''write summaries averaged over sources'''
                avg_val_source_loss = np.mean(avg_log['loss'])
                avg_val_source_acc = np.mean(avg_log['acc'])
                writer.add_scalar(f"val_task_loss/avg_val_source_loss", avg_val_source_loss, i + 1)
                writer.add_scalar(f"val_task_acc/avg_val_source_acc", avg_val_source_acc, i + 1)
                print(f"==>> val: avg_loss {avg_val_source_loss:.3f}, "
                      f"avg_accuracy {avg_val_source_acc:.3f}.")

                '''f'inner/loss/{task_idx}' and acc'''
                for task_idx in range(args['train.n_obj'] + args['train.n_mix']):
                    debugger.write_inner(val_log['scaler_df'], key=f'inner/loss/{task_idx}',
                                         i=i, writer=writer, prefix='val_')
                    debugger.write_inner(val_log['scaler_df'], key=f'inner/acc/{task_idx}',
                                         i=i, writer=writer, prefix='val_')

                avg_val_loss, avg_val_acc = avg_val_source_loss, avg_val_source_acc

                # saving checkpoints
                if avg_val_acc > best_val_acc and i > 0:    # do not consider init
                    best_val_loss, best_val_acc = avg_val_loss, avg_val_acc
                    is_best = True
                    print('====>> Best model so far!')
                else:
                    is_best = False
                extra_dict = {'epoch_log': epoch_log, 'val_log': val_log}
                checkpointer.save_checkpoint(
                        i, best_val_acc, best_val_loss,
                        is_best, optimizer=optimizer,
                        state_dict=pmo.get_state_dict(whole=False), extra=extra_dict)

                '''save epoch_val_loss and epoch_val_acc'''
                with open(os.path.join(args['out.dir'], 'summary', 'val_log.pickle'), 'wb') as f:
                    pickle.dump({'epoch': i + 1, 'val_log': val_log}, f)

                print(f"====>> {return_time()} Trained and evaluated at {i + 1}.\n")

    '''Close the writers'''
    writer.close()

    if start_iter < max_iter:
        print(f"""Done training with best_mean_val_loss: {best_val_loss:.3f}, 
        best_avg_val_acc: {best_val_acc:.3f}""")
    else:
        print(f"""Training not completed. Loaded checkpoint at {start_iter}, while max_iter was {max_iter}""")


if __name__ == '__main__':
    train()

    # '''nvidia-smi'''
    # print(os.system('nvidia-smi'))

    # run testing
    from test_extractor import main as test
    test(test_model='best')
    print("↑↑ best model")
    # test(test_model='last')
    # print("↑↑ last model")
