"""
This code allows you to train a domain classifier.

Author: Weiduo Liao
Date: 2023.07.31
"""

import os
import sys
import pickle
import copy
import torch
import torch.nn.functional as F
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data.meta_dataset_reader import (MetaDatasetBatchReader, MetaDatasetEpisodeReader,
                                      TRAIN_METADATASET_NAMES)
from models.losses import cross_entropy_loss, prototype_loss
from models.model_utils import (CheckPointer, UniformStepLR,
                                CosineAnnealRestartLR, ExpDecayLR)
from models.model_helpers import get_model, get_optimizer
from utils import Accumulator, device, set_determ, check_dir
from config import args

from pmo_utils import (Pool, Mixer,
                       cal_hv_loss, cal_hv, draw_objs, draw_heatmap, available_setting, check_available, task_to_device)

import warnings
warnings.filterwarnings('ignore')


def train():
    # Set seed
    set_determ(seed=1234)

    config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.allow_growth = False
    with tf.compat.v1.Session(config=config) as session:
        '''--------------------'''
        '''Initialization Phase'''
        '''--------------------'''
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

        # train_loader = MetaDatasetEpisodeReader('train', trainsets, valsets, testsets, test_type='5shot')
        # num_train_classes = train_loader.num_classes('train')
        # print(f'num_train_classes: {num_train_classes}')

        val_loader = MetaDatasetEpisodeReader('val', trainsets, valsets, testsets, test_type=args['test.type'])

        '''initialize models and optimizer'''
        start_iter, best_val_loss, best_val_acc = 0, 999999999, 0

        # pmo model load from url
        pmo = get_model(len(trainsets), args, base_network_name='url')    # resnet18_moe

        optimizer = get_optimizer(pmo, args, params=pmo.get_trainable_classifier_parameters())
        checkpointer = CheckPointer(args, pmo, optimizer=optimizer, save_all=True)
        if os.path.isfile(checkpointer.last_ckpt) and args['train.resume']:
            start_iter, best_val_loss, best_val_acc = \
                checkpointer.restore_model(ckpt='last')
        else:
            print('No checkpoint restoration for pmo.')
        if args['train.lr_policy'] == "step":
            lr_manager = UniformStepLR(optimizer, args, start_iter)
        elif "exp_decay" in args['train.lr_policy']:
            lr_manager = ExpDecayLR(optimizer, args, start_iter)
        elif "cosine" in args['train.lr_policy']:
            lr_manager = CosineAnnealRestartLR(optimizer, args, start_iter)

        # defining the summary writer
        writer = SummaryWriter(check_dir(os.path.join(args['out.dir'], 'summary'), False))

        '''-------------'''
        '''Training loop'''
        '''-------------'''
        max_iter = args['train.max_iter']

        def init_train_log():
            epoch_loss = {'loss': []}
            epoch_acc = {'acc': []}
            return epoch_loss, epoch_acc

        def model_train():
            # train mode
            pmo.train()

        def model_eval():
            # eval mode
            pmo.eval()

        def zero_grad():
            optimizer.zero_grad()

        def update_step(idx):
            optimizer.step()

            lr_manager.step(idx)

        epoch_loss, epoch_acc = init_train_log()
        epoch_val_loss = {}
        epoch_val_acc = {}

        print(f'\n>>>> Train start from {start_iter}.')
        for i in tqdm(range(start_iter, max_iter), ncols=100):

            zero_grad()
            model_train()

            '''obtain tasks from train_loaders'''
            samples = []
            images = dict()
            labels = dict()
            # loading images and labels
            for t_indx, (name, train_loader) in enumerate(train_loaders.items()):
                sample = train_loader.get_train_task(session, d=device)
                samples.append(sample)
                images[name] = torch.cat([sample['context_images'], sample['target_images']])
                labels[name] = torch.ones_like(
                    torch.cat([sample['context_labels'], sample['target_labels']])
                ).long() * t_indx

            '''forward'''
            features = pmo.embed(torch.cat(list(images.values())))
            logits = pmo.cls_fn(features)
            batch_loss, stats_dict, _ = cross_entropy_loss(logits, torch.cat(list(labels.values())))
            batch_loss.backward()

            '''log'''
            epoch_loss[f'loss'].append(stats_dict['loss'])
            epoch_acc[f'acc'].append(stats_dict['acc'])

            update_step(i)

            '''log iter-wise params change'''
            writer.add_scalar('params/learning_rate', optimizer.param_groups[0]['lr'], i+1)

            if (i + 1) % args['train.summary_freq'] == 0:        # 5; 2 for DEBUG
                print(f">> Iter: {i + 1}, train summary:")
                '''save epoch_loss and epoch_acc'''
                epoch_train_history = dict()
                if os.path.exists(os.path.join(args['out.dir'], 'summary', 'train_log.pickle')):
                    epoch_train_history = pickle.load(
                        open(os.path.join(args['out.dir'], 'summary', 'train_log.pickle'), 'rb'))
                epoch_train_history[i + 1] = {'loss': epoch_loss.copy(), 'acc': epoch_acc.copy()}
                with open(os.path.join(args['out.dir'], 'summary', 'train_log.pickle'), 'wb') as f:
                    pickle.dump(epoch_train_history, f)

                '''log task loss and accuracy'''
                average_loss = np.mean(epoch_loss['loss'])
                average_accuracy = np.mean(epoch_acc['acc'])
                writer.add_scalar(f"loss", average_loss, i+1)
                writer.add_scalar(f"accuracy", average_accuracy, i+1)
                print(f"==>> loss {average_loss:.3f}, "
                      f"accuracy {average_accuracy:.3f}.")

                epoch_loss, epoch_acc = init_train_log()

            '''----------'''
            '''Eval Phase'''
            '''----------'''
            # Evaluation inside the training loop
            if (i + 1) % args['train.eval_freq'] == 0:      # args['train.eval_freq']; 10 for DEBUG
                print(f"\n>> Iter: {i + 1}, evaluation:")
                # eval mode
                model_eval()

                '''collect val_losses/accs for all sources and cluster_losses/accs for all FiLMs'''
                val_accs, val_losses = {f'{name}': [] for name in valsets}, {f'{name}': [] for name in valsets}
                with torch.no_grad():
                    for v_indx, valset in enumerate(valsets):
                        print(f"==>> evaluate on {valset}.")
                        for j in tqdm(range(args['train.eval_size']), ncols=100):
                            '''obtain 1 task from val_loader'''
                            samples = val_loader.get_validation_task(session, valset, d=device)
                            context_images, target_images = samples['context_images'], samples['target_images']
                            context_labels, target_labels = samples['context_labels'], samples['target_labels']
                            # context_gt_labels, target_gt_labels = samples['context_gt'], samples['target_gt']
                            domain = v_indx
                            labels = torch.ones_like(
                                torch.cat([context_labels, target_labels])
                            ).long() * domain

                            '''forward'''
                            features = pmo.embed(torch.cat([context_images, target_images]))
                            logits = pmo.cls_fn(features)
                            _, stats_dict, _ = cross_entropy_loss(logits, labels)

                            val_losses[valset].append(stats_dict['loss'])
                            val_accs[valset].append(stats_dict['acc'])

                        '''write and print val on source'''
                        epoch_val_loss[valset] = np.mean(val_losses[valset])
                        epoch_val_acc[valset] = np.mean(val_accs[valset])
                        writer.add_scalar(f"val_loss/{valset}", epoch_val_loss[valset], i+1)
                        writer.add_scalar(f"val_accuracy/{valset}", epoch_val_acc[valset], i+1)
                        print(f"==>> val: loss {np.mean(val_losses[valset]):.3f}, "
                              f"accuracy {np.mean(val_accs[valset]):.3f}.")

                '''write summaries averaged over sources'''
                avg_val_source_loss = np.mean(np.concatenate([val_loss for val_loss in val_losses.values()]))
                avg_val_source_acc = np.mean(np.concatenate([val_acc for val_acc in val_accs.values()]))
                writer.add_scalar(f"val_loss/avg_val_source_loss", avg_val_source_loss, i+1)
                writer.add_scalar(f"val_accuracy/avg_val_source_acc", avg_val_source_acc, i+1)
                print(f"==>> val: avg_loss {avg_val_source_loss:.3f}, "
                      f"avg_accuracy {avg_val_source_acc:.3f}.")

                # evaluation acc based on cluster acc
                avg_val_loss, avg_val_acc = avg_val_source_loss, avg_val_source_acc

                # saving checkpoints
                if avg_val_acc > best_val_acc:
                    best_val_loss, best_val_acc = avg_val_loss, avg_val_acc
                    is_best = True
                    print('====>> Best model so far!')
                else:
                    is_best = False
                extra_dict = {'epoch_loss': epoch_loss, 'epoch_acc': epoch_acc,
                              'epoch_val_loss': epoch_val_loss, 'epoch_val_acc': epoch_val_acc}
                checkpointer.save_checkpoint(
                        i, best_val_acc, best_val_loss,
                        is_best, optimizer=optimizer,
                        state_dict=pmo.get_state_dict(), extra=extra_dict)

                '''save epoch_val_loss and epoch_val_acc'''
                with open(os.path.join(args['out.dir'], 'summary', 'val_log.pickle'), 'wb') as f:
                    pickle.dump({'epoch': i + 1, 'loss': epoch_val_loss, 'acc': epoch_val_acc}, f)

                print(f"====>> Trained and evaluated at {i + 1}.\n")

    '''Close the writers'''
    writer.close()

    if start_iter < max_iter:
        print(f"""Done training with best_mean_val_loss: {best_val_loss:.3f}, 
        best_avg_val_acc: {best_val_acc:.3f}""")
    else:
        print(f"""Training not completed. Loaded checkpoint at {start_iter}, while max_iter was {max_iter}""")


if __name__ == '__main__':
    train()
