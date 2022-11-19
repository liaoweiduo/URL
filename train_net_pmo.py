"""
This code allows you to train multi learned domain learning networks with pool mo technique.

Author: Weiduo Liao
Date: 2022.11.12
"""

import os
import sys
import pickle
import copy
import torch
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
from utils import Accumulator
from config import args


def train():
    config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.allow_growth = False
    with tf.compat.v1.Session(config=config) as session:

        # initialize datasets and loaders
        trainsets = TRAIN_METADATASET_NAMES
        valsets = TRAIN_METADATASET_NAMES
        testsets = TRAIN_METADATASET_NAMES
        trainsets, valsets, testsets = args['data.train'], args['data.val'], args['data.test']
        print(f'Train on: {trainsets}.')    # "ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower"
        print(f'Val on: {valsets}.')
        # print(f'Test on: {testsets}.')

        train_loaders = []
        num_train_classes = dict()
        for t_indx, trainset in enumerate(trainsets):
            train_loaders.append(MetaDatasetEpisodeReader('train', trainset, valsets, testsets,
                                                          test_type='5shot'))
            num_train_classes[trainset] = train_loaders[t_indx].num_classes('train')

        val_loader = MetaDatasetEpisodeReader('val', trainsets, valsets, testsets)

        # initialize model and optimizer
        models = []
        optimizers = []
        checkpointers = []
        start_iters, best_val_losses, best_val_accs = [], [], []
        lr_managers = []
        writers = []
        # init all starting issues for M(8) models.
        for m_indx in range(args['model.num_clusters']):        # 8
            model_args_with_name = copy.deepcopy(args)
            model_args_with_name['model.name'] = model_args_with_name['model.name'].format(m_indx)    # M0-net - M7-net
            _model = get_model(None, model_args_with_name, multi_device_id=m_indx)     # distribute model to multi-devices
            models.append(_model)
            _optimizer = get_optimizer(_model, model_args_with_name, params=_model.get_parameters())
            optimizers.append(_optimizer)

            # restoring the last checkpoint
            _checkpointer = CheckPointer(model_args_with_name, _model, optimizer=_optimizer)
            checkpointers.append(_checkpointer)

            if os.path.isfile(_checkpointer.last_ckpt) and args['train.resume']:
                _start_iter, _best_val_loss, _best_val_acc =\
                    _checkpointer.restore_model(ckpt='last')
            else:
                print('No checkpoint restoration')
                _best_val_loss = 999999999
                _best_val_acc = _start_iter = 0
            start_iters.append(_start_iter)
            best_val_losses.append(_best_val_loss)
            best_val_accs.append(_best_val_acc)

            # define learning rate policy
            if args['train.lr_policy'] == "step":
                _lr_manager = UniformStepLR(_optimizer, args, _start_iter)
            elif "exp_decay" in args['train.lr_policy']:
                _lr_manager = ExpDecayLR(_optimizer, args, _start_iter)
            elif "cosine" in args['train.lr_policy']:
                _lr_manager = CosineAnnealRestartLR(_optimizer, args, _start_iter)
            lr_managers.append(_lr_manager)

            # defining the summary writer
            _writer = SummaryWriter(_checkpointer.out_path)
            writers.append(_writer)

        # Training loop
        max_iter = args['train.max_iter']
        epoch_loss = {name: [] for name in trainsets}
        epoch_acc = {name: [] for name in trainsets}
        epoch_val_loss = {name: [] for name in valsets}
        epoch_val_acc = {name: [] for name in valsets}

        start_iter = np.min(start_iters)
        for i in tqdm(range(max_iter)):
            if i < start_iter:
                continue

            for _optimizer in optimizers:
                _optimizer.zero_grad()
            '''Collect samples from multiple train loaders'''
            samples = []
            images = []
            num_samples = []
            for t_indx, train_loader in enumerate(train_loaders):
                sample = train_loader.get_train_batch(session)
                samples.append(sample)
                images.append(sample['images'])
                num_samples.append(sample['images'].size(0))


            sample = train_loader.get_train_batch(session)
            logits = model.forward(sample['images'])
            if len(logits.size()) < 2:
                logits = logits.unsqueeze(0)
            batch_loss, stats_dict, _ = cross_entropy_loss(logits, sample['labels'])
            batch_dataset = sample['dataset_name']
            epoch_loss[batch_dataset].append(stats_dict['loss'])
            epoch_acc[batch_dataset].append(stats_dict['acc'])

            batch_loss.backward()
            optimizer.step()
            lr_manager.step(i)

            if (i + 1) % 200 == 0:
                for dataset_name in trainsets:
                    writer.add_scalar(f"loss/{dataset_name}-train_acc",
                                      np.mean(epoch_loss[dataset_name]), i)
                    writer.add_scalar(f"accuracy/{dataset_name}-train_acc",
                                      np.mean(epoch_acc[dataset_name]), i)
                    epoch_loss[dataset_name], epoch_acc[dataset_name] = [], []

                writer.add_scalar('learning_rate',
                                  optimizer.param_groups[0]['lr'], i)

            # Evaluation inside the training loop
            if (i + 1) % args['train.eval_freq'] == 0:
                model.eval()
                dataset_accs, dataset_losses = [], []
                for valset in valsets:
                    val_losses, val_accs = [], []
                    for j in tqdm(range(args['train.eval_size'])):
                        with torch.no_grad():
                            sample = val_loader.get_validation_task(session, valset)
                            context_features = model.embed(sample['context_images'])
                            target_features = model.embed(sample['target_images'])
                            context_labels = sample['context_labels']
                            target_labels = sample['target_labels']
                            _, stats_dict, _ = prototype_loss(context_features, context_labels,
                                                              target_features, target_labels)
                        val_losses.append(stats_dict['loss'])
                        val_accs.append(stats_dict['acc'])

                    # write summaries per validation set
                    dataset_acc, dataset_loss = np.mean(val_accs) * 100, np.mean(val_losses)
                    dataset_accs.append(dataset_acc)
                    dataset_losses.append(dataset_loss)
                    epoch_val_loss[valset].append(dataset_loss)
                    epoch_val_acc[valset].append(dataset_acc)
                    writer.add_scalar(f"loss/{valset}/val_loss", dataset_loss, i)
                    writer.add_scalar(f"accuracy/{valset}/val_acc", dataset_acc, i)
                    print(f"{valset}: val_acc {dataset_acc:.2f}%, val_loss {dataset_loss:.3f}")

                # write summaries averaged over datasets
                avg_val_loss, avg_val_acc = np.mean(dataset_losses), np.mean(dataset_accs)
                writer.add_scalar(f"loss/avg_val_loss", avg_val_loss, i)
                writer.add_scalar(f"accuracy/avg_val_acc", avg_val_acc, i)

                # saving checkpoints
                if avg_val_acc > best_val_acc:
                    best_val_loss, best_val_acc = avg_val_loss, avg_val_acc
                    is_best = True
                    print('Best model so far!')
                else:
                    is_best = False
                extra_dict = {'epoch_loss': epoch_loss, 'epoch_acc': epoch_acc, 'epoch_val_loss': epoch_val_loss, 'epoch_val_acc': epoch_val_acc}
                checkpointer.save_checkpoint(i, best_val_acc, best_val_loss,
                                             is_best, optimizer=optimizer,
                                             state_dict=model.get_state_dict(), extra=extra_dict)

                model.train()
                print(f"Trained and evaluated at {i}")

    writer.close()
    if start_iter < max_iter:
        print(f"""Done training with best_mean_val_loss: {best_val_loss:.3f}, best_avg_val_acc: {best_val_acc:.2f}%""")
    else:
        print(f"""No training happened. Loaded checkpoint at {start_iter}, while max_iter was {max_iter}""")


if __name__ == '__main__':
    train()
