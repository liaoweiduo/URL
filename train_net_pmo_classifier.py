"""
This code allows you to train specific classifiers for the given fixed feature extractors and class mapping.

Author: Weiduo Liao
Date: 2022.12.7
"""

import os
import sys
import json
import torch
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data.meta_dataset_reader import (MetaDatasetBatchReader,
                                      MetaDatasetEpisodeReader)
from models.losses import cross_entropy_loss, prototype_loss
from models.model_utils import (CheckPointer, UniformStepLR,
                                CosineAnnealRestartLR, ExpDecayLR)
from models.model_helpers import get_model, get_optimizer
from utils import Accumulator, set_determ, device
from config import args, BATCHSIZES

from pmo_utils import Pool


def train():
    # Set seed
    set_determ(seed=1234)

    config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.allow_growth = False
    with tf.compat.v1.Session(config=config) as session:
        # initialize datasets and loaders
        trainsets, valsets, testsets = args['data.train'], args['data.val'], args['data.test']
        print(f'Train on: {trainsets}.')    # "ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower"
        print(f'Val on: {valsets}.')
        # print(f'Test on: {testsets}.')

        train_loaders = dict()
        for t_indx, trainset in enumerate(trainsets):
            train_loaders[trainset] = MetaDatasetBatchReader(
                'train', [trainset], valsets, testsets, batch_size=BATCHSIZES[trainset])

        val_loader = MetaDatasetEpisodeReader('val', trainsets, valsets, testsets)

        # initialize model and optimizer
        '''obtain class mapping and num_classes'''
        class_mapping_dict = json.load(
            open(os.path.join(args['model.dir'], 'weights', 'pool', 'class_mapping_train.json'), 'r'))
        classes_in_cluster = class_mapping_dict['classes_in_cluster']
        # {C0-C7: [[gt_label, label_str, domain_str, re_label],...]}
        class_mapping = class_mapping_dict['class_mapping']
        num_train_classes = {cluster_name: len(items) for cluster_name, items in classes_in_cluster.items()}
        print(f'num_train_classes: {num_train_classes}')
        # num_train_classes: e.g., {'C0': 3093, 'C1': 0, 'C2': 70, 'C3': 23, 'C4': 23, 'C5': 2323, 'C6': 23, 'C7':23}
        val_class_mapping = json.load(
            open(os.path.join(args['model.dir'], 'weights', 'pool', 'class_mapping_val.json'), 'r'))['class_mapping']

        pool = Pool(capacity=args['model.num_clusters'])
        pool.restore(0)      # restore pool cluster centers, iter is not used.

        model_names = [args['model.name'].format(m_indx) for m_indx in range(args['model.num_clusters'])]
        cluster_model_name = 'C-net'
        cluster_names = [f'C{idx}' for idx in range(args['model.num_clusters'])]

        model_idx = model_names.index(args['model.name'])
        model_name = model_names[model_idx]
        cluster_name = cluster_names[model_idx]
        print(f'model_name: {model_name}, cluster_name: {cluster_name}')

        model = get_model(num_train_classes[cluster_name], args)
        optimizer = get_optimizer(model, args, params=model.get_parameters())

        # restoring the last checkpoint
        checkpointer = CheckPointer(args, model, optimizer=optimizer)
        if os.path.isfile(checkpointer.last_ckpt) and args['train.resume']:
            start_iter, best_val_loss, best_val_acc =\
                checkpointer.restore_model(ckpt='last')
        else:
            print('No checkpoint restoration')
            best_val_loss = 999999999
            best_val_acc = start_iter = 0

        # define learning rate policy
        if args['train.lr_policy'] == "step":
            lr_manager = UniformStepLR(optimizer, args, start_iter)
        elif "exp_decay" in args['train.lr_policy']:
            lr_manager = ExpDecayLR(optimizer, args, start_iter)
        elif "cosine" in args['train.lr_policy']:
            lr_manager = CosineAnnealRestartLR(optimizer, args, start_iter)

        # defining the summary writer
        writer = SummaryWriter(checkpointer.out_path)

        # Training loop
        max_iter = args['train.max_iter']
        epoch_loss = {cluster_name: []}
        epoch_acc = {cluster_name: []}
        epoch_val_loss = {cluster_name: []}
        epoch_val_acc = {cluster_name: []}

        for i in tqdm(range(max_iter), ncols=100):
            if i < start_iter:
                continue

            optimizer.zero_grad()

            '''obtain samples from loaders only when buffer has not enough samples'''
            while len(pool.buffer) < args['train.batch_size']:
                '''obtain tasks from train_loaders'''
                for t_indx, trainset in enumerate(trainsets):
                    num_task_per_batch = 1
                    if trainset == 'ilsvrc_2012':
                        num_task_per_batch = 2

                    for _ in range(num_task_per_batch):
                        sample = train_loaders[trainset].get_train_batch(session)
                        pool.batch_put_into_buffer(sample, class_mapping, trainset, cluster_name,
                                                   train_loaders[trainset])

            sample = pool.batch_sample_from_buffer(args['train.batch_size'])
            logits = model.forward(sample['images'])
            if len(logits.size()) < 2:
                logits = logits.unsqueeze(0)
            batch_loss, stats_dict, _ = cross_entropy_loss(logits, sample['labels'])
            epoch_loss[cluster_name].append(stats_dict['loss'])
            epoch_acc[cluster_name].append(stats_dict['acc'])

            batch_loss.backward()
            optimizer.step()
            lr_manager.step(i)

            if (i + 1) % 200 == 0:
                writer.add_scalar(f"loss/{cluster_name}-train_acc",
                                  np.mean(epoch_loss[cluster_name]), i)
                writer.add_scalar(f"accuracy/{cluster_name}-train_acc",
                                  np.mean(epoch_acc[cluster_name]), i)
                epoch_loss[cluster_name], epoch_acc[cluster_name] = [], []

                writer.add_scalar('learning_rate',
                                  optimizer.param_groups[0]['lr'], i)

            # Evaluation inside the training loop
            if (i + 1) % args['train.eval_freq'] == 0:
                model.eval()

                val_pool = Pool(capacity=args['model.num_clusters'])
                val_pool.centers = pool.centers     # same centers and device as train_pool
                val_pool.eval()

                cluster_accs, cluster_losses = [[] for _ in range(args['model.num_clusters'])], \
                                               [[] for _ in range(args['model.num_clusters'])]
                for v_indx, valset in enumerate(valsets):
                    print(f"==>> collect classes from {valset}.")
                    for j in tqdm(range(args['train.eval_size']), ncols=100):
                        with torch.no_grad():
                            '''obtain 1 task from val_loader'''
                            sample_numpy = val_loader.get_validation_task(session, valset, d='numpy')
                            images_numpy = np.concatenate(
                                [sample_numpy['context_images'], sample_numpy['target_images']])
                            gt_labels_numpy = np.concatenate([sample_numpy['context_gt'], sample_numpy['target_gt']])
                            domain = v_indx

                            val_pool.cluster_and_assign_with_class_mapping(
                                images_numpy, gt_labels_numpy, domain, valset,
                                val_class_mapping, cluster_name, model_idx)

                            '''check if any cluster have sufficient class to construct 1 task'''
                            for idx, classes in enumerate(val_pool.current_classes()):
                                num_imgs = np.array([cls[1] for cls in classes])
                                available_way = len(num_imgs[num_imgs >= args['train.n_shot'] + args['train.n_query']])
                                if available_way >= args['train.n_way']:
                                    # enough classes to construct 1 task
                                    # then use all available classes to construct 1 task
                                    numpy_task, grad_ones = val_pool.episodic_sample(
                                        idx, n_way=available_way, remove_sampled_classes=True)

                                    '''put to device and forward model to obtain val_acc/loss'''
                                    d = device
                                    torch_task = val_pool.to_torch(numpy_task, grad_ones, device_list=[d])

                                    context_features = model.embed(torch_task[d]['context_images'])
                                    target_features = model.embed(torch_task[d]['target_images'])
                                    context_labels = torch_task[d]['context_labels']
                                    target_labels = torch_task[d]['target_labels']
                                    _, stats_dict, _ = prototype_loss(context_features, context_labels,
                                                                      target_features, target_labels)

                                    cluster_losses[idx].append(stats_dict['loss'])
                                    cluster_accs[idx].append(stats_dict['acc'])

                '''write and print'''
                for cluster_idx, (loss_list, acc_list) in enumerate(zip(cluster_losses, cluster_accs)):
                    if len(loss_list) > 0:
                        cluster_acc, cluster_loss = np.mean(acc_list).item() * 100, np.mean(loss_list).item()

                        epoch_val_loss[model_names[cluster_idx]].append(cluster_loss)
                        epoch_val_acc[model_names[cluster_idx]].append(cluster_acc)
                        writer.add_scalar(f"loss/val/{model_names[cluster_idx]}", cluster_loss, i+1)
                        writer.add_scalar(f"accuracy/val/{model_names[cluster_idx]}", cluster_acc, i+1)
                        print(f"==>> {model_names[cluster_idx]}: "
                              f"val_acc {cluster_acc:.2f}%, val_loss {cluster_loss:.3f}")

                    else:   # no class(task) is assign to this cluster
                        print(f"==>> {model_names[cluster_idx]}: "
                              f"val_acc No value, val_loss No value")

                # write summaries averaged over clusters
                avg_val_loss, avg_val_acc = np.mean(np.concatenate(cluster_losses)), np.mean(np.concatenate(cluster_accs))
                writer.add_scalar(f"loss/val/avg_val_loss", avg_val_loss, i+1)
                writer.add_scalar(f"accuracy/val/avg_val_acc", avg_val_acc, i+1)

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
