"""
This code allows you to evaluate performance of a single feature extractor + pa with NCC
on the test splits of all datasets (ilsvrc_2012, omniglot, aircraft, cu_birds, dtd, quickdraw, fungi, 
vgg_flower, traffic_sign, mscoco, mnist, cifar10, cifar100). 

To test the url model on the test splits of all datasets, run:
python test_extractor_pa.py --model.name=url --model.dir ./saved_results/url

To test the url model on the test splits of ilsrvc_2012, dtd, vgg_flower, quickdraw,
comment the line 'testsets = ALL_METADATASET_NAMES' and run:
python test_extractor_pa.py --model.name=url --model.dir ./saved_results/url -data.test ilsrvc_2012 dtd vgg_flower quickdraw
"""

import os
import torch
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from utils import check_dir

from models.losses import prototype_loss, knn_loss, lr_loss, scm_loss, svm_loss
from models.model_utils import CheckPointer
from models.model_helpers import get_model, get_model_moe
from models.pa import apply_selection, pa
from data.meta_dataset_reader import (MetaDatasetEpisodeReader, MetaDatasetBatchReader, TRAIN_METADATASET_NAMES,
                                      ALL_METADATASET_NAMES)
from config import args


def main(no_selection=False):
    TEST_SIZE = 600

    # Setting up datasets
    trainsets, valsets, testsets = args['data.train'], args['data.val'], args['data.test']
    testsets = ALL_METADATASET_NAMES # comment this line to test the model on args['data.test']
    if args['test.mode'] == 'mdl':
        # multi-domain learning setting, meta-train on 8 training sets
        trainsets = TRAIN_METADATASET_NAMES
    elif args['test.mode'] == 'sdl':
        # single-domain learning setting, meta-train on ImageNet
        trainsets = ['ilsvrc_2012']

    if args['model.name'] == 'pmo':
        # pmo model, fe load from url
        if no_selection:
            args['model.num_clusters'] = 1
            model = get_model_moe(None, args, base_network_name='url')  # resnet18_moe
        else:
            model = get_model_moe(None, args, base_network_name='url')  # resnet18_moe
    else:
        model = get_model(None, args)
    checkpointer = CheckPointer(args, model, optimizer=None)
    checkpointer.restore_model(ckpt='best', strict=False)       # ckpt='best'  'last'     1999 or mute to not restore
    model.eval()

    accs_names = ['NCC']
    var_accs = dict()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    with tf.compat.v1.Session(config=config) as session:
        # go over each test domain
        test_loader = MetaDatasetEpisodeReader('test', trainsets, trainsets, testsets, test_type=args['test.type'])
        for dataset in testsets:
            if dataset in trainsets:
                lr = 0.1
            else:
                lr = 1
            print(dataset)
            var_accs[dataset] = {name: [] for name in accs_names}

            for i in tqdm(range(TEST_SIZE)):
                with torch.no_grad():
                    sample = test_loader.get_test_task(session, dataset)
                    if args['model.name'] == 'pmo':
                        if no_selection:
                            context_features = model.embed(sample['context_images'], selection=torch.Tensor([[1]]).cuda())
                            target_features = model.embed(sample['target_images'], selection=torch.Tensor([[1]]).cuda())

                        else:
                            task_features = model.embed(torch.cat([sample['context_images'], sample['target_images']]))
                            [context_features, target_features], selection_info = model(
                                [sample['context_images'], sample['target_images']], task_features,
                                gumbel=False, hard=False)
                    else:
                        context_features = model.embed(sample['context_images'])
                        target_features = model.embed(sample['target_images'])
                    context_labels = sample['context_labels']
                    target_labels = sample['target_labels']

                # optimize selection parameters and perform feature selection
                selection_params = pa(context_features, context_labels, max_iter=40, lr=lr, distance=args['test.distance'])
                selected_context = apply_selection(context_features, selection_params)
                selected_target = apply_selection(target_features, selection_params)
                _, stats_dict, _ = prototype_loss(
                    selected_context, context_labels,
                    selected_target, target_labels, distance=args['test.distance'])

                var_accs[dataset]['NCC'].append(stats_dict['acc'])
            dataset_acc = np.array(var_accs[dataset]['NCC']) * 100
            print(f"{dataset}: test_acc {dataset_acc.mean():.2f}%")
    # Print nice results table
    print('results of {}'.format(args['model.name']))
    rows = []
    id_accs = {n: [] for n in accs_names}
    ood_accs = {n: [] for n in accs_names}
    for dataset_idx, dataset_name in enumerate(testsets):
        row = [dataset_name]
        for model_name in accs_names:
            acc = np.array(var_accs[dataset_name][model_name]) * 100
            mean_acc = acc.mean()
            conf = (1.96 * acc.std()) / np.sqrt(len(acc))
            row.append(f"{mean_acc:0.2f} +- {conf:0.2f}")
            if dataset_idx < 8:
                id_accs[model_name].append(mean_acc)
            else:
                ood_accs[model_name].append(mean_acc)
        rows.append(row)
        if dataset_idx == 7:
            row = ['ID']
            for model_name in accs_names:
                acc = np.array(id_accs[model_name])
                mean_acc = acc.mean()
                conf = (1.96 * acc.std()) / np.sqrt(len(acc))
                row.append(f"{mean_acc:0.2f} +- {conf:0.2f}")
            rows.append(row)
        elif dataset_idx == 12:
            row = ['OOD']
            for model_name in accs_names:
                acc = np.array(ood_accs[model_name])
                mean_acc = acc.mean()
                conf = (1.96 * acc.std()) / np.sqrt(len(acc))
                row.append(f"{mean_acc:0.2f} +- {conf:0.2f}")
            rows.append(row)
    out_path = os.path.join(args['out.dir'], 'weights')
    out_path = check_dir(out_path, True)
    out_path = os.path.join(out_path, '{}-{}-{}-{}-test-results.npy'.format(args['model.name'], args['test.type'], 'pa', args['test.distance']))
    np.save(out_path, {'rows': rows})

    table = tabulate(rows, headers=['model \\ data'] + accs_names, floatfmt=".2f")
    print(table)
    print("\n")


if __name__ == '__main__':
    main()



