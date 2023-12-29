"""
This code allows you to evaluate performance of a single feature extractor + a classifier
on the test splits of all datasets (ilsvrc_2012, omniglot, aircraft, cu_birds, dtd, quickdraw, fungi, 
vgg_flower, traffic_sign, mscoco, mnist, cifar10, cifar100). 

The default classifier used in this code is the NCC with cosine similarity. 
One can use other classifiers for meta-testing, 
e.g. use ```--test.loss-opt``` to select nearest centroid classifier (ncc, default), 
support vector machine (svm), logistic regression (lr), Mahalanobis distance from 
Simple CNAPS (scm), or k-nearest neighbor (knn); 
use ```--test.feature-norm``` to normalize feature (l2) or not for svm and lr; 
use ```--test.distance``` to specify the feature similarity function (l2 or cos) for NCC. 

To evaluate the feature extractor with NCC and cosine similarity on test splits of all datasets, run:
python test_extractor.py --test.loss-opt ncc --test.feature-norm none --test.distance cos --model.name=<model name> --model.dir <directory of url> 

To test the feature extractor one the test splits of ilsrvc_2012, dtd, vgg_flower, quickdraw,
comment the line 'testsets = ALL_METADATASET_NAMES' and run:
python test_extractor.py --test.loss-opt ncc --test.feature-norm none --test.distance cos --data.test ilsrvc_2012 dtd vgg_flower quickdraw --model.name=<model name> --model.dir <directory of url> 
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
from models.pa import apply_selection
from data.meta_dataset_reader import (MetaDatasetEpisodeReader, MetaDatasetBatchReader, TRAIN_METADATASET_NAMES,
                                      ALL_METADATASET_NAMES)
from config import args


def main(test_model='best'):
    TEST_SIZE = 600

    # Setting up datasets
    trainsets, valsets, testsets = args['data.train'], args['data.val'], args['data.test']
    testsets = ALL_METADATASET_NAMES # comment this line to test the model on args['data.test']
    trainsets = TRAIN_METADATASET_NAMES

    if args['model.name'] == 'pmo':
        # pmo model, fe load from url
        model = get_model_moe(None, args, base_network_name='url')  # resnet18_moe
    else:
        model = get_model(None, args, base_network_name='url')
    checkpointer = CheckPointer(args, model, optimizer=None)
    if test_model is not None:
        checkpointer.restore_out_model(ckpt=test_model, strict=False)
        # ckpt='best'  'last'     1999 or mute to not restore
    model.eval()
    accs_names = [args['test.loss_opt']]
    var_accs = dict()

    config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.allow_growth = False
    with tf.compat.v1.Session(config=config) as session:
        # go over each test domain
        test_loader = MetaDatasetEpisodeReader('test', trainsets, trainsets, testsets, test_type=args['test.type'])

        for dataset in testsets:
            print(dataset)
            var_accs[dataset] = {name: [] for name in accs_names}

            for i in tqdm(range(TEST_SIZE)):
                with torch.no_grad():
                    sample = test_loader.get_test_task(session, dataset)
                    if args['model.name'] == 'pmo' and 'film' in args['train.cond_mode']:
                        task_features = model.embed(torch.cat([sample['context_images'], sample['target_images']]))
                        [context_features, target_features], selection_info = model(
                            [sample['context_images'], sample['target_images']], task_features,
                            gumbel=False, hard=args['train.sim_gumbel'])

                    elif args['model.name'] == 'pmo' and 'pa' in args['train.cond_mode']:
                        context_features = model._embed(sample['context_images'])
                        target_features = model._embed(sample['target_images'])
                        task_features = model.embed(
                            torch.cat([sample['context_images'], sample['target_images']]))
                        selection, selection_info = model.selector(
                            task_features, gumbel=False, hard=False)
                        selection_params = [torch.mm(selection, model.pas.flatten(1)).view(512, 512, 1, 1)]
                        context_features = apply_selection(context_features, selection_params)
                        target_features = apply_selection(target_features, selection_params)

                    else:
                        context_features = model.embed(sample['context_images'])
                        target_features = model.embed(sample['target_images'])

                    context_labels = sample['context_labels']
                    target_labels = sample['target_labels']
                    if args['test.loss_opt'] == 'ncc':
                        _, stats_dict, _ = prototype_loss(
                            context_features, context_labels,
                            target_features, target_labels, distance=args['test.distance'])
                    elif args['test.loss_opt'] == 'knn':
                        _, stats_dict, _ = knn_loss(
                            context_features, context_labels,
                            target_features, target_labels)
                    elif args['test.loss_opt'] == 'lr':
                        _, stats_dict, _ = lr_loss(
                            context_features, context_labels,
                            target_features, target_labels, normalize=(args['test.feature_norm'] == 'l2'))
                    elif args['test.loss_opt'] == 'svm':
                        _, stats_dict, _ = svm_loss(
                            context_features, context_labels,
                            target_features, target_labels, normalize=(args['test.feature_norm'] == 'l2'))
                    elif args['test.loss_opt'] == 'scm':
                        _, stats_dict, _ = scm_loss(
                            context_features, context_labels,
                            target_features, target_labels, normalize=False)
                var_accs[dataset][args['test.loss_opt']].append(stats_dict['acc'])
            dataset_acc = np.array(var_accs[dataset][args['test.loss_opt']]) * 100
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
                row.append(f"{mean_acc:0.2f}")
            rows.append(row)
        elif dataset_idx == 12:
            row = ['OOD']
            for model_name in accs_names:
                acc = np.array(ood_accs[model_name])
                mean_acc = acc.mean()
                row.append(f"{mean_acc:0.2f}")
            rows.append(row)
    out_path = os.path.join(args['out.dir'], 'weights')
    out_path = check_dir(out_path, True)
    out_path = os.path.join(out_path, '{}-{}-{}-{}-{}-test-results.npy'.format(args['model.name'], args['test.type'], args['test.loss_opt'], args['test.feature_norm'], args['test.distance']))
    np.save(out_path, {'rows': rows})

    table = tabulate(rows, headers=['model \\ data'] + accs_names, floatfmt=".2f")
    print(table)
    print("\n")


if __name__ == '__main__':
    main()



