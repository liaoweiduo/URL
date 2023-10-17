import os
import gin
import torch
from functools import partial

from models.model_utils import CheckPointer
from models.models_dict import DATASET_MODELS_RESNET18, DATASET_MODELS_RESNET18_PNF
from utils import device


def get_model_moe(num_classes, args, base_network_name=None, d=None, freeze_fe=False):
    train_classifier = args['model.classifier']
    dropout_rate = args.get('model.dropout', 0)
    if base_network_name is None:       # for fe before selector
        base_network_name = DATASET_MODELS_RESNET18['ilsvrc_2012']
    # moe_base_network_name = DATASET_MODELS_RESNET18['ilsvrc_2012']        # need to change source_moe to sdl's path
    moe_base_network_name = base_network_name       # use URL for moe backbone

    from models.resnet18_moe import resnet18 as resnet18_moe
    if args['model.pretrained']:
        '''load feature extractor'''
        base_network_path = os.path.join(args['source_moe'], 'weights', moe_base_network_name, 'model_best.pth.tar')
        model_fn = partial(resnet18_moe, dropout=dropout_rate,
                           pretrained_model_path=base_network_path,
                           film_head=args['model.num_clusters'],
                           tau=args['train.gumbel_tau'],
                           logit_scale=args['cluster.logit_scale'],
                           num_clusters=args['model.num_clusters'],
                           opt=args['cluster.opt'],
                           cond_mode=args['train.cond_mode'],
                           freeze_backbone=args['train.freeze_backbone'])
    else:
        model_fn = partial(resnet18_moe, dropout=dropout_rate,
                           film_head=args['model.num_clusters'],
                           tau=args['train.gumbel_tau'],
                           logit_scale=args['cluster.logit_scale'],
                           num_clusters=args['model.num_clusters'],
                           opt=args['cluster.opt'],
                           cond_mode=args['train.cond_mode'],
                           freeze_backbone=False)

    model = model_fn(classifier=train_classifier,
                     num_classes=num_classes,
                     global_pool=False)

    from models.resnet18 import resnet18 as resnet18_fe
    if args['model.pretrained']:
        base_network_path = os.path.join(args['source'], 'weights', base_network_name,
                                         'model_best.pth.tar')
        # base_network_path = os.path.join(args['source_moe'], 'weights', moe_base_network_name, 'model_best.pth.tar')

        model_fn = partial(resnet18_fe, dropout=dropout_rate,
                           pretrained_model_path=base_network_path)
    else:
        model_fn = partial(resnet18_fe, dropout=dropout_rate)

    model_fe = model_fn(classifier=train_classifier,
                        num_classes=num_classes,
                        global_pool=False)
    if args['model.pretrained']:
        # freeze
        if args['model.pretrained']:
            for k, v in model_fe.named_parameters():
                if 'cls' not in k and 'running' not in k:
                    v.requires_grad = False
        model_fe.eval()
    model.feature_extractor = model_fe

    if d is None:
        d = device
    model.to(d)
    print(f'Move moe model to {d}.')

    if freeze_fe:           # only classifier not freeze
        for name, param in model.named_parameters():
            if 'cls' not in name:       # cls_fn
                param.requires_grad = False
        model.eval()

    return model


def get_model(num_classes, args, base_network_name=None, d=None, freeze_fe=False):
    train_classifier = args['model.classifier']
    model_name = args['model.backbone']
    dropout_rate = args.get('model.dropout', 0)
    if base_network_name is None:
        base_network_name = DATASET_MODELS_RESNET18['ilsvrc_2012']

    if 'pnf' in model_name:
        from models.resnet18_pnf import resnet18
        base_network_path = os.path.join(args['source'], 'weights', base_network_name, 'model_best.pth.tar')
        model_fn = partial(resnet18, dropout=dropout_rate,
                           pretrained_model_path=base_network_path)
    elif num_classes is not None and isinstance(num_classes, list):
        from models.resnet18_mdl import resnet18
        if args['model.pretrained']:
            base_network_path = os.path.join(args['source'], 'weights', base_network_name,
                                 'model_best.pth.tar')
            model_fn = partial(resnet18, dropout=dropout_rate,
                           pretrained_model_path=base_network_path)
        else:
            model_fn = partial(resnet18, dropout=dropout_rate)
    else:
        from models.resnet18 import resnet18
        if args['model.pretrained']:
            base_network_path = os.path.join(args['source'], 'weights', base_network_name,
                                 'model_best.pth.tar')
            model_fn = partial(resnet18, dropout=dropout_rate,
                           pretrained_model_path=base_network_path)
        else:
            model_fn = partial(resnet18, dropout=dropout_rate)

    model = model_fn(classifier=train_classifier,
                     num_classes=num_classes,
                     global_pool=False)

    if d is not None:
        model.to(d)
        print(f'Move model {base_network_name} to {d}.')
    else:
        model.to(device)

    if freeze_fe:
        for name, param in model.named_parameters():
            if 'cls' not in name:       # cls_fn
                param.requires_grad = False

    return model


def get_optimizer(model, args, params=None):
    learning_rate = args['train.learning_rate']
    weight_decay = args['train.weight_decay']
    optimizer = args['train.optimizer']
    params = model.parameters() if params is None else params
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(params,
                                     lr=learning_rate,
                                     weight_decay=weight_decay)
    elif optimizer == 'momentum':
        optimizer = torch.optim.SGD(params,
                                    lr=learning_rate,
                                    momentum=0.9, nesterov=args['train.nesterov_momentum'],
                                    weight_decay=weight_decay)
    elif optimizer == 'ada':
        optimizer = torch.optim.Adadelta(params, lr=learning_rate)
    else:
        assert False, 'No such optimizer'
    return optimizer


def get_domain_extractors(trainset, dataset_models, args, num_classes=None):
    if 'pnf' in args['model.backbone']:
        return get_pnf_extractor(trainset, dataset_models, args)
    else:
        return get_multinet_extractor(trainset, dataset_models, args, num_classes)


def get_multinet_extractor(trainsets, dataset_models, args, num_classes=None):
    extractors = dict()
    for dataset_name in trainsets:
        if dataset_name not in dataset_models:
            continue
        args['model.name'] = dataset_models[dataset_name]
        if num_classes is None:
            extractor = get_model(None, args)
        else:
            extractor = get_model(num_classes[dataset_name], args)
        checkpointer = CheckPointer(args, extractor, optimizer=None)
        extractor.eval()
        checkpointer.restore_model(ckpt='best', strict=False)
        extractors[dataset_name] = extractor

    def embed_many(images, return_type='dict', kd=False, logits=False):
        with torch.no_grad():
            all_features = dict()
            all_logits = dict()
            for name, extractor in extractors.items():
                if logits:
                    if kd:
                        all_logits[name], all_features[name] = extractor(images[name], kd=True)
                    else:
                        all_logits[name] = extractor(images[name])
                else:
                    if kd:
                        all_features[name] = extractor.embed(images[name])
                    else:
                        all_features[name] = extractor.embed(images)

        if return_type == 'list':
            return list(all_features.values()), list(all_logits.values())
        else:
            return all_features, all_logits
    return embed_many


def get_pnf_extractor(trainsets, dataset_models, args):
    film_layers = dict()
    for dataset_name in trainsets:
        if dataset_name not in dataset_models or 'ilsvrc' in dataset_name:
            continue
        base_network_name = DATASET_MODELS_RESNET18_PNF[dataset_name]
        ckpt_path = os.path.join(args['source'], 'weights', base_network_name,
                                 'model_best.pth.tar')

        state_dict = torch.load(ckpt_path, map_location=device)['state_dict']
        film_layers[dataset_name] = {k: v for k, v in state_dict.items()
                                     if 'cls' not in k}
        print('Loaded FiLM layers from {}'.format(ckpt_path))

    # define the base extractor
    base_extractor = get_model(None, args)
    base_extractor.eval()
    base_layers = {k: v for k, v in base_extractor.get_state_dict().items() if 'cls' not in k}

    # initialize film layers of base extractor to identity
    film_layers['ilsvrc_2012'] = {k: v.clone() for k, v in base_layers.items()}

    def embed_many(images, return_type='dict'):
        with torch.no_grad():
            all_features = dict()

            for domain_name in trainsets:
                # setting up domain-specific film layers
                domain_layers = film_layers[domain_name]
                for layer_name in base_layers.keys():
                    base_layers[layer_name].data.copy_(domain_layers[layer_name].data)

                # inference
                all_features[domain_name] = base_extractor.embed(images)
        if return_type == 'list':
            return list(all_features.values())
        else:
            return all_features
    return embed_many
