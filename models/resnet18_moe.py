import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial

from models.model_utils import CosineClassifier
from models.adaptors import adaptor


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class CatFilm(nn.Module):
    """Film layer that performs per-channel affine transformation."""
    def __init__(self, planes):
        super(CatFilm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, planes))
        self.beta = nn.Parameter(torch.zeros(1, planes))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        return gamma * x + beta


def film(x, gamma, beta):
    """Film function."""
    gamma = gamma.view(*gamma.shape, 1, 1)
    beta = beta.view(*beta.shape, 1, 1)
    return gamma * x + beta


class BasicBlockFilm(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, film_head=1):
        super(BasicBlockFilm, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.film_head = film_head

        # """if not init with 1 and 0 for gamma and beta"""
        # self.film1_gammas = nn.Parameter(torch.randn(film_head, planes))
        # self.film1_betas = nn.Parameter(torch.randn(film_head, planes))
        # self.film2_gammas = nn.Parameter(torch.randn(film_head, planes))
        # self.film2_betas = nn.Parameter(torch.randn(film_head, planes))

        self.film1_gammas = nn.Parameter(torch.ones(film_head, planes))
        self.film1_betas = nn.Parameter(torch.zeros(film_head, planes))
        self.film2_gammas = nn.Parameter(torch.ones(film_head, planes))
        self.film2_betas = nn.Parameter(torch.zeros(film_head, planes))

        # self.film1 = nn.ModuleList([CatFilm(planes) for _ in range(film_head)])
        # self.film2 = nn.ModuleList([CatFilm(planes) for _ in range(film_head)])

    def forward(self, x, selection):    # [bs, file_head]: [1,0,...,0] or soft or None
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)

        if selection is not None:
            gamma = torch.mm(selection, self.film1_gammas)      # [bs, planes]
            beta = torch.mm(selection, self.film1_betas)        # [bs, planes]
            out = film(out, gamma, beta)

            # film_out = []
            # for idx in range(self.film_head):
            #     film_out.append(self.film1[idx](out) * selection[:, idx].view(len(x), 1, 1, 1))
            # out = torch.sum(torch.stack(film_out), dim=0)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if selection is not None:
            gamma = torch.mm(selection, self.film2_gammas)      # [bs, planes]
            beta = torch.mm(selection, self.film2_betas)        # [bs, planes]
            out = film(out, gamma, beta)

            # film_out = []
            # for idx in range(self.film_head):
            #     film_out.append(self.film2[idx](out) * selection[:, idx].view(len(x), 1, 1, 1))
            # out = torch.sum(torch.stack(film_out), dim=0)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, classifier=None, num_classes=None,
                 dropout=0.0, global_pool=True,
                 film_head=1, tau=1,
                 num_clusters=8, opt='linear'):
        super(ResNet, self).__init__()
        self.initial_pool = False
        self.film_head = film_head

        # """if not init with 1 and 0 for gamma and beta"""
        # self.film_normalize_gammas = nn.Parameter(torch.randn(film_head, 3))
        # self.film_normalize_betas = nn.Parameter(torch.randn(film_head, 3))

        self.film_normalize_gammas = nn.Parameter(torch.ones(film_head, 3))
        self.film_normalize_betas = nn.Parameter(torch.zeros(film_head, 3))

        # self.film_normalize = nn.ModuleList([CatFilm(3) for _ in range(film_head)])
        inplanes = self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, inplanes, layers[0], film_head=film_head)
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2, film_head=film_head)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2, film_head=film_head)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2, film_head=film_head)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.outplanes = 512

        # selector
        self.selector = Selector(rep_dim=64, num_clusters=num_clusters, opt=opt, metric='cosine', tau=tau)     # cosine

        # handle classifier creation
        if num_classes is not None and num_classes != 0:
            if classifier == 'linear':
                self.cls_fn = nn.Linear(self.outplanes, num_classes)
            elif classifier == 'cosine':
                self.cls_fn = CosineClassifier(self.outplanes, num_classes)
        else:
            self.cls_fn = nn.Identity()

        # initialize everything
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, film_head=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, film_head=film_head))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, film_head=film_head))

        # return nn.Sequential(*layers)
        return nn.ModuleList(layers)       # forward need head_idx, can not use Sequential

    def forward(self, x_list, task_x, gumbel=True, hard=True, selected_idx=None):
        """task_x contains task image samples for task-conditioning."""
        if isinstance(x_list, torch.Tensor):
            x_list = [x_list]
        # features = torch.mean(self.embed(task_x), dim=0, keepdim=True)        # [1, 512]
        features = self.embed(task_x)        # [bs, 512] forward backbone without film
        selection, selection_info = self.selector(features, gumbel=gumbel, hard=hard)       # [1, n_clusters]

        if selected_idx is not None:
            '''select specific film head to forward by ``hard trick`` rather than the argmax head'''
            y_soft = selection_info['y_soft']
            bs, nc = y_soft.shape
            index = selected_idx*torch.ones(bs, 1, device=y_soft.device).long()
            y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)
            # [bs, nc], one hot at selected_idx.
            selection = y_hard - y_soft.detach() + y_soft

        results = []
        for x in x_list:
            x = self.embed(x, selection=selection)
            x = self.cls_fn(x)          # cls_fn is identity if no classifier
            results.append(x)
        return results, selection_info

    def embed(self, x, selection=None, squeeze=True, param_dict=None):
        """
        selection is None: forward resnet18 backbone and skip films.
        """
        if selection is not None:
            assert (selection.shape[1] == self.film_head
                    ), f"Input selection: {selection} does not match `film_head' {self.film_head}."

        """Computing the features"""
        if selection is not None:
            gamma = torch.mm(selection, self.film_normalize_gammas)      # [bs, 3]
            beta = torch.mm(selection, self.film_normalize_betas)        # [bs, 3]
            x = film(x, gamma, beta)

            # film_out = []
            # for idx in range(self.film_head):
            #     film_out.append(self.film_normalize[idx](x) * selection[:, idx].view(len(x), 1, 1, 1))
            # x = torch.sum(torch.stack(film_out), dim=0)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)

        for block_idx in range(len(self.layer1)):
            x = self.layer1[block_idx](x, selection=selection)
        for block_idx in range(len(self.layer2)):
            x = self.layer2[block_idx](x, selection=selection)
        for block_idx in range(len(self.layer3)):
            x = self.layer3[block_idx](x, selection=selection)
        for block_idx in range(len(self.layer4)):
            x = self.layer4[block_idx](x, selection=selection)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        return x.flatten(1)         # x.squeeze()

    def freeze_backbone(self):
        for k, v in self.named_parameters():
            if 'selector' not in k and 'film' not in k and 'cls' not in k and 'running' not in k:
                v.requires_grad = False

    def get_state_dict(self, whole=True):
        if whole:
            """Outputs all the state elements"""
            return self.state_dict()
        else:
            """Outputs the state elements that are domain-specific"""
            return {k: v for k, v in self.state_dict().items()
                    if 'selector' in k or 'film' in k or 'cls' in k or 'running' in k}

    def get_parameters(self, whole=False):
        if whole:
            """Outputs all the parameters"""
            return [v for k, v in self.named_parameters()]
        else:
            """Outputs only the parameters that are domain-specific"""
            return [v for k, v in self.named_parameters()
                    if 'selector' in k or 'film' in k or 'cls' in k]

    def get_trainable_parameters(self):
        return [v for k, v in self.named_parameters()
                if v.requires_grad]

    def get_trainable_film_parameters(self):
        return [v for k, v in self.named_parameters()
                if v.requires_grad and 'film' in k]

    def get_trainable_selector_parameters(self, include_cluster_center=True):
        if include_cluster_center:
            return [v for k, v in self.named_parameters()
                    if v.requires_grad and 'selector' in k]
        else:
            return [v for k, v in self.named_parameters()
                    if v.requires_grad and ('selector' in k and 'prototypes' not in k)]

    def get_trainable_cluster_center_parameters(self):
        return [v for k, v in self.named_parameters()
                if v.requires_grad and 'selector.prototypes' in k]

    def get_trainable_classifier_parameters(self):
        return [v for k, v in self.named_parameters()
                if v.requires_grad and 'cls' in k]


def resnet18(pretrained=False, pretrained_model_path=None, freeze_backbone=False, **kwargs):
    """
        Constructs a FiLM adapted ResNet-18 model.
    """
    model = ResNet(BasicBlockFilm, [2, 2, 2, 2], **kwargs)

    # loading shared convolutional weights
    if pretrained_model_path is not None:
        device = model.get_parameters()[0].device
        ckpt_dict = torch.load(pretrained_model_path, map_location=device)['state_dict']
        shared_state = {k: v for k, v in ckpt_dict.items() if 'cls' not in k}
        model.load_state_dict(shared_state, strict=False)
        print('Loaded shared weights from {}'.format(pretrained_model_path))

    # freeze backbone except film and cls
    if freeze_backbone:
        model.freeze_backbone()

    return model


class Selector(nn.Module):
    """
    Selector tasks feature vector ([bs, 512]) as input.
    """
    def __init__(self, input_dim=512, rep_dim=64, num_clusters=8, opt='linear', metric='cosine', tau=1.0):
        super(Selector, self).__init__()
        self.input_dim = input_dim
        self.rep_dim = rep_dim
        self.n_class = num_clusters
        self.opt = opt
        self.metric = metric
        self.tau = nn.Parameter(torch.ones([]) * tau)       # learnable tau

        self.encoder = adaptor(num_datasets=1, dim_in=input_dim, dim_out=rep_dim, opt=opt)
        # self.hierarchical_net()

        self.prot_shape = (1, 1)
        self.prototype_shape = (self.n_class, self.rep_dim, *self.prot_shape)
        self.prototypes = nn.Parameter(torch.rand(self.prototype_shape))
        self.logit_scale = nn.Parameter(torch.ones([]) * 1)     # 1
        # self.logit_scale = torch.ones([])     # 1
        # self.cluster_centers = nn.Parameter(torch.randn((num_clusters, emb_dim)))

        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)
        # self.ones = torch.ones(self.prototype_shape)

    def forward(self, inputs, gumbel=True, hard=True, average=True):
        """
        :param inputs: [batch_size, fea_embed], [bs,512]
        :param gumbel: whether to use gumbel softmax for selection
        :param hard: whether to use hard selection.
        :param average: whether to average after encoder.
        :return selection: hard selection [bs, n_class] if not average, else [1, n_class]
        """
        bs = inputs.shape[0]
        embeddings = self.encoder([inputs])[0]     # [bs, 64]
        if average:
            embeddings = torch.mean(embeddings, dim=0, keepdim=True)      # [1, 64]
            bs = 1
        embeddings = F.sigmoid(embeddings)          # apply sigmoid activation on embeddings
        embeddings = embeddings.view(bs, self.rep_dim, *self.prot_shape)    # [bs, 64, 1, 1]

        dist = self._distance(embeddings).view(bs, self.n_class)        # [bs, n_proto]  similarity

        if self.metric == 'euclidean':
            dist = -dist        # dist to similarity -> warning: since no sqrt, value is very large

        # gumbel softmax or softmax
        dim = 1
        if gumbel:
            y_soft = F.gumbel_softmax(dist, dim=dim, hard=False, tau=self.tau.exp())
        else:
            y_soft = F.softmax(dist, dim=dim)
        normal_soft = F.softmax(dist, dim=dim)

        if hard:
            # hard trick
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(dist, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            selection = y_hard - y_soft.detach() + y_soft
        else:
            selection = y_soft

        return selection, {
            'y_soft': y_soft,
            'normal_soft': normal_soft,
        }

    def _distance(self, x):
        if callable(self.metric):
            dist = self.metric(x)
        elif self.metric == 'dot':      # checked,    == torch.mm(x, proto.T)
            # dist = self.logit_scale.exp() * F.conv2d(x, weight=self.prototypes)
            dist = F.conv2d(x, weight=self.prototypes)
        elif self.metric == 'euclidean':
            dist = self._l2_convolution(x)      # checked,  == torch.sum((a-b)**2)
        elif self.metric == 'cosine':   # checked
            x = x / x.norm(dim=1, keepdim=True)
            weight = self.prototypes / self.prototypes.norm(dim=1, keepdim=True)
            dist = self.logit_scale.exp() * F.conv2d(x, weight=weight)
        else:
            raise NotImplementedError('Metric {} not implemented.'.format(self.metric))

        return dist

    def _l2_convolution(self, x):
        """
        Taken from https://github.com/cfchen-duke/ProtoPNet
        apply self.prototype_vectors as l2-convolution filters on input x
        == torch.sum((x-proto)**2)
        """
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        p2 = self.prototypes ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=self.prototypes)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances

    def get_state_dict(self):
        """Outputs all the state elements"""
        return self.state_dict()

    def get_parameters(self):
        """Outputs all the parameters"""
        return [v for k, v in self.named_parameters()]


if __name__ == '__main__':
    import os

    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    import numpy as np

    '''resnet18 moe'''
    res = resnet18(film_head=8, freeze_backbone=True,
                   pretrained_model_path=os.path.join('../../URL-experiments/saved_results/url',
                                                      'weights', 'url', 'model_best.pth.tar'))

    # bs 5, img (3, 128, 128), film_head 8
    x_ = torch.randn(5, 3, 128, 128)
    selection_ = torch.zeros(5, 8)
    selection_[:, 1] = 1
    out_ = res.embed(x_, selection_)

    param_count = 0
    for k, v in res.named_parameters():
        print(f'{k}: {v.shape}, {v.requires_grad}')
        if v.requires_grad:
            param_count += 1
    print(f'param_count: {param_count}')

    '''selector'''
    # net = Selector(tau=1)
    #
    # # bs 5, feature size 512
    # inputs_ = torch.randn(5, 512)
    #
    # fig, axes = plt.subplots(5, 5, figsize=(20, 16))
    #
    # for i, tau in enumerate(np.logspace(-1, 1, 5)):
    #     net.tau = tau
    #     num_seed = 5 if i < 4 else 4
    #     for seed_i in range(num_seed):
    #         selection_, y_soft_ = net(inputs_, gumbel=True)
    #
    #         idx, idy = seed_i, i
    #         sns.heatmap(y_soft_['y_soft'].detach().cpu().numpy(), annot=True, fmt=".2f", linewidth=.5, cbar=False,
    #                     ax=axes[idy, idx])
    #         axes[idy, idx].set_title(f'tau={tau:.2f}')
    #
    # sns.heatmap(y_soft_['normal_soft'].detach().cpu().numpy(), annot=True, fmt=".2f", linewidth=.5, cbar=False,
    #             ax=axes[4, 4])
    # axes[4, 4].set_title(f'softmax')
    #
    # plt.savefig(os.path.join('D:', 'Downloads', 'fig.png'), dpi=400, bbox_inches='tight')
