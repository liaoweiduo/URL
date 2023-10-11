import sys
import os
import numpy as np
import time
import copy
from datetime import datetime
import json


def return_time():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def template_exp_sh(target, path, name, params, out_path='../avalanche-experiments/out/task.out', cuda=0):
    """
    Generate sh file from 1 params dict
    :param target: target py file with relative path to it.
    :param path: store the sh, 'tests/tasks/TASK_NAME'
    :param name: sh file name, '1'
    :param params: a list of param dict
    :param out_path: path to the root of std out file.  No use
    :param cuda: device used
    """
    '''Make dir'''
    if not os.path.exists(path):
        os.makedirs(path)

    template_str = \
        f"#!/bin/sh\n" \
        f"ulimit -n 50000\n" \
        "export META_DATASET_ROOT=../meta-dataset\n"\
        "export RECORDS=../datasets/tfrecords\n"

    for param_idx, param in enumerate(params):
        '''Param to str'''
        param_str = ''
        for key, value in param.items():
            if type(value) is not bool:
                param_str += f" --{key} {value}"
            elif value is True:
                param_str += f" --{key}"
        # param_str = ' '.join([f"--{key} {value}" for key, value in params.items() if type(value) is not bool])
        # param_str_ = ' '.join([f"--{key}" for key, value in params.items() if value is True])       # True
        # param_str = ' '.join([param_str, param_str_])

        # template_str += \
        #     f"CUDA_VISIBLE_DEVICES={cuda} python3 {target}{param_str}" \
        #     f" >> {out_path} 2>&1\n"
        template_str += \
            f"CUDA_VISIBLE_DEVICES={cuda} python3 -u {target}{param_str}\n"

    '''Write to file'''
    with open(os.path.join(path, f'{name}.sh'), 'w', newline='') as f:
        f.write(template_str)


def template_sustech(name_list, cmd_path, path):
    """
    Generate slurm bash for file_list and 1 sh contains all sbatch $run_id$.slurm
    """
    '''Make dir'''
    if not os.path.exists(path):
        os.makedirs(path)

    task_str = f"#!/bin/sh"

    '''Generate slurm bash'''
    for idx, name in enumerate(name_list):
        task_str += f"\nsh slurm{name}.sh"

        template_str = \
            f"#!/bin/bash\n" \
            f"cd ~\n" \
            f"sbatch url_bash.slurm {name} {path.split('/')[-1]}\n"

        '''Write to file'''
        with open(os.path.join(path, f'slurm{name}.sh'), 'w', newline='') as f:
            f.write(template_str)

    '''Generate task.sh'''
    with open(os.path.join(path, 'task.sh'), 'w', newline='') as f:
        f.write(task_str)


def template_hisao(name_list, cmd_path, path):
    """
    Generate bash for file_list and 1 sh contains all sh $run_id$.bash.
    this bash is to cd the working path.
    """
    '''Make dir'''
    if not os.path.exists(path):
        os.makedirs(path)

    task_str = f"#!/bin/sh"

    '''Generate slurm bash'''
    for idx, name in enumerate(name_list):
        task_str += f"\nsh {name}.bash >> {name}.out 2>&1"

        template_str = \
            f"#!/bin/sh\n" \
            f"cd ../../../URL\n" \
            f"sh {cmd_path}/{name}.sh\n"

        '''Write to file'''
        with open(os.path.join(path, f'{name}.bash'), 'w', newline='') as f:
            f.write(template_str)

    '''Generate task.sh'''
    with open(os.path.join(path, 'task.sh'), 'w', newline='') as f:
        f.write(task_str)


def generate_params(common_args, param_grid, exp_name_template):
    keys = set(param_grid.keys())

    print(exp_name_template, param_grid)

    def unfold(_params, _param, _choice=None):
        """recursion to get all choice of params.
            _choice: (key, value)
        """
        _param = copy.deepcopy(_param)
        if _choice is not None:     # assign value
            _param[_choice[0]] = _choice[1]

        if len(_param.keys()) == len(keys):
            '''complete'''
            _params.append(_param)
        else:
            '''select 1 unsigned key and call unfold'''
            selected_key = list(keys - set(_param.keys()))[0]
            for choice in param_grid[selected_key]:
                unfold(_params, _param, _choice=(selected_key, choice))

    '''Generate instance params for grid search in param_scope'''
    params = []
    unfold(params, dict(), None)

    for iter, param in enumerate(params):
        '''
        Merge common_args into param but param has higher priority
        Generate exp_name according to param
        '''
        merged_param = common_args.copy()
        merged_param.update(param)

        param_dot_to_underline = {k.replace('.', '_'): v for k, v in merged_param.items()}
        # format does not allow key `.` (e.g., train.learning_rate)
        exp_name = exp_name_template.replace('.', '_').format(**param_dot_to_underline)
        exp_name = exp_name.replace('.', '_')      # for float, change '0.1' to '0_1' for potential problem in Windows.
        merged_param['model.dir'] = merged_param['model.dir'] + exp_name
        params[iter] = merged_param

    return params


def main(params, fix_device=True, start_iter=0, func=template_hisao):
    """Generate sh files"""
    names = []
    params_temp = []
    iter = start_iter
    for idx, param in enumerate(params):
        if len(params_temp) < num_runs_1sh:
            params_temp.append(param)

        if len(params_temp) == num_runs_1sh or idx == len(params) - 1:  # every num_runs_1sh or last runs
            print(f'Generate {iter}.sh with params: {params_temp}.')
            template_exp_sh(
                target=target,
                path=f'{task_store_path}/{task_name}',
                name=iter,
                params=params_temp,
                # out_path=f"{exp_root}/out/{task_name}-{iter}.out",
                cuda=0 if fix_device else iter,
            )
            names.append(iter)
            params_temp = []
            iter += 1

    '''Generate bash for server'''
    # template_sustech, template_hisao
    func(
        name_list=names,
        cmd_path=f'{task_root}/{task_name}',
        path=f'{task_store_path}/{task_name}'
    )


target = 'train_net_pmo.py'
task_name = return_time()   # defined by time
print(f'task: {task_name}')
task_store_path = '../URL-experiments/tasks/'
func = template_sustech
task_root = 'scripts'        # path for sh in the working path
# task_root = '../URL-experiments/tasks'        # path for sh out of working path
num_runs_1sh = 1        # num of runs in 1 sh file
fix_device = True      # cuda self-increase for each run if False, else use cuda:0
start_iter = 0
common_args = {
    'model.name': 'pmo',
    'model.dir': '../URL-experiments/saved_results/',   # need to add a folder name
    'model.num_clusters': 10, 'model.backbone': 'resnet18_moe',
    'model.pretrained': True, 'source': '../URL-experiments/saved_results/url',
    'source_moe': '../URL-experiments/saved_results/url',
    'data.train': 'ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower',
    'data.val': 'ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower',
    'data.test': 'ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower '
                 'traffic_sign mscoco mnist cifar10 cifar100',
    'train.type': 'standard',
    'train.freeze_backbone': True,
    'train.loss_type': 'task+ce+pure+hv',
    'train.optimizer': 'adam', 'train.learning_rate': 1e-5, 'train.weight_decay': 2e-7,
    'train.selector_learning_rate': 1e-3,
    'train.max_iter': 1000, 'train.summary_freq': 100, 'train.pool_freq': 10,
    'train.mo_freq': 100, 'train.n_mo': 10, 'train.n_obj': 2, 'train.hv_coefficient': 1,
    'train.cosine_anneal_freq': 200, 'train.eval_freq': 200, 'train.eval_size': 50,
}

params = []


"""
exp: try 1 iter = 1 tasks 
"""
num_runs_1sh = 9        # num of runs in 1 sh file
common_args.update({
    'tag': 'pmo-debug-numcluster',
    'train.max_iter': 1000, 'train.summary_freq': 100, 'train.pool_freq': 10,
    'train.mo_freq': 10, 'train.n_mo': 1,
    'train.cosine_anneal_freq': 1000, 'train.eval_freq': 200,
    'train.selector_learning_rate': 1e-4,
})
param_grid = {
    'train.learning_rate': [1e-5, 1e-4, 1e-3],
    'train.loss_type': ['task+ce'],
    'model.num_clusters': [1, 2, 5],
    # 'train.loss_type': ['task+ce+pure+hv', 'task+ce+pure', 'task+pure+hv'],
    # 'train.pure_coefficient': [10, 100],         # [1, 10],
    # 'train.hv_coefficient': [1, 10, 100],
}
exp_name_template = common_args['tag'] + \
                    '-lt{train.loss_type}' + \
                    '-lr{train.learning_rate}' + \
                    '-nc{model.num_clusters}' # + \
                    # '-pc{train.pure_coefficient}' + \
                    # '-hvc{train.hv_coefficient}'

params_temp = generate_params(common_args, param_grid, exp_name_template)
for p in params_temp:
    p['train.weight_decay'] = p['train.learning_rate'] / 50
    # p['train.selector_learning_rate'] = p['train.learning_rate']
params.extend(params_temp)


"""
exp: for debug
"""
# num_runs_1sh = 1        # num of runs in 1 sh file
# common_args.update({
#     'tag': 'pmo-debug-1000i-store_feas',
#     'train.max_iter': 1000, 'train.summary_freq': 2000, 'train.pool_freq': 10,
#     'train.mo_freq': 100, 'train.n_mo': 1,
#     'train.cosine_anneal_freq': 2000, 'train.eval_freq': 20000,    # no eval
#     'train.loss_type': 'task+ce+pure',
# })
# param_grid = {
#     'train.selector_learning_rate': [1e-3],
#     'train.learning_rate': [1e-3],
# }
# exp_name_template = common_args['tag'] + \
#                     '-slr{train.selector_learning_rate}' + \
#                     '-flr{train.learning_rate}'
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# for p in params_temp:
#     p['train.weight_decay'] = p['train.learning_rate'] / 50
# params.extend(params_temp)


main(params, fix_device, start_iter, func=func)
