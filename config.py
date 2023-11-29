import argparse
from paths import PROJECT_ROOT

parser = argparse.ArgumentParser(description='Train prototypical networks')

# data args
parser.add_argument('--data.train', type=str, default='cu_birds', metavar='TRAINSETS', nargs='+', help="Datasets for training extractors")
parser.add_argument('--data.val', type=str, default='cu_birds', metavar='VALSETS', nargs='+',
                    help="Datasets used for validation")
parser.add_argument('--data.test', type=str, default='cu_birds', metavar='TESTSETS', nargs='+',
                    help="Datasets used for testing")
parser.add_argument('--data.num_workers', type=int, default=32, metavar='NEPOCHS',
                    help="Number of workers that pre-process images in parallel")

# model args
default_model_name = 'noname'
parser.add_argument('--model.name', type=str, default=default_model_name, metavar='MODELNAME',
                    help="A name you give to the extractor".format(default_model_name))
parser.add_argument('--tag', type=str, default="checkpoint store folder name.")
parser.add_argument('--model.backbone', default='resnet18', help="Use ResNet18 for experiments (default: False)")
parser.add_argument('--model.classifier', type=str, default='linear', choices=['none', 'linear', 'cosine'], help="Do classification using cosine similatity between activations and weights")
parser.add_argument('--model.dropout', type=float, default=0, help="Adding dropout inside a basic block of widenet")
parser.add_argument('--model.pretrained', action='store_true', help="Using pretrained model for learning or not")
parser.add_argument('--model.num_clusters', type=int, default=8, help="Number of clusters for multi-domain learning")
# adaptor args
parser.add_argument('--adaptor.opt', type=str, default='linear', help="type of adaptor, linear or nonlinear")
# Selector model args
parser.add_argument('--cluster.opt', type=str, default='nonlinear', help="type of cluster model, linear or nonlinear")
parser.add_argument('--cluster.logit_scale', type=float, default=0.5, metavar='LOGIT_SCALE',
                    help='logit scale s, and softmax(exp(s) * dist) -> sim.')

# train args
parser.add_argument('--train.type', type=str, choices=['standard', '1shot', '5shot'],
                    default='standard', metavar='TRAIN_TYPE',
                    help="standard varying number of ways and shots as in Meta-Dataset, "
                         "1shot for five-way-one-shot "
                         "and 5shot for varying-way-five-shot evaluation.")
parser.add_argument('--train.batch_size', type=int, default=10, metavar='BS',
                    help='number of images in a batch. '
                         'In episodic setting, it is the batch size for tasks.')
parser.add_argument('--train.max_iter', type=int, default=500000, metavar='NEPOCHS',
                    help='number of epochs to train (default: 10000)')
parser.add_argument('--train.weight_decay', type=float, default=7e-4, metavar='WD',
                    help="weight decay coef")
parser.add_argument('--train.optimizer', type=str, default='momentum', metavar='OPTIM',
                    help='optimization method (default: momentum)')
parser.add_argument('--train.learning_rate', type=float, default=0.03, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--train.sigma', type=float, default=1, metavar='SIGMA',
                    help='weight of CKA loss on features')
parser.add_argument('--train.beta', type=float, default=1, metavar='BETA',
                    help='weight of KL-divergence loss on logits')
parser.add_argument('--train.lr_policy', type=str, default='cosine', metavar='LR_policy',
                    help='learning rate decay policy')
parser.add_argument('--train.lr_decay_step_gamma', type=int, default=1e-1, metavar='DECAY_GAMMA',
                    help='the value to divide learning rate by when decayin lr')
parser.add_argument('--train.lr_decay_step_freq', type=int, default=10000, metavar='DECAY_FREQ',
                    help='the value to divide learning rate by when decayin lr')
parser.add_argument('--train.exp_decay_final_lr', type=float, default=8e-5, metavar='FINAL_LR',
                    help='the value to divide learning rate by when decayin lr')
parser.add_argument('--train.exp_decay_start_iter', type=int, default=30000, metavar='START_ITER',
                    help='the value to divide learning rate by when decayin lr')
parser.add_argument('--train.cosine_anneal_freq', type=int, default=4000, metavar='ANNEAL_FREQ',
                    help='the value to divide learning rate by when decayin lr')
parser.add_argument('--train.nesterov_momentum', action='store_true',
                    help="If to augment query images in order to average the embeddings")

# evaluation during training
parser.add_argument('--train.summary_freq', type=int, default=200, metavar='SUMMARY_FREQ',
                    help='How often to summary epoch acc and loss during training')
parser.add_argument('--train.eval_freq', type=int, default=5000, metavar='EVAL_FREQ',
                    help='How often to evaluate model during training')
parser.add_argument('--train.eval_size', type=int, default=300, metavar='EVAL_SIZE',
                    help='How many episodes to sample for validation')
parser.add_argument('--train.resume', type=int, default=1, metavar='RESUME_TRAIN',
                    help="Resume training starting from the last checkpoint (default: True)")
parser.add_argument('--train.best_criteria', type=str, default='domain', metavar='BEST_MODEL',
                    help='Best model based on which [hv, cluster, domain, avg_span, min_cd]')

# pmo training
parser.add_argument('--train.selector_learning_rate', type=float, default=0.03, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--train.inner_learning_rate', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.0001) if necessary')
parser.add_argument('--train.freeze_backbone', action='store_true', help="Freeze resnet18 backbone when using MOE")
parser.add_argument('--train.cluster_center_mode', type=str, default='prototypes', metavar='CLUSTER_CENTER_MODE',
                    choices=['kmeans', 'hierarchical', 'prototypes', 'mov_avg'],
                    help='use kmeans(average) or hierarchical clustering net.')
parser.add_argument('--train.sim_gumbel', action='store_true',
                    help='use gumbel for selector output: similarity.')
parser.add_argument('--train.cond_mode', type=str, default='film_opt', metavar='CONDITIONING_MODE',
                    choices=['film_random', 'film_opt', 'pa'],
                    help='use randn init film (film-randm) or (1 and 0) init film (film-opt).')
parser.add_argument('--train.mov_avg_alpha', type=float, default=0.2, metavar='MOV_AVG_ALPHA',
                    help='alpha on current class centroid. only activate if use mov_avg. ')
parser.add_argument('--train.gumbel_tau', type=float, default=1, metavar='GUMBEL_TAU',
                    help='temperature for gumbel softmax. (default: 1)'
                         'use exp(tau) to ensure positive, so 1 means e.')
parser.add_argument('--train.loss_type', type=str, default='task+kd+pure+hv+ce', metavar='LOSS_TYPE',
                    help='backward losses.'
                         'can be any combination of task, hv, task+pure+hv, pure, pure+hv')
parser.add_argument('--train.cluster_loss_type', type=str, default='ce', metavar='CL_TYPE',
                    help='choice: ce, kl')
parser.add_argument('--train.pool_freq', type=int, default=1, metavar='POOL_FREQ',
                    help='How often to update pool. ')
parser.add_argument('--train.mo_freq', type=int, default=2000, metavar='MO_FREQ',
                    help='How often to apply mo train phase. '
                         'Usually equals to train.summary_freq, that do mo train at the last iter before summary.')
parser.add_argument('--train.recon_weight', type=float, default=0.001, metavar='WEIGHT',
                    help='coeffient for reconstruction loss.')
parser.add_argument('--train.kd_type', type=str, default='kernelcka', metavar='KD_TYPE',
                    help='choice: kl, kernelcka, film_param_l2')
parser.add_argument('--train.kd_T_extent', type=float, default=2, metavar='KD_T_EXTENT',
                    help='max{kd_coefficient*(1-t/(cosine_anneal_freq*kd_T_extent)), 0}.')
parser.add_argument('--train.kd_coefficient', type=float, default=1, metavar='KD_COEFFICIENT',
                    help='coeffient for distillation loss.')
parser.add_argument('--train.ce_coefficient', type=float, default=1, metavar='CE_COEFFICIENT',
                    help='coeffient for selection ce loss.')
parser.add_argument('--train.pure_coefficient', type=float, default=1, metavar='PURE_COEFFICIENT',
                    help='coeffient for pure task ncc loss.')
parser.add_argument('--train.hv_coefficient', type=float, default=1, metavar='HV_COEFFICIENT',
                    help='coeffient for hv loss.')
parser.add_argument('--train.et_coefficient', type=float, default=1, metavar='ET_COEFFICIENT',
                    help='coeffient for et loss.')
parser.add_argument('--train.max_sampling_iter_for_pool', type=int, default=1, metavar='ITER',
                    help='Number of sampling iteration for sampling tasks to put into pool.')
parser.add_argument('--train.n_mo', type=int, default=9, metavar='N_MO',
                    help='number of MO sampling to train. '
                         'each sample chooses n_obj clusters randomly to construct 1 mo_obj.')
parser.add_argument('--train.n_et_cond', type=int, default=1, metavar='N_ET',
                    help='entanglement improvement: number of task sampling for each cluster. '
                         'each sample is used to condition model.')
parser.add_argument('--train.n_et_update', type=int, default=1, metavar='N_ET',
                    help='entanglement improvement: number of task sampling for each condition model. '
                         'each sample is used to generate query ncc loss.')
parser.add_argument('--train.mo_task_type', type=str, choices=['standard', '1shot', '5shot'],
                    default='standard', metavar='TASK_TYPE',
                    help="meta-train type for task sampling from pool, "
                         "standard varying number of ways and shots (ten-query) as in Meta-Dataset, "
                         "1shot for five-way-one-shot-ten-query "
                         "and 5shot for varying-way-five-shot-ten-query evaluation.")
parser.add_argument('--train.n_obj', type=int, default=2, metavar='N_OBJ',
                    help='number of objs (models considered in 1 iter) to train')
parser.add_argument('--train.mix_mode', type=str, default='cutmix', metavar='MIX_MODE',
                    choices=['cutmix', 'mixup'],
                    help='mix mode for mixer. ')
parser.add_argument('--train.n_mix', type=int, default=2, metavar='N_MIX',
                    help='number of mixed tasks generated in 1 iter')
parser.add_argument('--train.n_mix_source', type=int, default=2, metavar='N_MIX_SOURCE',
                    help='number of tasks used to generate 1 mixed task')
parser.add_argument('--train.ref', type=int, default=1, metavar='REF',
                    help='absolute reference point localtion for calculate hv.')

# creating a database of features
parser.add_argument('--dump.name', type=str, default='', metavar='DUMP_NAME',
                    help='Name for dumped dataset of features')
parser.add_argument('--dump.mode', type=str, default='test', metavar='DUMP_MODE',
                    help='What split of the original dataset to dump')
parser.add_argument('--dump.size', type=int, default=600, metavar='DUMP_SIZE',
                    help='Howe many episodes to dump')


# test args
parser.add_argument('--test.size', type=int, default=600, metavar='TEST_SIZE',
                    help='The number of test episodes sampled')
parser.add_argument('--test.mode', type=str, choices=['mdl', 'sdl'], default='mdl', metavar='TEST_MODE',
                    help="Test mode: multi-domain learning (mdl) or single-domain learning (sdl) settings")
parser.add_argument('--test.type', type=str, choices=['standard', '1shot', '5shot'], default='standard', metavar='LOSS_FN',
                    help="meta-test type, standard varying number of ways and shots as in Meta-Dataset, 1shot for five-way-one-shot and 5shot for varying-way-five-shot evaluation.")
parser.add_argument('--test.distance', type=str, choices=['cos', 'l2'], default='cos', metavar='DISTANCE_FN',
                    help="feature similarity function")
parser.add_argument('--test.loss-opt', type=str, choices=['ncc', 'knn', 'lr', 'svm', 'scm'], default='ncc', metavar='LOSS_FN',
                    help="Loss function for meta-testing, knn or prototype loss (ncc), Support Vector Machine (svm), Logistic Regression (lr) or Mahalanobis Distance (scm)")
parser.add_argument('--test.feature-norm', type=str, choices=['l2', 'none'], default='none', metavar='LOSS_FN',
                    help="normalization options")

# task-specific adapters
parser.add_argument('--test.tsa-ad-type', type=str, choices=['residual', 'serial', 'none'], default='none', metavar='TSA_AD_TYPE',
                    help="adapter type")
parser.add_argument('--test.tsa-ad-form', type=str, choices=['matrix', 'vector', 'none'], default='matrix', metavar='TSA_AD_FORM',
                    help="adapter form")
parser.add_argument('--test.tsa-opt', type=str, choices=['alpha', 'beta', 'alpha+beta'], default='alpha+beta', metavar='TSA_OPT',
                    help="task adaptation option")
parser.add_argument('--test.tsa-init', type=str, choices=['random', 'eye'], default='eye', metavar='TSA_INIT',
                    help="initialization for adapter")

# path args
parser.add_argument('--model.dir', default='', type=str, metavar='PATH',
                    help='path of single domain learning models')
parser.add_argument('--out.dir', default='', type=str, metavar='PATH',
                    help='directory to output the result and checkpoints')
parser.add_argument('--source', default='', type=str, metavar='PATH',
                    help='path of pretrained model')
parser.add_argument('--source_moe', default='', type=str, metavar='PATH',
                    help='path of pretrained model for moe.')

# for jupyter
parser.add_argument('-f', default='', type=str, metavar='F',
                    help='jupyter argument')

# log args
args = vars(parser.parse_args())
if not args['model.dir']:
    args['model.dir'] = PROJECT_ROOT
if not args['out.dir']:
    args['out.dir'] = args['model.dir']

BATCHSIZES = {
                "ilsvrc_2012": 448,
                "omniglot": 64,
                "aircraft": 64,
                "cu_birds": 64,
                "dtd": 64,
                "quickdraw": 64,
                "fungi": 64,
                "vgg_flower": 64
                }

LOSSWEIGHTS = {
                "ilsvrc_2012": 1,
                "omniglot": 1,
                "aircraft": 1,
                "cu_birds": 1,
                "dtd": 1,
                "quickdraw": 1,
                "fungi": 1,
                "vgg_flower": 1
                }

# lambda^f in our paper
KDFLOSSWEIGHTS = {
                    "ilsvrc_2012": 4,
                    "omniglot": 1,
                    "aircraft": 1,
                    "cu_birds": 1,
                    "dtd": 1,
                    "quickdraw": 1,
                    "fungi": 1,
                    "vgg_flower": 1
                }
# lambda^p in our paper
KDPLOSSWEIGHTS = {
                    "ilsvrc_2012": 4,
                    "omniglot": 1,
                    "aircraft": 1,
                    "cu_birds": 1,
                    "dtd": 1,
                    "quickdraw": 1,
                    "fungi": 1,
                    "vgg_flower": 1
                }
# k in our paper
KDANNEALING = {
                    "ilsvrc_2012": 5,
                    "omniglot": 2,
                    "aircraft": 1,
                    "cu_birds": 1,
                    "dtd": 1,
                    "quickdraw": 2,
                    "fungi": 2,
                    "vgg_flower": 1
                }