################### Train Multiple Models with pmo ###################
ulimit -n 50000
export META_DATASET_ROOT=../meta-dataset
export RECORDS=../datasets/tfrecords

NAME="pmo"
OUTNAME="pmo-ab-task-pure-ce-flr1e-5-slr1e-2-clr1e-1"

CUDA_VISIBLE_DEVICES=0 python train_net_pmo.py \
    --model.name=$NAME --model.num_clusters 10 --model.backbone resnet18_moe \
    --model.dir ../URL-experiments/saved_results/$OUTNAME \
    --model.pretrained --source ../URL-experiments/saved_results/url \
    --data.train ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower \
    --data.val ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower mscoco \
    --data.test ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower traffic_sign mscoco mnist cifar10 cifar100 \
    --train.optimizer=adam --train.learning_rate=1e-5 --train.weight_decay=2e-7 \
    --train.selector_learning_rate=1e-2 --train.cluster_center_learning_rate=1e-1 \
    --train.max_iter=100 --train.summary_freq=10 \
    --train.type=standard --train.freeze_backbone --train.loss_type=task+pure+ce \
    --train.n_mo=1 --train.hv_coefficient=1 --train.mo_freq=1 \
    --train.cosine_anneal_freq=100 --train.eval_freq=100 --train.eval_size 50 # \
#    1> ../URL-experiments/out/$OUTNAME.out  # 2> ../URL-experiments/out/pmo.err
#    2>&1

