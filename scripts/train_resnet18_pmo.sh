################### Train Multiple Models with pmo ###################
ulimit -n 50000
export META_DATASET_ROOT=../meta-dataset
export RECORDS=../datasets/tfrecords

NAME="pmo"
OUTNAME="pmo-moe-sgd-ce_select-cluster_nonlinear-lr1e-3-sele-1e-1"

CUDA_VISIBLE_DEVICES=0 python train_net_pmo.py \
    --model.name=$NAME --model.num_clusters 8 --model.backbone resnet18_moe \
    --model.dir ../URL-experiments/saved_results/$OUTNAME \
    --model.pretrained --source ../URL-experiments/saved_results/url \
    --data.train ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower \
    --data.val ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower \
    --data.test ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower \
    --train.optimizer=momentum --train.learning_rate=1e-3 --train.weight_decay=5e-5 \
    --train.max_iter=200 --train.summary_freq=10 \
    --train.type=standard --train.freeze_backbone --train.loss_type=task+pure+hv \
    --train.n_mo=10 --train.hv_coefficient=0.001 --train.mo_freq=10 \
    --train.cosine_anneal_freq=200 --train.eval_freq=200 --train.eval_size 50 # \
#    1> ../URL-experiments/out/$OUTNAME.out  # 2> ../URL-experiments/out/pmo.err
#    2>&1

