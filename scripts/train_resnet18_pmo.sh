################### Train Multiple Models with pmo ###################
ulimit -n 50000
export META_DATASET_ROOT=../meta-dataset
export RECORDS=../datasets/tfrecords

NAME="pmo"
OUTNAME="pmo-moe"

CUDA_VISIBLE_DEVICES=0 python train_net_pmo.py \
    --model.name=$NAME --model.num_clusters 8 --model.backbone resnet18_moe \
    --model.dir ../URL-experiments/saved_results/$OUTNAME \
    --model.pretrained --source ../URL-experiments/saved_results/url \
    --data.train ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower \
    --data.val ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower \
    --data.test ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower \
    --train.optimizer=adam --train.learning_rate=1e-4 --train.weight_decay=5e-4 \
    --train.max_iter=2000 --train.summary_freq=100 \
    --train.type=standard --train.freeze_backbone --train.loss_type=task+hv \
    --train.n_mo=10 --train.hv_coefficient=1.0 --train.mo_freq=100 \
    --train.cosine_anneal_freq=200 --train.eval_freq=200 --train.eval_size 300 # \
#    1> ../URL-experiments/out/$OUTNAME.out  # 2> ../URL-experiments/out/pmo.err
#    2>&1

