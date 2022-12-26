################### Train Multiple Models with pmo ###################
ulimit -n 50000
export META_DATASET_ROOT=../meta-dataset
export RECORDS=../datasets/tfrecords

NAME="M{}-net"

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net_pmo_gt8models.py --model.name=$NAME --model.num_clusters 8 \
    --model.dir ../URL-experiments/saved_results/sdl \
    --out.dir ../URL-experiments/saved_results/pmo-gt8models \
    --data.train ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower \
    --data.val ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower \
    --data.test ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower \
    --train.learning_rate=3e-2 --train.max_iter=2000 --train.summary_freq=1 \
    --train.hv_coefficient=1e-2 \
    --train.cosine_anneal_freq=50 --train.eval_freq=50 --train.eval_size 300 \
    1> ../URL-experiments/out/pmo_gt8models.out  # 2> ../URL-experiments/out/pmo.err
#    2>&1

# eval_freq 4800
