################### Train Multiple Models with pmo ###################
ulimit -n 50000
export META_DATASET_ROOT=../meta-dataset
export RECORDS=../datasets/tfrecords

################### Training Single Domain Learning Networks ###################
function train_fn {
    CUDA_VISIBLE_DEVICES=$1 python train_net_pmo_classifier.py --model.dir ./saved_results/sdl --map.dir ./saved_results/pmo --model.name=$2 \
    --data.train ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower \
    --data.val ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower \
    --data.test ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower \
    --train.batch_size=$3 --train.learning_rate=$4 --train.max_iter=$5 --train.cosine_anneal_freq=$6 --train.eval_freq=$6
}

# Train an single domain learning network on every training dataset (the following models could be trained in parallel)
#        d  name  bs  lr   iter  val freq
train_fn 0 M0-net 64 3e-2 480000 48000
#&
#train_fn 1 M1-net 64 3e-2 480000 48000 &
#train_fn 2 M2-net 64 3e-2 480000 48000 &
#train_fn 3 M3-net 64 3e-2 480000 48000 &
#train_fn 4 M4-net 64 3e-2 480000 48000 &
#train_fn 5 M5-net 64 3e-2 480000 48000 &
#train_fn 6 M6-net 64 3e-2 480000 48000 &
#train_fn 7 M7-net 64 3e-2 480000 48000 &





#NAME="imagenet-net"; TRAINSET="ilsvrc_2012"; BATCH_SIZE=64; LR="3e-2"; MAX_ITER=480000; ANNEAL_FREQ=48000
#train_fn $NAME $TRAINSET $BATCH_SIZE $LR $MAX_ITER $ANNEAL_FREQ
#
## Omniglot
#NAME="omniglot-net"; TRAINSET="omniglot"; BATCH_SIZE=16; LR="3e-2"; MAX_ITER=50000; ANNEAL_FREQ=3000
#train_fn $NAME $TRAINSET $BATCH_SIZE $LR $MAX_ITER $ANNEAL_FREQ
#
## Aircraft
#NAME="aircraft-net"; TRAINSET="aircraft"; BATCH_SIZE=8; LR="3e-2"; MAX_ITER=50000; ANNEAL_FREQ=3000
#train_fn $NAME $TRAINSET $BATCH_SIZE $LR $MAX_ITER $ANNEAL_FREQ
#
## Birds
#NAME="birds-net"; TRAINSET="cu_birds"; BATCH_SIZE=16; LR="3e-2"; MAX_ITER=50000; ANNEAL_FREQ=3000
#train_fn $NAME $TRAINSET $BATCH_SIZE $LR $MAX_ITER $ANNEAL_FREQ
#
## Textures
#NAME="textures-net"; TRAINSET="dtd"; BATCH_SIZE=32; LR="3e-2"; MAX_ITER=50000; ANNEAL_FREQ=1500
#train_fn $NAME $TRAINSET $BATCH_SIZE $LR $MAX_ITER $ANNEAL_FREQ
#
## Quick Draw
#NAME="quickdraw-net"; TRAINSET="quickdraw"; BATCH_SIZE=64; LR="1e-2"; MAX_ITER=480000; ANNEAL_FREQ=48000
#train_fn $NAME $TRAINSET $BATCH_SIZE $LR $MAX_ITER $ANNEAL_FREQ
#
## Fungi
#NAME="fungi-net"; TRAINSET="fungi"; BATCH_SIZE=32; LR="3e-2"; MAX_ITER=480000; ANNEAL_FREQ=15000
#train_fn $NAME $TRAINSET $BATCH_SIZE $LR $MAX_ITER $ANNEAL_FREQ
#
## VGG Flower
#NAME="vgg_flower-net"; TRAINSET="vgg_flower"; BATCH_SIZE=8; LR="3e-2"; MAX_ITER=50000; ANNEAL_FREQ=1500
#train_fn $NAME $TRAINSET $BATCH_SIZE $LR $MAX_ITER $ANNEAL_FREQ

echo "All domain-specific networks are trained!"
