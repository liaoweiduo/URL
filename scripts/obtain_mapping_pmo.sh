################### Obtain Class Mapping ###################
ulimit -n 50000
export META_DATASET_ROOT=../meta-dataset
export RECORDS=../datasets/tfrecords

NAME="C-net"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python obtain_mapping_pmo.py --model.name=$NAME --model.num_clusters 8 \
    --model.dir ../URL-experiments/saved_results/pmo \
    --data.train ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower \
    --data.val ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower \
    --data.test ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower \
    1> ../URL-experiments/out/pmo-obtain_mapping.out  # 2> ../URL-experiments/out/pmo.err
#    2>&1

# eval_freq 4800