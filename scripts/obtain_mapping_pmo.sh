################### Obtain Class Mapping ###################
ulimit -n 50000
export META_DATASET_ROOT=../meta-dataset
export RECORDS=../datasets/tfrecords

NAME="C-net"
for TARGET in "train" "val" "test"
do
echo $TARGET
CUDA_VISIBLE_DEVICES=0 python obtain_mapping_pmo.py --model.name=$NAME --model.num_clusters 8 \
    --model.dir ../URL-experiments/saved_results/pmo-kmeans \
    --data.train ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower \
    --data.val ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower \
    --data.test ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower \
    --map.target $TARGET
done