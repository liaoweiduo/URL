################### Test URL Model with PA ###################
ulimit -n 50000
export META_DATASET_ROOT=../meta-dataset
export RECORDS=../datasets/tfrecords

CUDA_VISIBLE_DEVICES=0 python test_extractor_pa.py --model.name=pmo --model.dir ../URL-experiments/saved_results/pmo-tcph--slr0_001-flr0_001 \
--model.num_clusters 10 --model.backbone resnet18_moe --model.pretrained --source ../URL-experiments/saved_results/url \
--source_moe ../URL-experiments/saved_results/sdl
