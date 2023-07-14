################### Test URL Model with PA ###################
ulimit -n 50000
export META_DATASET_ROOT=../meta-dataset
export RECORDS=../datasets/tfrecords

CUDA_VISIBLE_DEVICES=7 python test_extractor_pa.py --model.name=url_repro --model.dir ../URL-experiments/saved_results/url
