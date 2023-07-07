################### Training URL Model ###################
ulimit -n 50000
export META_DATASET_ROOT=../meta-dataset
export RECORDS=../datasets/tfrecords

CUDA_VISIBLE_DEVICES=0 python train_net_url.py --model.name=url_repro --model.dir ../URL-experiments/saved_results/sdl --out.dir ../URL-experiments/saved_results/url --data.train ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower --data.val ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower --train.learning_rate=3e-2 --train.max_iter=240000 --train.cosine_anneal_freq=48000 --train.eval_freq=48000