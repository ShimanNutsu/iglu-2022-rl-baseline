docker run --shm-size 20G --env WANDB_API_KEY=e5ac79d62944a4e1910f83da82ae92c37b09ecdf --env CUDA_VISIBLE_DEVICES=0 --rm -it --gpus all -v $(pwd):/code -w /code iglu-fast-baseline python main.py
