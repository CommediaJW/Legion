python3 legion_server.py --dataset_path /data/legion_dataset --dataset_name products --train_batch_size 1000 --gpu_number 2 --epoch 5 --cache_memory 38000000 --usenvlink 0 --fanout 12,12,12
rm /dev/shm/*
python3 legion_server.py --dataset_path /data/legion_dataset --dataset_name products --train_batch_size 1000 --gpu_number 2 --epoch 5 --cache_memory 38000000 --usenvlink 0 --fanout 12,12,12
rm /dev/shm/*