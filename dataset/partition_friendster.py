import argparse
import pandas as pd
import numpy as np
import torch
import os

if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Graph Partitioning.")
    argparser.add_argument('--num-parts', type=int, required=True)
    argparser.add_argument('--dataset-path',
                           type=str,
                           default='/data/legion_dataset')
    argparser.add_argument('--dataset-name', type=str, default="friendster")
    argparser.add_argument('--batch-size', type=int)
    args = argparser.parse_args()

    if args.dataset_name == "friendster":
        path = args.dataset_path + "/friendster/"
        vertices_num = 124836180
        edges_num = 1806067135
        features_dim = 256
        train_set_num = 1248361
        valid_set_num = 0
        test_set_num = 0
    else:
        raise NotImplementedError


    out_dir = os.path.join(args.dataset_path, args.dataset_name)
    train_nids = np.memmap(os.path.join(out_dir, "trainingset"),
                           dtype=np.int32,
                           mode="r",
                           shape=(train_set_num, ))
    node_part_id = np.memmap(os.path.join(out_dir, f"partition{args.num_parts}"), dtype='int32', mode='w+', shape=(vertices_num, ))
    node_part_id[:] = np.arange(vertices_num, dtype=np.int32) % args.num_parts

    if args.batch_size is not None:
        train_num = []
        for pid in range(args.num_parts):
            train_num.append(np.sum(node_part_id[train_nids] == pid))
        while max(train_num) - min(train_num) >= args.batch_size:
            print(train_num)
            max_pid = np.argmax(train_num)
            min_pid = np.argmin(train_num)
            change_num = (train_num[max_pid] - train_num[min_pid]) // 2
            max_train_nids = train_nids[node_part_id[train_nids] == max_pid]
            change_train_nids = max_train_nids[torch.randperm(
                max_train_nids.shape[0]).numpy()][:change_num]
            node_part_id[change_train_nids] = min_pid
            for pid in range(args.num_parts):
                train_num[pid] = np.sum(node_part_id[train_nids] == pid)
        print(train_num)
