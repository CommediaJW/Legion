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
    argparser.add_argument('--dataset-name', type=str, default="products")
    argparser.add_argument('--batch-size', type=int)
    args = argparser.parse_args()

    if args.dataset_name == "products":
        vertices_num = 2449029
        edges_num = 123718280
        features_dim = 100
        train_set_num = 196615
        valid_set_num = 39323
        test_set_num = 2213091
    elif args.dataset_name == "paper100M":
        vertices_num = 111059956
        edges_num = 1615685872
        features_dim = 128
        train_set_num = 1207179
        valid_set_num = 125265
        test_set_num = 214338
    else:
        print("invalid dataset path")
        exit

    out_dir = os.path.join(args.dataset_path, args.dataset_name)
    path_fn = os.path.join(
        out_dir, f"{args.dataset_name}_metis.graph.part{args.num_parts}")
    node_part_id = pd.read_csv(path_fn, header=None,
                               delimiter="\s+").to_numpy().astype(
                                   np.int32).flatten()
    train_nids = np.memmap(os.path.join(out_dir, "trainingset"),
                           dtype=np.int32,
                           mode="r",
                           shape=(train_set_num, ))
    print(node_part_id)
    print(train_nids)

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
    node_part_id.tofile(os.path.join(out_dir, f"partition{args.num_parts}"))
