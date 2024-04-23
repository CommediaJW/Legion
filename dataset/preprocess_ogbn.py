import numpy as np
from ogb import nodeproppred
import argparse
import os


def save_int32(tensor, path):
    data = tensor.int().numpy()
    fp = np.memmap(path, dtype='int32', mode='w+', shape=(data.shape[0], ))
    fp[:] = data[:]


def save_int64(tensor, path):
    data = tensor.long().numpy()
    fp = np.memmap(path, dtype='int64', mode='w+', shape=(data.shape[0], ))
    fp[:] = data[:]


def save_float32(tensor, path):
    data = tensor.float().numpy()
    fp = np.memmap(path,
                   dtype='float32',
                   mode='w+',
                   shape=(data.shape[0], data.shape[1]))
    fp[:] = data[:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/data')
    parser.add_argument('--dataset',
                        type=str,
                        default='ogbn-products',
                        choices=["ogbn-products", "ogbn-papers100M"])
    parser.add_argument('--out-dir', type=str, default='/data/legion_dataset')
    args = parser.parse_args()

    if args.dataset == "ogbn-products":
        graph_name = "products"
    elif args.dataset == "ogbn-papers100M":
        graph_name = "paper100M"
    args.out_dir = os.path.join(args.out_dir, graph_name)

    data = nodeproppred.DglNodePropPredDataset(name=args.dataset,
                                               root=args.root)
    splitted_idx = data.get_idx_split()
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx[
        'valid'], splitted_idx['test']
    g, labels = data[0]
    indptr, indices, _ = g.adj_tensors('csc')
    features = g.ndata.pop('feat')
    labels = np.squeeze(labels, 1)

    # save indptr
    save_int64(indptr, os.path.join(args.out_dir, "edge_src"))

    # save indices
    save_int32(indices, os.path.join(args.out_dir, "edge_dst"))
    print(indptr)
    print(indices)

    # save feature
    save_float32(features, os.path.join(args.out_dir, "features"))
    save_int32(labels, os.path.join(args.out_dir, "labels"))
    print(features)
    print(labels)
    print("Num class:", labels.max().item() + 1)

    # save nid
    save_int32(train_nid, os.path.join(args.out_dir, "trainingset"))
    save_int32(val_nid, os.path.join(args.out_dir, "validationset"))
    save_int32(test_nid, os.path.join(args.out_dir, "testingset"))
    print(train_nid.numel())
    print(val_nid.numel())
    print(test_nid.numel())

    # # xtrapulp_format
    # dst, src = g.adj_tensors('coo')
    # src, sort_idcs = torch.sort(src)
    # dst = dst[sort_idcs]
    # print(src)
    # print(dst)

    # src = src.reshape((1, src.shape[0]))
    # dst = dst.reshape(1, dst.shape[0])
    # xtrapulp_edges = torch.cat([src, dst], dim=0)
    # xtrapulp_edges = torch.transpose(xtrapulp_edges, 0, 1).flatten()
    # print(xtrapulp_edges)
    # save_int32(xtrapulp_edges,
    #            os.path.join(args.out_dir, f"{graph_name}_xtraformat"))
    # del xtrapulp_edges
