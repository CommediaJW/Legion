import numpy as np
from ogb import nodeproppred
import argparse
import os
import torch
from tqdm import tqdm

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
    train_nid = splitted_idx['train']
    g, _ = data[0]
    dst, src = g.adj_tensors('coo')

    src, sort_idcs = torch.sort(src)
    dst = dst[sort_idcs]

    train_mask = torch.zeros((g.num_nodes(), ), dtype=torch.bool)
    train_mask[train_nid] = True

    # parmetis cannot handle duplicated edges and self-loops. We should remove them
    # in the preprocessing.
    # Remove self-loops
    self_loop_idx = src == dst
    not_self_loop_idx = ~self_loop_idx
    dst = dst[not_self_loop_idx]
    src = src[not_self_loop_idx]
    edge_ids = (src * g.num_nodes() + dst).numpy()
    uniq_ids, idx = np.unique(edge_ids, return_index=True)
    duplicate_idx = np.setdiff1d(np.arange(len(edge_ids)), idx)
    duplicate_src = src[duplicate_idx]
    duplicate_dst = dst[duplicate_idx]
    src = src[idx]
    dst = dst[idx]

    print("There are {} edges, remove {} self-loops and {} duplicated edges".
          format(g.num_edges(),
                 self_loop_idx.nonzero().numel(), len(duplicate_src)))

    if graph_name == "paper100M":
        ud_src = torch.cat([src, dst])
        ud_dst = torch.cat([dst, src])
        src, sort_idcs = torch.sort(ud_src)
        dst = ud_dst[sort_idcs]
        print("Convert to bidirected graph, there are {} edges".format(
            src.shape[0]))

    indptr = torch.searchsorted(src, torch.arange(g.num_nodes() + 1))

    filename = os.path.join(args.out_dir, f"{graph_name}_metis.graph")
    f = open(filename, "w")
    hearder = "{} {} 010 2\n".format(g.num_nodes(), src.shape[0])
    f.write(hearder)

    with tqdm(total=g.num_nodes()) as pbar:
        processed = 0
        for nid in range(g.num_nodes()):
            if train_mask[nid].item() == True:
                f.write("1 0")
            else:
                f.write("0 1")
            begin = indptr[nid].item()
            end = indptr[nid + 1].item()
            neighbor_list = (dst[begin:end] + 1).tolist()
            neighbor_list = [str(neighbor) for neighbor in neighbor_list]
            line = " " + " ".join(neighbor_list) + "\n"
            f.write(line)

            processed += 1
            pbar.update(processed)

    f.close()
