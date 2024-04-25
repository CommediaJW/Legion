import torch
import argparse
from preprocess_ogbn import save_int32, save_int64, save_float32
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/data')
    parser.add_argument('--dataset', type=str, default='friendster')
    parser.add_argument('--out-dir', type=str, default='/data/legion_dataset')
    args = parser.parse_args()

    print("load {}...".format(args.dataset))
    args.out_dir = os.path.join(args.out_dir, args.dataset)
    metadata = torch.load(os.path.join(args.root, "metadata.pt"))
    labels = torch.load(os.path.join(args.root, "labels.pt"))
    indptr = torch.load(os.path.join(args.root, "indptr.pt"))
    indices = torch.load(os.path.join(args.root, "indices.pt"))
    train_nid = torch.load(os.path.join(args.root, "train_nid.pt"))

    print("save data...")
    save_int32(labels, os.path.join(args.out_dir, "labels"))
    save_int64(indptr, os.path.join(args.out_dir, "edge_src"))
    save_int32(indices, os.path.join(args.out_dir, "edge_dst"))
    save_int32(train_nid, os.path.join(args.out_dir, "trainingset"))

    try:
        print("try load valid...")
        valid_nid = torch.load(os.path.join(args.root, "valid_nid.pt"))
        valid = True
    except:
        valid = False
    if valid:
        print("save valid...")
        save_int32(valid_nid, os.path.join(args.out_dir, "validationset"))

    try:
        print("try load test...")
        test_nid = torch.load(os.path.join(args.root, "test_nid.pt"))
        test = True
    except:
        test = False
    if test:
        print("save test...")
        save_int32(test_nid, os.path.join(args.out_dir, "testset"))
