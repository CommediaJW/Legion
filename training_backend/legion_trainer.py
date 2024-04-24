import os
import sys
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as Func

import ipc_service
import dgl
import dgl.nn.pytorch as dglnn
from dgl.heterograph import DGLBlock
import time
import numpy as np
import torchmetrics

torch.set_printoptions(threshold=np.inf)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    if torch.cuda.is_available():
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
    else:
        dist.init_process_group('gloo', rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class SAGE(nn.Module):

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        for i in range(0, n_layers):
            in_dim = in_feats if i == 0 else n_hidden
            out_dim = n_classes if i == n_layers - 1 else n_hidden
            self.layers.append(dglnn.SAGEConv(in_dim, out_dim, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


class GAT(nn.Module):

    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 n_heads,
                 activation=Func.relu,
                 feat_dropout=0.6,
                 attn_dropout=0.6):
        assert len(n_heads) == n_layers
        assert n_heads[-1] == 1

        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_heads = n_heads

        self.layers = nn.ModuleList()
        for i in range(0, n_layers):
            in_dim = in_feats if i == 0 else n_hidden * n_heads[i - 1]
            out_dim = n_classes if i == n_layers - 1 else n_hidden
            layer_activation = None if i == n_layers - 1 else activation
            self.layers.append(
                dglnn.GATConv(in_dim,
                              out_dim,
                              n_heads[i],
                              feat_drop=feat_dropout,
                              attn_drop=attn_dropout,
                              activation=layer_activation,
                              allow_zero_in_degree=True))

    def forward(self, blocks, x):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i == self.n_layers - 1:
                h = h.mean(1)
            else:
                h = h.flatten(1)
        return h


def create_dgl_block(src, dst, num_src_nodes, num_dst_nodes):
    gidx = dgl.heterograph_index.create_unitgraph_from_coo(2,
                                                           num_src_nodes,
                                                           num_dst_nodes,
                                                           src,
                                                           dst,
                                                           'coo',
                                                           row_sorted=True)
    g = DGLBlock(gidx, (['_N'], ['_N']), ['_E'])

    return g


def train_one_step(args, model, optimizer, loss_fcn, device, feat_len, iter,
                   device_id):
    step_batch = ipc_service.get_next(feat_len)
    step_block_sizes = ipc_service.get_block_size()
    ids, features, labels = step_batch[0], step_batch[1], step_batch[2]
    blocks = []
    for l in range(args.hops_num):
        block_agg_src = step_batch[3 + l * 2]
        block_agg_dst = step_batch[4 + l * 2]
        block_src_num = step_block_sizes[l * 2]
        block_dst_num = step_block_sizes[1 + l * 2]
        blocks.append(
            create_dgl_block(block_agg_src, block_agg_dst, block_src_num,
                             block_dst_num))
    batch_pred = model(blocks, features)
    long_labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    loss = loss_fcn(batch_pred, long_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    ipc_service.synchronize()
    return 0


def valid_one_step(args, model, metric, device, feat_len):
    step_batch = ipc_service.get_next(feat_len)
    step_block_sizes = ipc_service.get_block_size()
    ids, features, labels = step_batch[0], step_batch[1], step_batch[2]
    blocks = []
    for l in range(args.hops_num):
        block_agg_src = step_batch[3 + l * 2]
        block_agg_dst = step_batch[4 + l * 2]
        block_src_num = step_block_sizes[l * 2]
        block_dst_num = step_block_sizes[1 + l * 2]
        blocks.append(
            create_dgl_block(block_agg_src, block_agg_dst, block_src_num,
                             block_dst_num))
    batch_pred = model(blocks, features)
    long_labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    batch_pred = torch.softmax(batch_pred, dim=1).to(device)
    acc = metric(batch_pred, long_labels)
    ipc_service.synchronize()
    return acc


def test_one_step(args, model, metric, device, feat_len):
    step_batch = ipc_service.get_next(feat_len)
    step_block_sizes = ipc_service.get_block_size()
    ids, features, labels = step_batch[0], step_batch[1], step_batch[2]
    blocks = []
    for l in range(args.hops_num):
        block_agg_src = step_batch[3 + l * 2]
        block_agg_dst = step_batch[4 + l * 2]
        block_src_num = step_block_sizes[l * 2]
        block_dst_num = step_block_sizes[1 + l * 2]
        blocks.append(
            create_dgl_block(block_agg_src, block_agg_dst, block_src_num,
                             block_dst_num))

    batch_pred = model(blocks, features)
    long_labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    batch_pred = torch.softmax(batch_pred, dim=1).to(device)
    acc = metric(batch_pred, long_labels)
    ipc_service.synchronize()
    return acc


def worker_process(rank, world_size, args):
    print(f"Running GNN Training on CUDA {rank}.")
    device_id = rank
    setup(rank, world_size)
    cuda_device = torch.device("cuda:{}".format(device_id))
    torch.cuda.set_device(cuda_device)
    ipc_service.initialize()
    train_steps, valid_steps, test_steps = ipc_service.get_steps()
    print("#Training iterations:", train_steps)
    print("#Valid iterations:", valid_steps)
    print("#Test iterations:", test_steps)

    feat_len = args.features_num

    if args.model == "sage":
        model = SAGE(in_feats=args.features_num,
                     n_hidden=args.hidden_dim,
                     n_classes=args.class_num,
                     n_layers=args.hops_num,
                     activation=Func.relu,
                     dropout=args.drop_rate).to(cuda_device)
    elif args.model == "gat":
        args.hidden_dim = int(args.hidden_dim / args.heads_num)
        heads = [args.heads_num for _ in range(args.hops_num - 1)]
        heads.append(1)
        model = GAT(in_feats=args.features_num,
                    n_hidden=args.hidden_dim,
                    n_classes=args.class_num,
                    n_layers=args.hops_num,
                    n_heads=heads,
                    activation=Func.relu,
                    feat_dropout=args.drop_rate,
                    attn_dropout=args.drop_rate).to(cuda_device)
    else:
        raise NotImplemented

    if dist.is_initialized():
        model = DDP(model, device_ids=[device_id])
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()

    epoch_num = args.epoch
    epoch_time_log = []
    iter_time_log = []
    for epoch in range(epoch_num):
        forward = 0
        start = time.time()
        epoch_time = 0
        for iter in range(train_steps):
            # print(f"Train {iter}")
            tic = time.time()
            train_loss = train_one_step(args, model, optimizer, loss_fcn,
                                        cuda_device, feat_len, iter, device_id)
            toc = time.time()
            iter_time_log.append(toc - tic)
            if device_id == 0:
                print('Iter {} Train Loss :{:.5f} Time: {:.3f} ms'.format(
                    iter, train_loss, (toc - tic) * 1000))
        epoch_time += time.time() - start

        model.eval()
        metric = torchmetrics.Accuracy('multiclass',
                                       num_classes=args.class_num)
        metric = metric.to(device_id)
        model.metric = metric
        with torch.no_grad():
            for iter in range(valid_steps):
                # print(f"Val {iter}")
                valid_one_step(args, model, metric, cuda_device, feat_len)
            acc_val = metric.compute()
        if device_id == 0:
            print("Epoch:{}, Cost:{} s, Val Acc: {}".format(
                epoch, epoch_time, acc_val))
        epoch_time_log.append(epoch_time)

    if device_id == 0:
        print("Avg iteration time: {:.3f} ms".format(
            np.mean(iter_time_log[5:]) * 1000))
        print("Ave epoch time: {:.3f} s".format(np.mean(epoch_time_log[2:])))

    model.eval()
    metric = torchmetrics.Accuracy('multiclass', num_classes=args.class_num)
    metric = metric.to(device_id)
    model.metric = metric
    with torch.no_grad():
        for iter in range(test_steps):
            test_one_step(args, model, metric, cuda_device, feat_len)
        acc = metric.compute()
    if device_id == 0:
        print("Accuracy on test data: {}".format(acc))
    metric.reset()

    ipc_service.finalize()
    cleanup()


def run_distribute(dist_fn, world_size, args):
    mp.spawn(dist_fn, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    cur_path = sys.path[0]
    argparser = argparse.ArgumentParser("Train GNN.")
    argparser.add_argument('--class_num', type=int, default=2)
    argparser.add_argument('--features_num', type=int, default=128)
    argparser.add_argument('--hidden_dim', type=int, default=32)
    argparser.add_argument('--heads_num', type=int, default=4)
    argparser.add_argument('--model',
                           type=str,
                           default="sage",
                           choices=["sage", "gat"])
    argparser.add_argument('--hops_num', type=int, default=3)
    argparser.add_argument('--drop_rate', type=float, default=0.2)
    argparser.add_argument('--learning_rate', type=float, default=0.003)
    argparser.add_argument('--epoch', type=int, default=2)
    argparser.add_argument('--gpu_number', type=int, default=2)
    args = argparser.parse_args()

    world_size = args.gpu_number
    print(args)
    run_distribute(worker_process, world_size, args)
