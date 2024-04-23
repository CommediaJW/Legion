import numpy as np
import argparse
import os
if __name__ == "__main__":

    argparser = argparse.ArgumentParser("Gen sets.")
    argparser.add_argument('--dataset_path', type=str, default='dataset')
    argparser.add_argument('--dataset_name', type=str, default="ukunion")
    args = argparser.parse_args()

    if args.dataset_name == "products":
        vertices_num = 2449029
        edges_num = 123718280
        features_dim = 100
        train_set_num = 196615
        valid_set_num = 39323
        test_set_num = 2213091
    elif args.dataset_name == "paper100m":
        vertices_num = 111059956
        edges_num = 1615685872
        features_dim = 128
        train_set_num = 11105995
        valid_set_num = 100000
        test_set_num = 100000
    elif args.dataset_name == "com-friendster":
        vertices_num = 65608366
        edges_num = 1806067135
        features_dim = 256
        train_set_num = 6560836
        valid_set_num = 100000
        test_set_num = 100000
    elif args.dataset_name == "ukunion":
        vertices_num = 133633040
        edges_num = 5507679822
        features_dim = 256
        train_set_num = 13363304
        valid_set_num = 100000
        test_set_num = 100000
    elif args.dataset_name == "uk2014":
        vertices_num = 787801471
        edges_num = 47284178505
        features_dim = 128
        train_set_num = 78780147
        valid_set_num = 100000
        test_set_num = 100000
    elif args.dataset_name == "clueweb":
        vertices_num = 955207488
        edges_num = 42574107469
        features_dim = 128
        train_set_num = 95520748
        valid_set_num = 100000
        test_set_num = 100000
    else:
        print("invalid dataset path")
        exit

path = os.path.join(args.dataset_path, args.dataset_name)
all_ids = np.arange(vertices_num)
np.random.shuffle(all_ids)

trainset = all_ids[:train_set_num]
validset = all_ids[train_set_num:train_set_num + valid_set_num]
testset = all_ids[train_set_num + valid_set_num:train_set_num + valid_set_num +
                  test_set_num]

trainset = trainset.astype(np.int32)
trainset.tofile(os.path.join(path, 'trainingset'))

validset = validset.astype(np.int32)
validset.tofile(os.path.join(path, 'validationset'))

testset = testset.astype(np.int32)
testset.tofile(os.path.join(path, 'testingset'))
