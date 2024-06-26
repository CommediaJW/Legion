#include "storage_management.cuh"
#include "storage_management_impl.cuh"

void StorageManagement::EnableP2PAccess()
{
    int32_t device_count = -1;
    cudaGetDeviceCount(&device_count);
    for (int32_t i = 0; i < device_count; i++)
    {
        cudaSetDevice(i);
        cudaCheckError();
        for (int32_t j = 0; j < device_count; j++)
        {
            if (j != i)
            {
                int32_t accessible = 0;
                cudaDeviceCanAccessPeer(&accessible, i, j);
                cudaCheckError();
                if (accessible)
                {
                    cudaDeviceEnablePeerAccess(j, 0);
                    cudaCheckError();
                }
            }
        }
    }
}

void StorageManagement::ConfigPartition(BuildInfo *info, int32_t partition_count)
{
    info->partition_count = partition_count;
}

void StorageManagement::ReadMetaFIle(BuildInfo *info)
{
    std::istringstream iss;
    std::string buff;
    std::ifstream Metafile("./meta_config");
    if (!Metafile.is_open())
    {
        std::cout << "unable to open meta config file"
                  << "\n";
    }
    getline(Metafile, buff);
    iss.clear();
    iss.str(buff);
    if (in_memory_mode_)
    {
        iss >> dataset_path_;
        std::cout << "Dataset path:       " << dataset_path_ << "\n";
        iss >> raw_batch_size_;
        std::cout << "Raw Batchsize:      " << raw_batch_size_ << "\n";
        info->raw_batch_size = raw_batch_size_;
        iss >> node_num_;
        std::cout << "Graph nodes num:    " << node_num_ << "\n";
        iss >> edge_num_;
        std::cout << "Graph edges num:    " << edge_num_ << "\n";
        iss >> float_feature_len_;
        std::cout << "Feature dim:        " << float_feature_len_ << "\n";
        iss >> training_set_num_;
        std::cout << "Training set num:   " << training_set_num_ << "\n";
        iss >> validation_set_num_;
        validation_set_num_ = 0;
        std::cout << "Validation set num: " << validation_set_num_ << "\n";
        iss >> testing_set_num_;
        testing_set_num_ = 0;
        std::cout << "Testing set num:    " << testing_set_num_ << "\n";
        iss >> cache_memory_;
        std::cout << "Cache memory:       " << cache_memory_ << "\n";
        iss >> epoch_;
        std::cout << "Train epoch:        " << epoch_ << "\n";
        info->epoch = epoch_;
    }
    else
    {
        iss >> dataset_path_;
        std::cout << "Dataset path:       " << dataset_path_ << "\n";
        iss >> raw_batch_size_;
        std::cout << "Raw Batchsize:      " << raw_batch_size_ << "\n";
        info->raw_batch_size = raw_batch_size_;
        iss >> node_num_;
        std::cout << "Graph nodes num:    " << node_num_ << "\n";
        iss >> edge_num_;
        std::cout << "Graph edges num:    " << edge_num_ << "\n";
        iss >> float_feature_len_;
        std::cout << "Feature dim:        " << float_feature_len_ << "\n";
        iss >> training_set_num_;
        std::cout << "Training set num:   " << training_set_num_ << "\n";
        iss >> validation_set_num_;
        std::cout << "Validation set num: " << validation_set_num_ << "\n";
        iss >> testing_set_num_;
        std::cout << "Testing set num:    " << testing_set_num_ << "\n";
        iss >> cache_memory_;
        std::cout << "Cache memory:       " << cache_memory_ << "\n";
        iss >> epoch_;
        std::cout << "Train epoch:        " << epoch_ << "\n";
        info->epoch = epoch_;
        iss >> partition_;
        std::cout << "Partition?:         " << partition_ << "\n";
        iss >> num_ssd_;
        std::cout << "SSD Num?:           " << num_ssd_ << "\n";
        iss >> num_queues_per_ssd_;
        std::cout << "Q/SSD    ?:         " << num_queues_per_ssd_ << "\n";
        iss >> cpu_cache_capacity_;
        std::cout << "CPU Cache Capacity: " << cpu_cache_capacity_ << "\n";
        iss >> gpu_cache_capacity_;
        std::cout << "GPU Cache Capacity: " << gpu_cache_capacity_ << "\n";
    }
}

void StorageManagement::LoadGraph(BuildInfo *info)
{

    int32_t node_num = node_num_;
    int64_t edge_num = edge_num_;
    info->total_edge_num = edge_num;
    info->cache_edge_num = cache_edge_num_;

    // uva
    cudaHostAlloc(&(info->csr_node_index), int64_t(int64_t(node_num + 1) * sizeof(int64_t)), cudaHostAllocMapped);
    cudaHostAlloc(&(info->csr_dst_node_ids), int64_t(int64_t(edge_num) * sizeof(int32_t)), cudaHostAllocMapped);
    std::string edge_src_path = dataset_path_ + "edge_src";
    std::string edge_dst_path = dataset_path_ + "edge_dst";

    mmap_indptr_read(edge_src_path, info->csr_node_index);
    mmap_indices_read(edge_dst_path, info->csr_dst_node_ids);
}

void StorageManagement::LoadFeature(BuildInfo *info)
{

    int32_t partition_count = info->partition_count;

    int32_t node_num = node_num_;
    int32_t nf = float_feature_len_;

    info->numElems = uint64_t(node_num) * nf;

    (info->training_set_ids).resize(partition_count);
    (info->training_labels).resize(partition_count);
    (info->validation_set_ids).resize(partition_count);
    (info->validation_labels).resize(partition_count);
    (info->testing_set_ids).resize(partition_count);
    (info->testing_labels).resize(partition_count);

    std::string training_path = dataset_path_ + "trainingset";
    std::string validation_path = dataset_path_ + "validationset";
    std::string testing_path = dataset_path_ + "testingset";
    std::string features_path = dataset_path_ + "features";
    std::string labels_path = dataset_path_ + "labels";

    std::string partition_path = dataset_path_ + "partition" + std::to_string(partition_count);

    std::vector<int32_t> training_ids;
    training_ids.resize(training_set_num_);
    std::vector<int32_t> validation_ids;
    validation_ids.resize(validation_set_num_);
    std::vector<int32_t> testing_ids;
    testing_ids.resize(testing_set_num_);
    std::vector<int32_t> all_labels;
    all_labels.resize(node_num);
    int32_t *partition_index = (int32_t *)malloc(int64_t(node_num) * sizeof(int32_t));
    float *host_float_feature;

    mmap_trainingset_read(training_path, training_ids);
    // mmap_trainingset_read(validation_path, validation_ids);
    // mmap_trainingset_read(testing_path, testing_ids);
    if (in_memory_mode_)
    {
        cudaHostAlloc(&host_float_feature, uint64_t(uint64_t(node_num) * nf) * sizeof(float), cudaHostAllocMapped);
        mmap_features_read(features_path, host_float_feature);
    }
    mmap_labels_read(labels_path, all_labels);

    int32_t fdret = mmap_partition_read(partition_path, partition_index);
    if (fdret >= 0)
    {
        std::cout << "Use partition file: " << partition_path << std::endl;
    }
    std::cout << "Finish Reading All Files\n";

    // partition nodes
    int trainingset_count = 0;
    for (int32_t i = 0; i < training_set_num_; i += 1)
    {
        int32_t tid = training_ids[i];
        int32_t part_id;
        if (fdret >= 0)
        {
            part_id = partition_index[tid];
        }
        else
        {
            part_id = tid % partition_count;
        }
        if (part_id < partition_count)
        {
            (info->training_set_ids[part_id]).push_back(tid);
            trainingset_count++;
        }
    }
    for (int32_t i = 0; i < validation_set_num_; i++)
    {
        int32_t tid = validation_ids[i];
        int32_t part_id = tid % partition_count;

        if (part_id < partition_count)
        {
            (info->validation_set_ids[part_id]).push_back(tid);
        }
    }

    for (int32_t i = 0; i < testing_set_num_; i++)
    {
        int32_t tid = testing_ids[i];
        int32_t part_id = tid % partition_count;

        if (part_id < partition_count)
        {
            (info->testing_set_ids[part_id]).push_back(tid);
        }
    }
    free(partition_index);

    // partition labels
    for (int32_t part_id = 0; part_id < partition_count; part_id++)
    {
        for (int32_t i = 0; i < info->training_set_ids[part_id].size(); i++)
        {
            int32_t ts_label = all_labels[info->training_set_ids[part_id][i]];
            info->training_labels[part_id].push_back(ts_label);
        }
        info->training_set_num.push_back(info->training_set_ids[part_id].size());
    }
    for (int32_t part_id = 0; part_id < partition_count; part_id++)
    {
        for (int32_t i = 0; i < info->validation_set_ids[part_id].size(); i++)
        {
            int32_t ts_label = all_labels[info->validation_set_ids[part_id][i]];
            info->validation_labels[part_id].push_back(ts_label);
        }
        info->validation_set_num.push_back(info->validation_set_ids[part_id].size());
    }
    for (int32_t part_id = 0; part_id < partition_count; part_id++)
    {
        for (int32_t i = 0; i < info->testing_set_ids[part_id].size(); i++)
        {
            int32_t ts_label = all_labels[info->testing_set_ids[part_id][i]];
            info->testing_labels[part_id].push_back(ts_label);
        }
        info->testing_set_num.push_back(info->testing_set_ids[part_id].size());
    }

    for (int32_t i = 0; i < partition_count; i++)
    {
        std::cout << "Rank " << i << " train set count " << info->training_set_num[i] << "\n";
        std::cout << "Rank " << i << " valid set count " << info->validation_set_num[i] << "\n";
        std::cout << "Rank " << i << " test set count " << info->testing_set_num[i] << "\n";
    }

    info->host_float_feature = host_float_feature;
    info->float_feature_len = float_feature_len_;
    info->total_num_nodes = node_num_;
}

void StorageManagement::Initialze(int32_t partition_count, int32_t in_memory_mode)
{

    in_memory_mode_ = in_memory_mode;

    BuildInfo *info = new BuildInfo();

    EnableP2PAccess();

    ConfigPartition(info, partition_count);

    ReadMetaFIle(info);

    LoadGraph(info);

    LoadFeature(info);

    env_ = NewIPCEnv(partition_count);
    env_->Coordinate(info);

    feature_ = NewCompleteFeatureStorage();
    feature_->Build(info, in_memory_mode_);

    graph_ = NewCompleteGraphStorage();
    graph_->Build(info);

    cudaCheckError();

    cache_ = new UnifiedCache();

    int32_t train_step = env_->GetTrainStep();

    // cudaSetDevice(0);
    cache_->Initialize(cache_memory_, float_feature_len_, train_step, partition_count, cpu_cache_capacity_, gpu_cache_capacity_);
    // cudaSetDevice(0);
    std::cout << "Storage Initialized\n";
}

GraphStorage *StorageManagement::GetGraph()
{
    return graph_;
}

FeatureStorage *StorageManagement::GetFeature()
{
    return feature_;
}

UnifiedCache *StorageManagement::GetCache()
{
    return cache_;
}

IPCEnv *StorageManagement::GetIPCEnv()
{
    return env_;
}
