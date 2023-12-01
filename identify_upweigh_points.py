"""
- run a round of model inference to compute learned embeddings for all data points, and measure distance 
to the average embedding of minority group. keep track of the top k closest points
- form a list containing the weights for each data point to be passed into WeightedRandomSampler like 
weights = group_weights[self._group_array] in dro_dataset.py
"""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import os
import pandas as pd 
import numpy as np  
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
from data.celebA_dataset import CelebADataset 

# from dp_train import train


# TODO should probably make argument parser that takes in model type, etc

def load_celebA_dataset():
    full_dataset = CelebADataset(root_dir="/global/scratch/users/arinchang/celebA_dataset",
        target_name='Blond_Hair',
        confounder_names=['Male'],
        model_type='resnet50',
        augment_data=False)

    # split full dataset into train, val, test splits using torch.utils.data Subset. Based on get_splits function in 
    # confounder_dataset.py
    splits = ['train', 'val', 'test']
    split_df = pd.read_csv('/global/scratch/users/arinchang/celebA_dataset/data/list_eval_partition.csv')
    split_array = split_df['partition'].values
    split_dict = {
        'train': 0,
        'val': 1,
        'test': 2
    }
    subsets = {} 
    for split in splits:
        split_mask = split_array == split_dict[split]
        num_split = np.sum(split_mask)
        indices = np.where(split_mask)[0]
        subsets[split] = Subset(full_dataset, indices)

    train_dataset = subsets['train']
    val_dataset = subsets['val']
    test_dataset = subsets['test']
    print(f"LOADED CELEBA DATASET")
    return train_dataset, val_dataset, test_dataset

@torch.inference_mode()
def get_embeddings(model, loader, device):
    # Set the model to evaluation mode
    model.eval()

    embeds, labels = [], []
    dists, targets = None, None
    groups = [] 
    
    # compute embeddings for all training data points 
    for data in loader:
        samples, _labels, _groups = data[0].to(device), data[1], data[2]
        out = model(samples) 
        embeds.append(out) 
        labels.append(_labels) 
        groups.append(_groups)

    embeddings = torch.cat(embeds, dim=0) # embeddings are of size [162770, 2048, 1, 1]
    labels = torch.cat(labels, dim=0)
    groups = torch.cat(groups, dim=0)
    return embeddings, labels, groups 

def topk_embeddings_mask(non_min_embeds, avg_minority_embed, k, idx, num_points):
    """
    Return mask for k non-minority group training points that have embeddings closest to average minority embedding
    """

    # find distances between non-minority group embeddings to avg_minority_embed
    dists = torch.cdist(non_min_embeds, avg_minority_embed) 
    print(f"PRINT dists size {dists.size()}")

    topk_idx = torch.topk(dists, k, dim=0, largest=False)[1] # indices in dists of the topk closest embeddings to avg_minority_embed
    # print(f"PRINT topk_idx {topk_idx}")
    print(f"PRINT topk_idx {topk_idx.size()}")

    # now just index into idx using topk_idx 
    topk_idx_mapped = idx[topk_idx] # topk_idx_mapped contains the indices of non-minority points in original embeds array

    print(f"PRINT topk_idx_mapped {topk_idx_mapped.size()}")

    topk_mask = torch.zeros(num_points)
    topk_mask[topk_idx_mapped] = 1 
    return topk_mask


def compute_weights():
    """
    computes the new weights on the training data for WeightedRandomSampler
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")

    model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2") # load pretrained weights on imagenet1k
    model = torch.nn.Sequential(*(list(model.children())[:-1])) # remove classification head 
    model.to(device)

    train_dataset, val_dataset, test_dataset = load_celebA_dataset()

    loader = DataLoader(train_dataset, batch_size=512) # get training dataset
    # test_loader = DataLoader(test_dataset, batch_size=512) was using test_dataset to debug 

    embeds, labels, groups = get_embeddings(model, loader, device)
    print(f"Got embeddings")

    # get value of average embeddings for minority points 
    min_idx = torch.where(groups==3)[0] # group 3 is the minority group
    minority_embeds = embeds[min_idx]
    avg_minority_embed = torch.mean(minority_embeds, 0) 
    avg_minority_embed = avg_minority_embed.resize(1, 2048) # fix this hard coded 2048 in the future? 

    # idx will be our map from topk indices to non-minority group elements' original indices      
    idx = torch.where((groups==0) | (groups==1) | (groups==2))[0] # get indices of the non-minority group points
    non_min_embeds = embeds[idx]
    num_non_min = idx.size()[0]
    non_min_embeds = non_min_embeds.resize(num_non_min, 2048) 

    # move idx tensor to gpu
    idx = idx.to(device)

    num_points = 162770 # number of points we are computing embeddings for i.e. number of training points 
    k = num_points - num_non_min # roughly the number of minority points in the training data 
    print(f"PRINT test k, should be number of minority points in training data {k}")

    topk_mask = topk_embeddings_mask(non_min_embeds, avg_minority_embed, k, idx, num_points) # marks the locations of training points with topk closest embeddings as 1 
    print(f"PRINT topk_mask size {topk_mask.size()}")

    # check if the mask sums to k 
    check_mask_sum = torch.sum(topk_mask)
    print(f"PRINT check_mask_sum, should sum to k {check_mask_sum}")

    # group 1 has weight 117.3540, group 0 has weight 2.2724 
    new_groups = topk_mask.to(torch.long)  # group 1 is the points closest to the minority points - we want to upweight these, group 0 is everything else 
    new_group_weights = torch.tensor([2.2724, 117.3540])
    new_weights = new_group_weights[new_groups]
    return new_weights 

# TODO extensively test the idx->topk_mask mapping and check that the mask is made correctly 
# TODO fix this main 
def main():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f"DEVICE: {device}")

    # model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2") # load pretrained weights on imagenet1k
    # model = torch.nn.Sequential(*(list(model.children())[:-1])) # remove classification head 
    # model.to(device)

    # train_dataset, val_dataset, test_dataset = load_celebA_dataset()

    # # TODO first debug using test data points but need to use training dataset in the future
    # loader = DataLoader(train_dataset, batch_size=512) # get training dataset
    # test_loader = DataLoader(test_dataset, batch_size=512)

    # embeds, labels, groups = get_embeddings(model, test_loader, device)
    # print(f"Got embeddings")

    # # get value of average embeddings for minority points 
    # min_idx = torch.where(groups==2)[0] # TODO CHANGE GROUPS TO 3 WHEN WE USE TRAINING DATA SET 
    # minority_embeds = embeds[min_idx]
    # avg_minority_embed = torch.mean(minority_embeds, 0) 
    # avg_minority_embed = avg_minority_embed.resize(1, 2048)

    # # idx will be our map from topk indices to non-minority group elements' original indices      
    # idx = torch.where((groups==0) | (groups==1) | (groups==2))[0] # get indices of the non-minority group points
    # non_min_embeds = embeds[idx]
    # num_non_min = idx.size()[0]
    # non_min_embeds = non_min_embeds.resize(num_non_min, 2048)

    # # move idx tensor to gpu
    # idx = idx.to(device)

    # # TODO num_points should eventually be the number of minority points in our training data right?
    # k = 5
    # num_points = 19962 # number of points we are computing embeddings for. TODO will be number of training points = 162770
    # topk_mask = topk_embeddings_mask(non_min_embeds, avg_minority_embed, k, idx, num_points) # marks the locations of training points with topk closest embeddings as 1 
    # print(f"PRINT topk_mask size {topk_mask.size()}")

    # print(f"PRINT topk_mask {topk_mask}")

    # # check if the mask sums to k 
    # check_mask_sum = torch.sum(topk_mask)
    # print(f"PRINT check_mask_sum {check_mask_sum}")

    # # group 1 has weight 117.3540, group 0 has weight 2.2724 
    # new_groups = topk_mask.to(torch.long)  # group 1 is the points closest to the minority points - we want to upweight these, group 0 is everything else 
    # new_group_weights = torch.tensor([2.2724, 117.3540])
    # new_weights = new_group_weights[new_groups]

    # print(f"PRINT new_weights {new_weights}")
    weights = compute_weights()
    print(f"PRINT weights size {weights.size()}")
    print(f"PRINT weights {weights}")


if __name__ == "__main__":
    main()