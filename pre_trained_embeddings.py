import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from PIL import Image
import os
import pandas as pd 
import numpy as np 
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
from data.celebA_dataset import CelebADataset

from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt 

def find_best_threshold(dists, targets, device):
    best_thresh = 0.01
    best_correct = 0
    for thresh in torch.arange(0.0, 1.51, 0.01):
        predictions = dists <= thresh.to(device)
        correct = torch.sum(predictions == targets.to(device)).item()
        if correct > best_correct:
            best_thresh = thresh
            best_correct = correct

    accuracy = 100.0 * best_correct / dists.size(0)

    return best_thresh, accuracy

@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    embeds, labels = [], []
    dists, targets = None, None

    groups = [] 
 
    for data in loader:
        samples, _labels, _groups = data[0].to(device), data[1], data[2]    # figured out that group labels contained in data[2]
        out = model(samples) 
        embeds.append(out) 
        labels.append(_labels) 
        groups.append(_groups)
    
    # print(f"embeds size before cat {len(embeds)}")

    embeds = torch.cat(embeds, dim=0)
    labels = torch.cat(labels, dim=0)
    groups = torch.cat(groups, dim=0)

    embeds = embeds.resize(embeds.size()[0], embeds.size()[1])
    print(f"embeds size {embeds.size()}")

    dists = torch.cdist(embeds, embeds)

    labels = labels.unsqueeze(0)
    targets = labels == labels.t()

    groups = groups.unsqueeze(0)
    group_targets = groups == groups.t()

    print(f"dists size {dists.size()}")
    mask = torch.ones(dists.size()).triu() - torch.eye(dists.size(0))
    dists = dists[mask == 1] # keep only upper triangle of dists matrix since it was a symmetric matrix w/ 0 diag
    targets = targets[mask == 1]

    group_targets = group_targets[mask == 1]

    # threshold, accuracy = find_best_threshold(dists, targets, device)
    threshold, accuracy = find_best_threshold(dists, group_targets, device)


    print(f"accuracy: {accuracy:.3f}%, threshold: {threshold:.2f}")


# @torch.inference_mode()
# def evaluate(model, loader, device):
#     model.eval()
#     embeds, labels = [], []
#     dists, targets = None, None

#     groups = [] 

#     for data in loader:
#         samples, _labels, _groups = data[0].to(device), data[1], data[2]
#         out = model(samples) 
#         embeds.append(out) 
#         labels.append(_labels) 
#         groups.append(_groups)

#     embeds = torch.cat(embeds, dim=0)
#     labels = torch.cat(labels, dim=0)
#     groups = torch.cat(groups, dim=0)

#     dists = torch.cdist(embeds, embeds)

#     labels = labels.unsqueeze(0)
#     targets = labels == labels.t()

#     groups = groups.unsqueeze(0)
#     group_targets = groups == groups.t()

#     test_allocate = torch.ones(100, 100)
#     print(f"test_ones {test_allocate}")

#     print(f"dists.size(): {dists.size()}")
#     mask = torch.ones(dists.size()).triu() - torch.eye(dists.size(0))
#     dists = dists[mask == 1] # keep only upper triangle of dists matrix since it was a symmetric matrix w/ 0 diag
#     targets = targets[mask == 1]

#     group_targets = group_targets[mask == 1]

#     threshold, accuracy = find_best_threshold(dists, group_targets, device)


#     print(f"accuracy: {accuracy:.3f}%, threshold: {threshold:.2f}")


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

# Function to load and preprocess images
def load_and_preprocess_images(image_paths):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        images.append(img)

    return torch.stack(images)

# Function to generate embeddings using a pretrained model
def generate_embeddings(model, images):
    # Remove the classification head (top) of the model
    model = torch.nn.Sequential(*(list(model.children())[:-1]))

    # Set the model to evaluation mode
    model.eval() 

    # Generate embeddings
    with torch.no_grad():
        embeddings = model(images)

    return embeddings.squeeze()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")

    model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2") # load pretrained weights on imagenet1k
    # model = torch.hub.load("pytorch/vision", "resnet18", weights="IMAGENET1K_V1") # load pretrained weights on imagenet1k


    # Remove the classification head (top) of the model
    model = torch.nn.Sequential(*(list(model.children())[:-1]))

    model.to(device)

    train_dataset, val_dataset, test_dataset = load_celebA_dataset()

    # loader = DataLoader(train_dataset, batch_size=512) # get training dataset
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    print("Evaluating...")
    evaluate(model, test_loader, device)



    # embeds, labels = [], []
    # dists, targets = None, None
    # groups = [] 

    
    # Set the model to evaluation mode
    # model.eval()
    
    # compute embeddings for all training data points 
    # for data in loader:
    #     samples, _labels, _groups = data[0].to(device), data[1], data[2]
    #     with torch.inference_mode():
    #         out = model(samples) 
    #     embeds.append(out) 
    #     labels.append(_labels) 
    #     groups.append(_groups)

    # embeddings = torch.cat(embeds, dim=0) # embeddings are of size [162770, 2048, 1, 1]
    # labels = torch.cat(labels, dim=0)
    # groups = torch.cat(groups, dim=0)

    # evaluate the embeddings 
    # dists = torch.cdist(embeddings, embeddings)

    # labels = labels.unsqueeze(0)
    # targets = labels == labels.t()

    # groups = groups.unsqueeze(0)
    # group_targets = groups == groups.t()

    # mask = torch.ones(dists.size()).triu() - torch.eye(dists.size(0))
    # dists = dists[mask == 1] # keep only upper triangle of dists matrix since it was a symmetric matrix w/ 0 diag
    # targets = targets[mask == 1]
    # group_targets = group_targets[mask == 1]

    # threshold, accuracy = find_best_threshold(dists, group_targets, device)


    # print(f"accuracy: {accuracy:.3f}%, threshold: {threshold:.2f}")




    # """
    # use t-SNE to visualize embeddings
    # """

    # # TODO have different color markers for different groups embeddings in visualization 

    # # Perform t-SNE on the embeddings
    # tsne = TSNE(n_components=2, random_state=42)
    # embeddings_tsne = tsne.fit_transform(embeddings)

    # # Plot the t-SNE embeddings
    # plt.figure(figsize=(10, 8))
    # plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], marker='.')
    # plt.title('t-SNE Visualization of Embeddings')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    # plt.show()


if __name__ == "__main__":
    main()
