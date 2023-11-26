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

def load_celebA_dataset():
    # TODO: move this function into some utils file in /data 
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
    model.to(device)

    train_dataset, val_dataset, test_dataset = load_celebA_dataset()

    loader = DataLoader(train_dataset, batch_size=512) # get training dataset

    embeds, labels = [], []
    dists, targets = None, None
    groups = [] 

    # Remove the classification head (top) of the model
    model = torch.nn.Sequential(*(list(model.children())[:-1]))

    # Set the model to evaluation mode
    model.eval()
    
    # compute embeddings for all training data points 
    for data in loader:
        samples, _labels, _groups = data[0].to(device), data[1], data[2]
        with torch.no_grad():
            out = model(samples) 
        embeds.append(out) 
        labels.append(_labels) 
        groups.append(_groups)

    embeddings = torch.cat(embeds, dim=0) # embeddings are of size [162770, 2048, 1, 1]
    labels = torch.cat(labels, dim=0)
    groups = torch.cat(groups, dim=0)

    # cast embeddings from 4d tensor to 2d numpy array
    embeddings = embeddings.resize(162770, 2048)
    embeddings = embeddings.detach().numpy()

    """
    use t-SNE to visualize embeddings
    """
    # Replace this with your actual embeddings
    # Assume embeddings is a 2D NumPy array where each row represents an embedding
    # For example, embeddings = np.random.rand(100, 50) for 100 embeddings of size 50

    # Perform t-SNE on the embeddings
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)

    # Plot the t-SNE embeddings
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], marker='.')
    plt.title('t-SNE Visualization of Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()


if __name__ == "__main__":
    main()
