import os
import pandas as pd 
import numpy as np 
import torch
import torchvision.transforms as transforms
from triplet_loss import TripletMarginLoss
# from model import EmbeddingNet
from triplet_loss_sampler import PKSampler

from torch.optim import Adam  
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Subset
from torchvision.datasets import FashionMNIST 
 
from data.celebA_dataset import CelebADataset
from variable_width_resnet import resnet50vw, resnet18vw, resnet10vw


""" 
Learns 2-d embeddings for each data point via contrastive triplet loss for the task of predicting the group of each data point
"""

# TODO: figure out how to change embedding size // this is just the number of classes we have...
def train_epoch(model, optimizer, criterion, data_loader, device, epoch, print_freq):
    model.train()
    running_loss = 0
    running_frac_pos_triplets = 0
    # print(f"TEST DATA_LOADER{data_loader}")   
    for i, data in enumerate(data_loader):
        optimizer.zero_grad() 
        samples, targets = data[0].to(device), data[1].to(device)
        # print(f"SAMPLES VARIABLE: {samples}")
        # print(f"TARGETS VARIABLE: {targets}")
 
        embeddings = model(samples) 
        # current embeddings size is 64 x 2...

        loss, frac_pos_triplets = criterion(embeddings, targets)
        loss.backward()
        optimizer.step() 

        running_loss += loss.item() 
        running_frac_pos_triplets += float(frac_pos_triplets)

        if i % print_freq == print_freq - 1:
            i += 1
            avg_loss = running_loss / print_freq
            avg_trip = 100.0 * running_frac_pos_triplets / print_freq
            print(f"[{epoch:d}, {i:d}] | loss: {avg_loss:.4f} | % avg hard triplets: {avg_trip:.2f}%")
            running_loss = 0
            running_frac_pos_triplets = 0


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
def evaluate(model, loader, device, train_targets):
    model.eval()
    embeds, labels = [], []
    dists, targets = None, None

    groups = [] 
 
    # figured out that group labels contained in data[2]
    for data in loader:
        samples, _labels, _groups = data[0].to(device), data[1], data[2]
        out = model(samples) 
        embeds.append(out) 
        labels.append(_labels) 
        groups.append(_groups)

    embeds = torch.cat(embeds, dim=0)
    labels = torch.cat(labels, dim=0)
    groups = torch.cat(groups, dim=0)

    dists = torch.cdist(embeds, embeds)

    labels = labels.unsqueeze(0)
    targets = labels == labels.t()

    groups = groups.unsqueeze(0)
    group_targets = groups == groups.t()

    mask = torch.ones(dists.size()).triu() - torch.eye(dists.size(0))
    dists = dists[mask == 1] # keep only upper triangle of dists matrix since it was a symmetric matrix w/ 0 diag
    targets = targets[mask == 1]

    group_targets = group_targets[mask == 1]

    # print(f"TEST DISTS VARIABLE: {dists}")
    # print(f"TEST TARGETS VARIABLE: {targets}")



    # threshold, accuracy = find_best_threshold(dists, targets, device)
    threshold, accuracy = find_best_threshold(dists, group_targets, device)


    print(f"accuracy: {accuracy:.3f}%, threshold: {threshold:.2f}")


def save(model, epoch, save_dir, file_name):
    file_name = "epoch_" + str(epoch) + "__" + file_name
    save_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), save_path)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    p = args.labels_per_batch
    k = args.samples_per_label 
    batch_size = p * k

    model = resnet10vw(32, None, num_classes=2)
    if args.resume:
        model.load_state_dict(torch.load(args.resume))

    model.to(device)

    criterion = TripletMarginLoss(margin=args.margin, mining="batch_hard")
    optimizer = Adam(model.parameters(), lr=args.lr)

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize((224, 224)),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
        ]
    )

    print(f"LOADING CELEBA DATASET")   
    # load full dataset using CelebADataset
    full_dataset = CelebADataset(root_dir="/global/scratch/users/arinchang/celebA_dataset",
        target_name='Blond_Hair',
        confounder_names=['Male'],
        model_type='resnet10vw',
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
    
    #construct targets list for celebA by mapping data points to group number (0-3) using their male attribute and class 
    # 0: male 0 blonde 0, 1: male 0 blonde 1, 2: male 1 blonde 0, 3: male 1 blonde 1
    targets = []
    train_map = split_array == split_dict['train']
    train_indices = np.where(train_map)[0]
    for idx in train_indices:
        if full_dataset.y_array[idx] == 0 and full_dataset.confounder_array[idx] == 0:
            targets.append(0)
        elif full_dataset.y_array[idx] == 1 and full_dataset.confounder_array[idx] == 0:
            targets.append(1)
        elif full_dataset.y_array[idx] == 0 and full_dataset.confounder_array[idx] == 1:
            targets.append(2)
        elif full_dataset.y_array[idx] == 1 and full_dataset.confounder_array[idx] == 1:
            targets.append(3) 
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=PKSampler(targets, p, k), num_workers=args.workers
    )
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers)

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        evaluate(model, test_loader, device)
        return

    for epoch in range(1, args.epochs + 1):
        print("Training...")
        train_epoch(model, optimizer, criterion, train_loader, device, epoch, args.print_freq)

        print("Evaluating...")
        evaluate(model, test_loader, device, targets)

        print("Saving...")
        save(model, epoch, args.save_dir, "ckpt.pth")


def parse_args(): 
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Embedding Learning")

    parser.add_argument("--dataset-dir", default="/global/scratch/users/arinchang/celebA_dataset", type=str, help="celebA dataset directory path")
    parser.add_argument(
        "-p", "--labels-per-batch", default=4, type=int, help="Number of unique labels/classes per batch"
    )
    parser.add_argument("-k", "--samples-per-label", default=16, type=int, help="Number of samples per label in a batch")
    parser.add_argument("--eval-batch-size", default=512, type=int, help="batch size for evaluation")
    parser.add_argument("--epochs", default=10, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--lr", default=0.0001, type=float, help="initial learning rate")
    parser.add_argument("--margin", default=0.2, type=float, help="Triplet loss margin")
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--save-dir", default=".", type=str, help="Model save directory")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
