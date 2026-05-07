import os
import torch
import torch.nn as nn
import torch.optim as optim
import time 
import numpy as np
import random 
from torch.utils.data import DataLoader
from tqdm import tqdm
#from torch_ort import ORTModule

from datasets.fake_image_dataset import FakeImageDataset
from models.fake_image_model import build_model
from utils.augmentations import get_train_transforms, get_val_transforms
from utils.callbacks import EarlyStopping, SchedulerCallback
from torch.amp import autocast, GradScaler

scaler = GradScaler()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# for reproducibilty of the results setting the seed as : 
def set_seed(seed=45):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(45)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets and loaders
    train_ds = FakeImageDataset(args.train_dir, transform=get_train_transforms())
    val_ds   = FakeImageDataset(args.test_dir,  transform=get_val_transforms())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,pin_memory = True,num_workers=32,persistent_workers = True,prefetch_factor=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,pin_memory = True,num_workers=32,persistent_workers = True,prefetch_factor=4)
    # Model, optimizer, criterion
    # For Accelerated training using the onnx runtime to load the model via ORTModule
    # model = ORTModule(build_model(unfreeze_at_epoch=args.unfreeze_epoch)).to(device)
    model = build_model(unfreeze_at_epoch=args.unfreeze_epoch).to(device)
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer_type == 0 :
    # ====== 1 Using the AdaM optimizer ========== #
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        print("Using the Adam optimizer")
    elif args.optimizer_type == 1:
    # ====== 2 Using the AdamW optimizer ========== #
        optimizer = optim.AdamW(model.parameters(),eps = 1e-5,lr=args.learning_rate) 
        print("Using the AdamW optimizer")
    elif args.optimizer_type == 2:
    # ====== 3 Using the AdamW optimizer with amsgrad = True ========== #
        optimizer = optim.AdamW(model.parameters(),eps = 1e-5,lr=args.learning_rate,amsgrad=True) 
        print("Using the AdamW optimizer with amsgrad = True")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scheduler_cb = SchedulerCallback(scheduler)
    early_stopper = EarlyStopping(patience=args.patience, verbose=True)

    best_epoch = 0
    for epoch in range(args.epochs):
        model.train()
        if epoch == model.unfreeze_at:
            for p in model.base_model.features.parameters():
                p.requires_grad = True
            print(f"Unfroze backbone at epoch {epoch}")

        running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs,labels)
            # outputs = model(images)
            # loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.2%}, "
              f"Val Loss {val_loss:.4f}, Val Acc {val_acc:.2%}")

        # Callbacks
        scheduler_cb.step()
        # checking whether the path exits and of not create the directories reccursively 
        # os.makedirs(args.checkpoint_path, exist_ok=True)
        early_stopper(val_loss, model, args.checkpoint_path)
        if early_stopper.early_stop:
            print("Early stopping triggered")
            break
        
        if val_acc > train_acc:  # example checkpoint criterion
            best_epoch = epoch
            # best_validation_acc = val_acc
            torch.save(model.state_dict(), args.checkpoint_path)

    print(f"Training complete. Best epoch: {best_epoch}")

