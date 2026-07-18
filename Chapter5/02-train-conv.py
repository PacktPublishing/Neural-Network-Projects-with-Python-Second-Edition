import argparse
import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import albumentations as A
import cv2

import data

BATCH_SIZE = 256
MAX_EPOCHES = 300
MAX_EPOCHES_AUG = 600
LEARNING_RATE = 1e-4
LEARNING_RATE_AUG = 5e-4
LOADER_WORKERS = 4


class Network(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


@torch.no_grad()
def validate(model: Network, dataset: data.Dataset,
             class_weights: torch.Tensor,
             device: torch.device) -> float:
    loader = DataLoader(dataset, BATCH_SIZE, shuffle=True,
                        num_workers=LOADER_WORKERS)
    loss = nn.CrossEntropyLoss(weight=class_weights)
    losses = []
    for batch_x, batch_y in tqdm.tqdm(loader, desc="Validation"):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        out_t = model(batch_x)
        loss_t = loss(out_t, batch_y)
        losses.append(loss_t.detach().item())
    return np.mean(losses)


if __name__ == '__main__':
    cv2.setNumThreads(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Random seed",
                        default=data.TRAIN_TEST_DEFAULT_SEED)
    parser.add_argument("-n", "--name", required=True,
                        help="Name of the run")
    parser.add_argument("-g", "--gpu", default="cpu",
                        help="GPU device, default=cpu")
    parser.add_argument("--augment", default=False, action='store_true',
                        help="Apply augmentation to the images")
    args = parser.parse_args()
    device = torch.device(args.gpu)
    torch.manual_seed(args.seed)

    # optional augmentation
    augment = None
    learning_rate = LEARNING_RATE
    if args.augment:
        augment = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussianBlur(p=0.2),
            A.Rotate(angle_range=(-30, 30), p=0.5),
        ])
        learning_rate = LEARNING_RATE_AUG

    # data loading
    images_path = data.get_images_path()
    images_classes = data.get_classes(images_path)
    class_weights = data.get_class_weights(images_path, images_classes)
    samples = data.get_sorted_samples(images_path)
    train_samples, val_samples, test_samples = data.split_train_val_test(
        samples, args.seed, shuffle=True)
    train_dataset = data.ImagesDataset(
        images_classes, train_samples, augment=augment)
    val_dataset = data.ImagesDataset(images_classes, val_samples)
    test_dataset = data.ImagesDataset(images_classes, test_samples)
    print(f"Train={len(train_dataset)}, val={len(val_dataset)}, "
          f"test={len(val_dataset)}")
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True,
                              num_workers=LOADER_WORKERS)

    # model
    model = Network(len(images_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(f"Class weights: {class_weights}")
    class_weights_t = torch.tensor(class_weights).to(device)
    loss = nn.CrossEntropyLoss(weight=class_weights_t)
    print(model)
    best_val_loss = None
    epoches_limit = MAX_EPOCHES_AUG if args.augment else MAX_EPOCHES

    try:
        with SummaryWriter(log_dir=f"runs/{args.name}") as writer:
            for epoch in range(1, epoches_limit+1):
                losses = []
                for batch_x, batch_y in tqdm.tqdm(train_loader, desc="Training"):
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    optimizer.zero_grad()

                    out_t = model(batch_x)
                    loss_t = loss(out_t, batch_y)
                    loss_t.backward()
                    optimizer.step()
                    losses.append(loss_t.detach().item())
                train_loss = np.mean(losses)
                val_loss = validate(model, val_dataset, class_weights_t, device)
                print(f"Epoch {epoch}: train_loss={train_loss:.5f}, "
                      f"val_loss={val_loss:.5f}")
                writer.add_scalar("loss", train_loss, epoch)
                writer.add_scalar("loss-val", val_loss, epoch)
                if best_val_loss is None or best_val_loss > val_loss:
                    print(f"Model improved, saving")
                    torch.save(model.state_dict(), args.name + "_best.model")
                    best_val_loss = val_loss
    except KeyboardInterrupt:
        print("Interrupting, checking on test dataset...")
    finally:
        test_loss = validate(model, test_dataset, class_weights_t, device)
        print(f"Test loss={test_loss:.5f}")
