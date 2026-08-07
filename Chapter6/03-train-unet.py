import argparse
import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import numpy as np
import albumentations as A
import cv2

import data

BATCH_SIZE = 128
MAX_EPOCHES = 1000
LEARNING_RATE = 5e-4
LOADER_WORKERS = 4
LR_SCHEDULER_GAMMA = 0.998


class Decoder(nn.Module):
    def __init__(self, in_size: int, skip_size: int, out_size: int):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_size, out_size, kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(out_size + skip_size, out_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up = self.upsample(x)

        # handles odd input dimensions
        if up.shape != skip.shape:
            up = F.interpolate(
                up, size=skip.shape[-2:], mode="bilinear",
                align_corners=False)
        both = torch.cat([skip, up], dim=1)
        return self.conv(both)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.enc2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.enc3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dec3 = Decoder(256, 128, 128)
        self.dec2 = Decoder(128, 64, 64)
        self.dec1 = Decoder(64, 32, 32)

        self.output = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.output(d1)


@torch.no_grad()
def validate(model: UNet, dataset: data.Dataset,
             device: torch.device) -> float:
    loader = DataLoader(dataset, BATCH_SIZE, shuffle=True,
                        num_workers=LOADER_WORKERS)
    loss = nn.MSELoss()
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
    args = parser.parse_args()
    device = torch.device(args.gpu)
    torch.manual_seed(args.seed)

    # noise pipeline
    noise_pipeline = A.Compose([
        A.GaussNoise(p=1.0),
    ])

    # data loading
    images_path = data.get_images_path()
    samples = data.get_sorted_samples(images_path)
    train_samples, val_samples, test_samples = data.split_train_val_test(
        samples, args.seed, shuffle=True)
    train_samples.extend(test_samples)
    train_dataset = data.DenoiseImagesDataset(train_samples, noise_pipeline)
    val_dataset = data.DenoiseImagesDataset(val_samples, noise_pipeline)
    print(f"Train={len(train_dataset)}, val={len(val_dataset)}")
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True,
                              num_workers=LOADER_WORKERS)

    # model
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, LR_SCHEDULER_GAMMA)
    loss = nn.MSELoss()
    print(model)
    best_val_loss = None

    try:
        with SummaryWriter(log_dir=f"runs/{args.name}") as writer:
            for epoch in range(1, MAX_EPOCHES+1):
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
                lr_scheduler.step()
                train_loss = np.mean(losses)
                val_loss = validate(model, val_dataset, device)
                print(f"Epoch {epoch}: train_loss={train_loss:.5f}, "
                      f"val_loss={val_loss:.5f}")
                writer.add_scalar("loss", train_loss, epoch)
                writer.add_scalar("loss-val", val_loss, epoch)
                writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], epoch)
                if best_val_loss is None or best_val_loss > val_loss:
                    print(f"Model improved, saving")
                    torch.save(model.state_dict(), args.name + "_best.model")
                    best_val_loss = val_loss
    except KeyboardInterrupt:
        print("Interrupting...")
