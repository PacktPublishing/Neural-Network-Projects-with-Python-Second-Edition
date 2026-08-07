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
MAX_EPOCHES = 1000
LEARNING_RATE = 5e-4
LOADER_WORKERS = 4


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return self.output(dec)


@torch.no_grad()
def validate(model: Network, dataset: data.Dataset,
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
    model = Network().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
                train_loss = np.mean(losses)
                val_loss = validate(model, val_dataset, device)
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
