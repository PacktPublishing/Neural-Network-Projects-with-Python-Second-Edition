import argparse
import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2

import data_captcha
import models_captcha

BATCH_SIZE = 256
MAX_EPOCHES = 300
LEARNING_RATE = 1e-4
LOADER_WORKERS = 4


if __name__ == '__main__':
    cv2.setNumThreads(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Random seed",
                        default=113)
    parser.add_argument("-n", "--name", required=True,
                        help="Name of the run")
    parser.add_argument("-g", "--gpu", default="cpu",
                        help="GPU device, default=cpu")
    args = parser.parse_args()
    device = torch.device(args.gpu)
    torch.manual_seed(args.seed)

    dataset = data_captcha.OCRDataset(size_mul=10)
    train_loader = DataLoader(dataset, BATCH_SIZE, shuffle=True,
                              num_workers=LOADER_WORKERS)

    # model
    model = models_captcha.OCRNetwork(len(data_captcha.CAPTCHA_CHARS)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss = nn.CrossEntropyLoss()
    print(model)

    try:
        best_loss = None
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
                print(f"Epoch {epoch}: train_loss={train_loss:.5f}")
                writer.add_scalar("loss", train_loss, epoch)
                if best_loss is None or best_loss > train_loss:
                    print(f"Model improved, saving")
                    torch.save(model.state_dict(), args.name + "_best.model")
                    best_loss = train_loss
    except KeyboardInterrupt:
        print("Interrupting...")
