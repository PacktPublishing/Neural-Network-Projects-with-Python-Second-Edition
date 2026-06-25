import argparse
import tqdm
import pathlib
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import data

BATCH_CHUNKS = 5
MAX_EPOCHES = 200


class Network(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze()


@torch.no_grad()
def validate(model: Network, dataset: data.Dataset,
             device: torch.device) -> float:
    loader = DataLoader(dataset, BATCH_CHUNKS, shuffle=True,
                        collate_fn=data.collate_batches)
    loss = nn.MSELoss()
    losses = []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        out_t = model(batch_x)
        loss_t = loss(out_t, batch_y)
        losses.append(np.sqrt(loss_t.detach().item()))
    return np.mean(losses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=12,
        help="Random seed, default=12")
    parser.add_argument(
        "-d", "--data", default="data_v1.feather",
        help="Path to the training data"),
    parser.add_argument(
        "-n", "--name", required=True,
        help="Name of the run (used for tensorboard)")
    parser.add_argument(
        "-g", "--gpu", default="cpu",
        help="GPU device, default=cpu")
    args = parser.parse_args()

    device = torch.device(args.gpu)

    torch.manual_seed(args.seed)
    dataset = data.FeatherChunksDataset(pathlib.Path(args.data))
    train_dataset, val_dataset, test_dataset = \
        random_split(dataset, [0.8, 0.1, 0.1])
    print(f"Chunks train={len(train_dataset)}, "
          f"val={len(val_dataset)}, test={len(test_dataset)}")
    train_loader = DataLoader(
        train_dataset, BATCH_CHUNKS, shuffle=True,
        collate_fn=data.collate_batches)

    model = Network(len(dataset.columns)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss = nn.MSELoss()
    print(model)
    best_val_loss = None

    try:
        with SummaryWriter(log_dir=f"runs/{args.name}") as writer:
            for epoch in range(MAX_EPOCHES):
                losses = []
                for batch_x, batch_y in tqdm.tqdm(
                        train_loader, total=len(train_dataset)//BATCH_CHUNKS):
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    optimizer.zero_grad()
                    out_t = model(batch_x)

                    loss_t = loss(out_t, batch_y)
                    loss_t.backward()
                    optimizer.step()
                    losses.append(np.sqrt(loss_t.detach().item()))
                train_loss = np.mean(losses)
                val_loss = validate(model, val_dataset, device)
                print(f"Epoch {epoch+1}: train_rmse={train_loss:.5f}, "
                      f"val_rmse={val_loss:.5f}")
                writer.add_scalar("loss-rmse", train_loss, epoch)
                writer.add_scalar("loss-rmse-val", val_loss, epoch)
                if best_val_loss is None or best_val_loss > val_loss:
                    print("Model improved, saving")
                    torch.save(model.state_dict(), args.name + "_best.model")
                    best_val_loss = val_loss
    except KeyboardInterrupt:
        print("Interrupted, checking the model on test dataset...")
    finally:
        test_loss = validate(model, test_dataset, device)
        print(f"Test loss={test_loss:.5f}")
