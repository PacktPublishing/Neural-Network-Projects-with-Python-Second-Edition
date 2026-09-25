import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import numpy as np

import data

BATCH_SIZE = 128
MAX_EPOCHES = 2000
LEARNING_RATE = 1e-5


class Network(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze()


@torch.no_grad()
def validate(model: Network, dataset: TensorDataset) -> float:
    loader = DataLoader(dataset, BATCH_SIZE, shuffle=False)
    loss = nn.BCEWithLogitsLoss()
    losses = []
    for batch_x, batch_y in loader:
        out_t = model(batch_x)
        loss_t = loss(out_t, batch_y)
        losses.append(loss_t.detach().item())
    return np.mean(losses)


@torch.no_grad()
def test_classifier(
        name: str, model: Network, dataset: TensorDataset):
    loader = DataLoader(dataset, BATCH_SIZE, shuffle=False)
    pred_probs = []
    for batch_x, batch_y in loader:
        out_t = model(batch_x)
        probs_t = torch.sigmoid(out_t)
        pred_probs.extend(probs_t.detach().numpy())
    pred_probs = np.array(pred_probs)
    binary_preds = pred_probs > 0.5

    data.show_confusion_matrix(labels_test, binary_preds)
    out_prefix = f"nn-{name}"
    data.make_pr_plot(labels_test, pred_probs, out_prefix + "-pr.svg", baseline=0.5)
    data.make_roc_plot(labels_test, pred_probs, out_prefix + "-roc.svg")
    best_thr = data.make_f1_plot(labels_test, pred_probs, out_prefix + "-f1.svg")

    binary_preds = pred_probs > best_thr
    data.show_confusion_matrix(labels_test, binary_preds)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=data.DEFAULT_SEED,
        help="Random seed, default=" + str(data.DEFAULT_SEED))
    parser.add_argument(
        "-d", "--data", required=True,
        help="Prefix to the numpy arrays"),
    parser.add_argument(
        "-n", "--name", required=True,
        help="Name of the run (used for tensorboard)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    emb, labels = data.load_embeddings(args.data)
    print(emb.shape, labels.shape)

    emb_train, emb_test, labels_train, labels_test = train_test_split(
        emb, labels, random_state=args.seed, test_size=0.2)
    print(emb_train.shape)

    dataset_train = TensorDataset(
        torch.tensor(emb_train),
        torch.tensor(labels_train))
    dataset_test = TensorDataset(
        torch.tensor(emb_test),
        torch.tensor(labels_test))
    loader_train = DataLoader(
        dataset_train, BATCH_SIZE, shuffle=True)

    model = Network(emb.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss = nn.BCEWithLogitsLoss()
    print(model)
    best_val_loss = None
    best_model = Network(emb.shape[1])

    try:
        with SummaryWriter(log_dir=f"runs/{args.name}") as writer:
            for epoch in range(MAX_EPOCHES):
                losses = []
                for batch_x, batch_y in loader_train:
                    optimizer.zero_grad()
                    out_t = model(batch_x)

                    loss_t = loss(out_t, batch_y)
                    loss_t.backward()
                    optimizer.step()
                    losses.append(np.sqrt(loss_t.detach().item()))
                train_loss = np.mean(losses)
                val_loss = validate(model, dataset_test)
                print(f"Epoch {epoch+1}: train={train_loss:.5f}, "
                      f"val={val_loss:.5f}")
                writer.add_scalar("loss", train_loss, epoch)
                writer.add_scalar("loss-val", val_loss, epoch)
                if best_val_loss is None or best_val_loss > val_loss:
                    print("Model improved, saving")
                    best_model.load_state_dict(model.state_dict())
                    best_val_loss = val_loss
    except KeyboardInterrupt:
        print("Interrupted, exiting..")
    finally:
        print(f"Testing the best model (with val={best_val_loss:.5f})...")
        test_classifier(args.name, best_model, dataset_test)
