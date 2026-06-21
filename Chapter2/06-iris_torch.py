#!/usr/bin/env -S uv run --script
import data_iris

from sklearn.model_selection import train_test_split
from timer_context import TimerContext
import torch
from torch import nn, optim


class Network(nn.Module):
    def __init__(self, input_size: int, out_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, out_size),
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    torch.manual_seed(19)
    df = data_iris.iris_df()
    one_hot_df = data_iris.one_hot_classes(df)
    data_x = data_iris.data_features(one_hot_df)
    data_y = data_iris.data_pred(one_hot_df)
    print(f"Features shape: {data_x.shape}")
    print(f"Predictions shape: {data_y.shape}")

    X_train, X_test, Y_train, Y_test = train_test_split(
        data_x, data_y, test_size=0.2, random_state=5, shuffle=True)
    print(f"Train shape: {X_train.shape}, test shape: {X_test.shape}")
    X_t = torch.as_tensor(X_train, dtype=torch.float)
    Y_t = torch.as_tensor(Y_train, dtype=torch.float)

    model = Network(X_train.shape[1], Y_train.shape[1])
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss = nn.CrossEntropyLoss()
    print(model)

    with TimerContext() as timer:
        for epoch in range(300):
            optimizer.zero_grad()
            out_t = model(X_t)
            loss_t = loss(out_t, Y_t)
            loss_t.backward()
            optimizer.step()
            if epoch % 30 == 0:
                print(f"{epoch}: loss={loss_t.item()}")
        print(f"Trained in {timer.duration}")

    # inference - predict results on test dataset
    X_test_t = torch.as_tensor(X_test, dtype=torch.float)
    Y_test_t = torch.as_tensor(Y_test, dtype=torch.float)
    with torch.no_grad():
        preds_t = model(X_test_t).detach()
        preds = preds_t.numpy()

    # use argmax, as confusion_matrix requires class labels, not one-hot
    data_iris.show_confusion_matrix(Y_test.argmax(axis=1),
                                    preds.argmax(axis=1))

    probs = torch.softmax(preds_t, dim=-1).detach().numpy()
    data_iris.make_roc_plot(Y_test, probs, "iris-torch-roc.png")
