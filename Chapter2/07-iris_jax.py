#!/usr/bin/env -S uv run --script
import data_iris
import typing as tt
from timer_context import TimerContext

import jax
from jax import value_and_grad, random
import jax.numpy as jnp
from sklearn.model_selection import train_test_split


TLayer = tt.Tuple[jax.Array, jax.Array]
LR = 0.1


def dense_params(m: int, n: int, key, scale: float = 1e-2) -> TLayer:
    """
    Create parameters for the dense layer
    :param m: count of rows
    :param n: count of columns
    :param key: random key
    :param scale: scale of the random values used for initalization
    :return: weight matrix and bias vector
    """
    w_key, b_key = random.split(key)
    w = random.normal(w_key, (m, n)) * scale
    b = random.normal(b_key, (n, )) * scale
    return w, b


def make_net(key) -> tt.List[TLayer]:
    res = []
    key, t_key = random.split(key)
    res.append(dense_params(len(data_iris.COLS_FEATURES), 100, t_key))
    key, t_key = random.split(key)
    res.append(dense_params(100, len(data_iris.COLS_PRED), t_key))
    return res


def show_net(net: tt.List[TLayer]):
    print("Network layers:")
    for i, (w, b) in enumerate(net):
        print(f"{i}: w={w.shape}), b={b.shape}")


def predict(net: tt.List[TLayer], batch: jax.Array) -> jax.Array:
    (W1, b1), (W2, b2) = net
    v = batch.dot(W1) + b1
    v = jnp.maximum(v, 0.0)
    v = v.dot(W2) + b2
    return v


def loss(net: tt.List[TLayer], x: jax.Array, y: jax.Array) -> jax.Array:
    pred = predict(net, x)
    log_probs = jax.nn.log_softmax(pred, axis=-1)
    res = -jnp.mean(jnp.sum(log_probs * y, axis=-1))
    return res


def update(net: tt.List[TLayer], x: jax.Array, y: jax.Array) -> \
        tt.Tuple[tt.List[TLayer], jax.Array]:
    loss_value, grads = value_and_grad(loss)(net, x, y)
    return [
        (w - LR * dw, b - LR * db)
        for (w, b), (dw, db) in zip(net, grads)
    ], loss_value


if __name__ == '__main__':
    key = random.key(42)
    key, t_key = random.split(key)
    net = make_net(t_key)
    show_net(net)

    df = data_iris.iris_df()
    one_hot_df = data_iris.one_hot_classes(df)
    data_x = data_iris.data_features(one_hot_df)
    data_y = data_iris.data_pred(one_hot_df)
    X_train, X_test, Y_train, Y_test = train_test_split(
        data_x, data_y, test_size=0.2, random_state=4, shuffle=True)

    X_a = jnp.array(X_train)
    Y_a = jnp.array(Y_train)

    with TimerContext() as timer:
        for epoch in range(400):
            net, loss_val = update(net, X_a, Y_a)
            if epoch % 30 == 0:
                print(f"{epoch}: {loss_val}")
        print(f"Trained in {timer.duration}")

    Xt_a = jnp.array(X_test)
    pred_a = predict(net, Xt_a)
    data_iris.show_confusion_matrix(Y_test.argmax(axis=1),
                                    pred_a.argmax(axis=1))
