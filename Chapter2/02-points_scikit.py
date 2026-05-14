#!/usr/bin/env -S uv run --script
import data_points

import numpy as np
import pickle
import pathlib
from sklearn.neural_network import MLPRegressor
from timer_context import TimerContext


if __name__ == "__main__":
    # show the input and target data shapes
    x, y = data_points.get_points()
    print(f"Features shape: {x.shape}")
    print(f"Predictions shape: {y.shape}")

    # add an extra dimension to convert the vector into 1-column matrix
    x = np.expand_dims(x, -1)

    # our model - two-layer perceptron
    model = MLPRegressor(
        loss="squared_error",
        hidden_layer_sizes=(100, ),
        learning_rate_init=0.01,
        random_state=13,
    )
    # fit the model
    with TimerContext() as timer:
        model.fit(x, y)
        print(f"Trained in {timer.duration}")

    # model was trained, check the weights
    print("\nModel coefficient shapes:")
    for i, (w, b) in enumerate(zip(model.coefs_, model.intercepts_)):
        print(f"{i}: weights={w.shape}, bias={b.shape}")

    print("\nTesting:")
    # inference - generate points to test for different ranges
    for min_x, max_x in (
            ( data_points.MIN_X, data_points.MAX_X),      # training data range
            ( data_points.MAX_X, data_points.MAX_X*2),    # right after the training range
            (-data_points.MAX_X, 0.0),                    # right before the training range
    ):
        test_x, test_y = data_points.get_points(min_x=min_x, max_x=max_x)
        test_x = np.expand_dims(test_x, -1)
        pred = model.predict(test_x)
        mse = data_points.calc_mse(test_y, pred)
        mae = data_points.calc_mae(test_y, pred)
        print(f"Range {min_x:5.1f}...{max_x:4.1f}: "
              f"MSE = {mse:.6f}, MAE = {mae:.6f}")

    data = pickle.dumps(model)
    pathlib.Path("points-scikit.dat").write_bytes(data)
    print(f"\n{len(data)} bytes stored into points-scikit.dat")
