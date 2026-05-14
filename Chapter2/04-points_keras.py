#!/usr/bin/env -S uv run --script
import data_points

import numpy as np
from timer_context import TimerContext
from keras.utils import set_random_seed
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, ReLU
from keras.losses import MeanAbsoluteError
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD


if __name__ == '__main__':
    set_random_seed(12)
    x, y = data_points.get_points()
    x = np.expand_dims(x, -1)
    x_val, y_val = data_points.get_points(10, 20)
    x_val = np.expand_dims(x_val, -1)

    model = Sequential(name="points")
    model.add(Input(shape=(x.shape[1], )))
    model.add(Dense(100))
    model.add(ReLU())
    model.add(Dropout(0.05))
    model.add(Dense(1))
    model.summary()

    sgd = SGD(learning_rate=0.001)
    model.compile(loss=MeanAbsoluteError(), optimizer=sgd)

    with TimerContext() as timer:
        history = model.fit(
            x, y,
            epochs=100,
            verbose=False,
            validation_data=(x_val, y_val),
            callbacks=[EarlyStopping(patience=10)]
        )
        print(f"Trained in {timer.duration}, did "
              f"{len(history.history['loss'])} epochs")

    print("\nTesting:")
    for min_x, max_x in (
            ( data_points.MIN_X, data_points.MAX_X),
            ( data_points.MAX_X, data_points.MAX_X*2),
            (-data_points.MAX_X, 0.0),
    ):
        test_x, test_y = data_points.get_points(min_x=min_x, max_x=max_x)
        test_x = np.expand_dims(test_x, -1)
        pred = model.predict(test_x, verbose=False)
        pred = np.squeeze(pred)
        mse = data_points.calc_mse(test_y, pred)
        mae = data_points.calc_mae(test_y, pred)
        print(f"Range {min_x:5.1f}...{max_x:4.1f}: "
              f"MSE = {mse:.6f}, MAE = {mae:.6f}")
