#!/usr/bin/env -S uv run --script
import data

from sklearn.model_selection import train_test_split
from timer_context import TimerContext

from keras.utils import set_random_seed
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.activations import sigmoid, relu
from keras.losses import BinaryCrossentropy
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD


if __name__ == '__main__':
    set_random_seed(112)
    orig_df = data.load_dataset()
    prep_df = data.clean_dataset_simple(orig_df)
    data_x = data.data_features(prep_df)
    data_y = data.data_target(prep_df)

    # test data
    X_train, X_test, Y_train, Y_test = train_test_split(
        data_x, data_y, test_size=0.2, random_state=14, shuffle=True)

    # model
    model = Sequential(name="depression")
    model.add(Input(shape=(X_train.shape[1], )))
    model.add(Dense(200, activation=relu))
    model.add(Dense(200, activation=relu))
    model.add(Dense(200, activation=relu))
    model.add(Dense(1, activation=sigmoid))
    model.summary()

    model.compile(
        loss=BinaryCrossentropy(),
        optimizer=SGD(learning_rate=5e-3),
    )

    with TimerContext() as timer:
        history = model.fit(
            X_train, Y_train,
            epochs=400,
            verbose=False,
            validation_data=(X_test, Y_test),
            callbacks=[
                EarlyStopping(patience=20),
            ]
        )
        print(f"Trained in {timer.duration}, did "
              f"{len(history.history['loss'])} epochs")
        data.make_loss_plot(history.history['loss'], "keras-loss.svg",
                            history.history['val_loss'])

    preds = model.predict(X_test, verbose=False)
    preds = preds.squeeze()
    binary_preds = (preds > 0.5).astype(int)
    data.show_confusion_matrix(Y_test, binary_preds, "Test data", show_numbers=True)

    # PR, ROC plots
    baseline = data_y.sum() / data_y.shape[0]
    data.make_pr_plot(Y_test, preds, "keras-pr.svg", baseline=baseline)
    data.make_roc_plot(Y_test, preds, "keras-roc.svg")
    best_thr = data.make_f1_plot(Y_test, preds, "keras-f1.svg")

    binary_preds_thr = (preds > best_thr).astype(int)
    data.show_confusion_matrix(Y_test, binary_preds_thr, "Test data", show_numbers=True)

    model.save("model.keras")
