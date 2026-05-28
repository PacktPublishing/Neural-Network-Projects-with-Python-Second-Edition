#!/usr/bin/env -S uv run --script
import data

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from timer_context import TimerContext


if __name__ == '__main__':
    orig_df = data.load_dataset()
    prep_df = data.clean_dataset_simple(orig_df)
    data_x = data.data_features(prep_df)
    data_y = data.data_target(prep_df)
    print(f"Total samples: {data_y.shape[0]}, positive: {data_y.sum()}")

    # test data
    X_train, X_test, Y_train, Y_test = train_test_split(
        data_x, data_y, test_size=0.2, random_state=14, shuffle=True)

    print(f"Train: {X_train.shape}, test: {X_test.shape}")

    model = MLPClassifier(
        hidden_layer_sizes=(200, 200, 200),
        max_iter=1000,
        learning_rate_init=0.01,
        random_state=17,
    )

    with TimerContext() as timer:
        model.fit(X_train, Y_train)
        print(f"Trained in {timer.duration}")

    preds = model.predict(X_test)
    data.show_confusion_matrix(Y_test, preds, "Test data", show_numbers=True)

    # Loss plot
    data.make_loss_plot(model.loss_curve_, "scikit-loss.svg")

    # PR/F1/ROC plot
    probs = model.predict_proba(X_test)
    baseline = data_y.sum() / data_y.shape[0]
    data.make_pr_plot(Y_test, probs[:, 1], "scikit-pr.svg", baseline=baseline)
    data.make_roc_plot(Y_test, probs[:, 1], "scikit-roc.svg")
    best_thr = data.make_f1_plot(Y_test, probs[:, 1], "scikit-f1.svg")

    preds_best = (probs > best_thr)[:, 1].astype(int)
    data.show_confusion_matrix(
        Y_test, preds_best, f"Test data at threshold={best_thr:.4f}",
        show_numbers=True)
