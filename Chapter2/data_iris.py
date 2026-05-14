# iris dataset from huggingface
import huggingface_hub as hf
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tabulate import tabulate

import plotly.graph_objects as go

# input features
COLS_FEATURES = ("SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm")

# values to predict (one-hot encoded)
COLS_PRED = ("class_Iris-setosa", "class_Iris-versicolor", "class_Iris-virginica")

# Class names
CLASS_NAMES = tuple(map(lambda s: s.split("-")[-1], COLS_PRED))


def iris_df() -> pd.DataFrame:
    """
    Load Iris dataframe from huggingface, drop Id column and filter two outliers.
    :return: Dataframe with data
    """
    path = hf.hf_hub_download("scikit-learn/iris", "Iris.csv", repo_type="dataset")
    df = pd.read_csv(path)
    df.drop('Id', axis=1, inplace=True)

    outlier_cond = (df.SepalWidthCm <= 2.3) & (df.Species == 'Iris-setosa') | \
                   (df.SepalLengthCm <= 4.9) & (df.Species == 'Iris-virginica')
    df.drop(df[outlier_cond].index, inplace=True)
    return df


def one_hot_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode of classes replacing the Species column with class_xxx one-hot values
    :param df: dataframe
    :return: dataframe with class replaced with one-hot encoded data
    """
    return pd.get_dummies(df, columns=["Species"], prefix="class", dtype=int)


def data_features(df: pd.DataFrame) -> np.array:
    """
    From dataframe get the input vectors
    :param df: iris dataframe
    :return: numpy array
    """
    return df[list(COLS_FEATURES)].to_numpy()


def data_pred(df: pd.DataFrame) -> np.array:
    """
    From dataframe get the target vectors
    :param df: iris dataframe
    :return: numpy array
    """
    return df[list(COLS_PRED)].to_numpy()


def show_confusion_matrix(true_y: np.array, pred_y: np.array):
    matrix = confusion_matrix(true_y, pred_y)
    idx = list(map(lambda s: "Predicted " + s, CLASS_NAMES))
    col = list(map(lambda s: "True " + s, CLASS_NAMES))
    df = pd.DataFrame(matrix, index=idx)
    print("\nConfusion matrix")
    print(tabulate(df, headers=col, tablefmt='rounded_grid'))


def make_roc_plot(true_y: np.array, prob_y: np.array, img_name: str):
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    for i, class_name in enumerate(COLS_PRED):
        fpr, tpr, _ = roc_curve(true_y[:, i], prob_y[:, i])
        name = f"{class_name.split('-')[1]} (AUC={auc(fpr, tpr):.4f})"
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))
    fig.write_image(img_name, scale=5)
