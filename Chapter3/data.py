# data transformations, feature engineering, etc
import pandas as pd
import numpy as np
import kagglehub as kh
import typing as tt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from tabulate import tabulate
import matplotlib.pyplot as plt


SLEEP_DURATION_MAP = {
    '5-6 hours':            5.5,
    '7-8 hours':            7.5,
    'Less than 5 hours':    4.0,
    'More than 8 hours':   10.0,
}

TARGET_COL = "Depression"
CLASS_NAMES = ("no_depression", "depression")


def load_dataset() -> pd.DataFrame:
    """
    Load the original full dataset, but normalize columns: drop spaces, fix one weird column
    :return: Dataframe
    """
    df = kh.dataset_load(
        kh.KaggleDatasetAdapter.PANDAS,
        "hopesb/student-depression-dataset",
        "Student Depression Dataset.csv",
    )
    df.columns = df.columns.str.replace({
        ' ': '',
        '/': '',
        'Haveyoueverhadsuicidalthoughts?': 'HadSuicidalThoughts'
    })
    return df


def drop_rows_inplace(df: pd.DataFrame, filter: pd.Series):
    """
    Drop rows from dataframe based on filter
    :param df: dataframe to drop rows
    :param filter: filter series
    """
    df.drop(df.index[filter], inplace=True)


def sleep_duration_to_hours(sleep_duration: str) -> float:
    return SLEEP_DURATION_MAP.get(sleep_duration, 0.0)


def clean_dataset_simple(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean up the dataset in a simple way - drop weird rows instead of imputing, etc.
    :param df: dataframe to process, result of load_dataset()
    :return: copy of the dataframe with applied transformations
    """
    # not super efficient, but dataset is not large. For bigger data,
    # we might prefer inplace transformations.
    res_df = df.copy()

    # Gender: convert to 0/1 feature
    res_df["IsFemale"] = (res_df.Gender == "Female").astype(int)

    # Age: drop rows with age above 34
    drop_rows_inplace(res_df, res_df.Age > 34)

    # City: drop rows with too rare values, one-hot the rest
    city_cond = res_df.City.value_counts() <= 2
    drop_rows_inplace(res_df, res_df.City.isin(city_cond[city_cond].index))

    # AcademicPressure, WorkPressure: Drop when both are zero
    drop_rows_inplace(res_df, (res_df.AcademicPressure < 0.5) & \
                              (res_df.WorkPressure < 0.5))
    # fill WP -> AP when AP is zero
    ap_zero = res_df.AcademicPressure < 0.5
    res_df.loc[ap_zero, "AcademicPressure"] = res_df.WorkPressure[ap_zero]

    # CGPA: drop rows with zero
    drop_rows_inplace(res_df, res_df.CGPA < 0.5)

    # StudySatisfaction: drop zero values
    drop_rows_inplace(res_df, res_df.StudySatisfaction < 0.5)

    # SleepDuration: drop Others, fill count of hours
    drop_rows_inplace(res_df, res_df.SleepDuration == 'Others')
    res_df['SleepHours'] = res_df.SleepDuration.apply(sleep_duration_to_hours)

    # FinancialStress: drop rows with missing value
    drop_rows_inplace(res_df, res_df.FinancialStress.isna())

    # DietaryHabits: drop rows with "Others", one-hot the rest
    drop_rows_inplace(res_df, res_df.DietaryHabits == 'Others')

    # HadSuicidalThoughts, FamilyHistoryofMentalIllness: convert to 0/1 value
    res_df.HadSuicidalThoughts = \
        (res_df.HadSuicidalThoughts == "Yes").astype(int)
    res_df.FamilyHistoryofMentalIllness = \
        (res_df.FamilyHistoryofMentalIllness == "Yes").astype(int)

    # Drop columns
    for col in ('id', 'Profession', 'JobSatisfaction',
                'SleepDuration', 'Gender'):
        res_df.drop(col, axis=1, inplace=True)

    res_df = pd.get_dummies(
        res_df, columns=['Degree', 'DietaryHabits', 'City'],
        dtype=int)
    return res_df


def data_features(df: pd.DataFrame) -> np.ndarray:
    """
    Return numpy array with features from dataframe processed with clean_dataset_xxx() method
    :param df: dataframe
    :return: numpy array with features
    """
    cols = df.columns.to_list()
    cols.remove(TARGET_COL)
    return df[cols].to_numpy()


def data_target(df: pd.DataFrame) -> np.ndarray:
    """
    Return target column as numpy array
    :param df: dataframe
    :return: numpy array
    """
    return df[TARGET_COL].to_numpy()


def show_confusion_matrix(true_y: np.ndarray, pred_y: np.ndarray, name: str,
                          show_numbers: bool = False, plain: bool = False):
    matrix = confusion_matrix(true_y, pred_y, normalize='true')
    raw_matrix = confusion_matrix(true_y, pred_y, normalize=None)
    idx = list(map(lambda s: "Predicted " + s, CLASS_NAMES))
    col = list(map(lambda s: "True " + s, CLASS_NAMES))

    if show_numbers:
        data = [
            [f"{n:<5} ({p*100:0.2f}%)" for p, n in zip(r1, r2)]
            for r1, r2 in zip(matrix, raw_matrix)
        ]
        df = pd.DataFrame(data, index=idx)
    else:
        df = pd.DataFrame(matrix, index=idx)
        df = df.map(lambda f: f"{f*100:0.2f}%")
    print(f"\n{name} confusion matrix")
    print(tabulate(df, headers=col, tablefmt='outline' if plain else 'rounded_grid'))

    prec = 100 * raw_matrix[1][1] / (raw_matrix[1][1] + raw_matrix[1][0])
    rec = 100 * raw_matrix[1][1] / (raw_matrix[1][1] + raw_matrix[0][1])
    f1 = 2 / (1 / prec + 1 / rec)
    print(f"Precision = {prec:.2f}%, recall = {rec:.2f}%, F1 = {f1:.2f}%\n")


def make_loss_plot(loss: np.ndarray, img_name: str, val: tt.Optional[np.ndarray] = None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(loss, color='black', label="loss", lw=1)
    if val is not None:
        ax.plot(val, '--', color='black', label="validation loss", lw=1)
        ax.legend()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    fig.savefig(img_name)
    print("Loss plot stored to", img_name)


def make_roc_plot(true_y: np.ndarray, prob_y: np.ndarray, img_name: str):
    fpr, tpr, _ = roc_curve(true_y, prob_y)
    auc_score = auc(fpr, tpr)
    print(f"AUC score = {auc_score:.5f}")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    name = f"Depression classifier (AUC={auc_score:.4f})"
    ax.plot(fpr, tpr, '-', label=name, color='black', lw=1.2)
    ax.axline((0, 0), slope=1, ls='--', label="Baseline classifier", color='black', lw=1)
    ax.set_xlabel("False Positive rate")
    ax.set_ylabel("True Positive rate")
    ax.set_xlim(0.0, 1)
    ax.set_ylim(0.0, 1.005)
    ax.legend()
    fig.savefig(img_name)
    print("ROC curve stored to", img_name)


def make_pr_plot(true_y: np.ndarray, prob_y: np.ndarray, img_name: str, baseline: float):
    precision, recall, _ = precision_recall_curve(true_y, prob_y)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    name = f"Depression classifier"
    ax.plot(recall, precision, '-', label=name, color='black', lw=1.2)
    ax.axline((0, baseline), slope=0, ls='--', label="Baseline classifier", color='black', lw=1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1)
    ax.set_ylim(0.0, 1.005)
    ax.legend()
    fig.savefig(img_name)
    print("PR curve stored to", img_name)


def make_f1_plot(true_y: np.ndarray, prob_y: np.ndarray, img_name: str) -> float:
    precision, recall, threshold = precision_recall_curve(true_y, prob_y)
    f1 = 2*precision[:-1]*recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)
    idx = np.argmax(f1)
    best_threshold = threshold[idx]
    best_f1 = f1[idx]
    print(f"Best F1 at threshold={best_threshold:.3f}, f1={100*best_f1:.2f}%")

    idx_05 = np.argmin(np.abs(threshold - 0.5))
    f1_05 = f1[idx_05]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(threshold, f1, '-', color='black', lw=1.2)
    ax.vlines(x=best_threshold, ymin=0, ymax=best_f1, lw=1, linestyles='dashed', colors='black')
    ax.vlines(x=0.5, ymin=0, ymax=f1_05, lw=1, linestyles='dashed', colors='black')
    ax.text(x=best_threshold-0.07, y=best_f1+0.05, s=f"Threshold = {best_threshold:.3f}\n$F_1$ = {best_f1:.3f}")
    ax.text(x=0.5-0.07, y=f1_05+0.02, s=f"Threshold = 0.5\n$F_1$ = {f1_05:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("$F_1$")
    ax.set_xlim(0.0, 1)
    ax.set_ylim(0.0, 1.005)
    fig.savefig(img_name)
    print("F1 curve stored to", img_name)
    return best_threshold