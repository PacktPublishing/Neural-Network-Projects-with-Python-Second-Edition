import numpy as np
import typing as tt
import pandas as pd
import huggingface_hub as hf
from transformers.models.bert.tokenization_bert import BertTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt


DEFAULT_SEED = 122


def load_dataset() -> tt.Tuple[tt.List[str], tt.List[bool]]:
    texts, labels = [], []
    for name in ("train-00000-of-00001.parquet", "test-00000-of-00001.parquet"):
        path = hf.hf_hub_download(
            "stanfordnlp/imdb", "plain_text/" + name,
            repo_type="dataset")
        df = pd.read_parquet(path)
        texts.extend(df.text.to_list())
        labels.extend(df.label.to_list())
    return texts, labels


def load_dataset_unlabelled() -> tt.List[str]:
    path = hf.hf_hub_download(
        "stanfordnlp/imdb", "plain_text/unsupervised-00000-of-00001.parquet",
        repo_type="dataset")
    df = pd.read_parquet(path)
    return df.text.to_list()


def load_embeddings(prefix: str) -> tt.Tuple[np.ndarray, np.ndarray]:
    emb = np.load(prefix + "-emb.npy")
    labels = np.load(prefix + "-labels.npy")
    return np.astype(emb, np.float32), np.astype(labels, np.float32)


def get_long_texts_indices(
        texts: tt.List[str],
        max_tokens: tt.Optional[int],
        tokenizer: BertTokenizer
) -> tt.List[int]:
    """
    Find indices of texts which end up in more tokens than given limit
    :param texts: list of texts
    :param max_tokens: limit of tokens
    :param tokenizer: tokenizer
    :return: list of texts' indices with long tokenizations
    """
    res = []
    if max_tokens is None:
        return res
    for idx, txt in enumerate(texts):
        tokens = len(tokenizer.tokenize(txt))
        if tokens > max_tokens-2:
            res.append(idx)
    return res


def get_long_text_embedding(
        text: str, model: SentenceTransformer
) -> np.ndarray:
    """
    Calculate embedding vector from the text longer than model's token limit.
    We split text in chunks, encode them and average resulting chunks.
    :param text: text to embed
    :param model: model to use
    :return: resulting embedding vector
    """
    tokens = model.tokenizer.encode(text)
    ofs = 0
    seq_len = model.max_seq_length - 2
    embeddings = []
    while ofs < len(tokens):
        piece = [
            token
            for token in tokens[ofs:ofs+seq_len]
            if token not in model.tokenizer.all_special_ids
        ]

        text_piece = model.tokenizer.decode(piece)
        embeddings.append(model.encode(text_piece))
        ofs += seq_len
    return np.mean(embeddings, axis=0)


def show_confusion_matrix(true_y: np.ndarray, pred_y: np.ndarray):
    matrix = confusion_matrix(true_y, pred_y, normalize='true')
    print("Confusion matrix:")
    print(matrix)


def make_roc_plot(true_y: np.ndarray, prob_y: np.ndarray, img_name: str):
    fpr, tpr, _ = roc_curve(true_y, prob_y)
    auc_score = auc(fpr, tpr)
    print(f"AUC score = {auc_score:.5f}")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    name = f"Classifier (AUC={auc_score:.4f})"
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
    name = f"Classifier"
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
    print(f"F1 at threshold=0.5, f1={100*f1_05:.2f}%")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(threshold, f1, '-', color='black', lw=1.2)
    ax.vlines(x=best_threshold, ymin=0, ymax=best_f1, lw=1, linestyles='dashed', colors='black')
    ax.text(x=best_threshold-0.07, y=best_f1+0.05, s=f"Threshold = {best_threshold:.3f}\n$F_1$ = {best_f1:.3f}")
    if best_f1 - f1_05 > 0.01:
        ax.vlines(x=0.5, ymin=0, ymax=f1_05, lw=1, linestyles='dashed', colors='black')
        ax.text(x=0.5-0.07, y=f1_05+0.02, s=f"Threshold = 0.5\n$F_1$ = {f1_05:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("$F_1$")
    ax.set_xlim(0.0, 1)
    ax.set_ylim(0.0, 1.005)
    fig.savefig(img_name)
    print("F1 curve stored to", img_name)
    return best_threshold