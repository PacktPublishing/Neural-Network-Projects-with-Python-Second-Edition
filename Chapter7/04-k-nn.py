import argparse
import data
import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from timer_context import TimerContext
import faiss


def normalize_vectors(emb: np.ndarray) -> np.ndarray:
    """
    Return normalized vectors (divide them on L2 norm)
    :param emb: matrix with vectors as rows
    :return: normalized vectors
    """
    # compute the euclidean length of all the vectors
    norm = np.linalg.vector_norm(emb, ord=2, axis=1)
    # normalize vectors by broadcasting along the 2nd dimension
    return emb / np.expand_dims(norm, 1)


def predict_numpy(
        emb_train: np.ndarray, labels_train: np.ndarray,
        emb_test: np.ndarray, labels_test: np.ndarray,
        k: int
) -> np.ndarray:
    e_train = normalize_vectors(emb_train)
    e_test = normalize_vectors(emb_test)

    pred_probs = []
    for query_emb, true_label in tqdm.tqdm(zip(e_test, labels_test),
                                           total=labels_test.shape[0]):
        sim = np.matmul(e_train, query_emb)
        ord = np.argsort(sim, descending=True)
        pred = np.mean(labels_train[ord][:k])
        pred_probs.append(float(pred))
    return np.array(pred_probs)


def predict_faiss(
        emb_train: np.ndarray, labels_train: np.ndarray,
        emb_test: np.ndarray, labels_test: np.ndarray,
        k: int
) -> np.ndarray:
    e_train = normalize_vectors(emb_train)
    e_test = normalize_vectors(emb_test)
    # build Inner Product index
    index = faiss.IndexFlatIP(e_train.shape[1])
    index.add(e_train)

    pred_probs = []
    for query_emb, true_label in tqdm.tqdm(zip(e_test, labels_test),
                                           total=labels_test.shape[0]):
        q = np.expand_dims(query_emb, 0)
        dist, indices = index.search(q, k)
        pred = np.mean(labels_train[indices[0]])
        pred_probs.append(float(pred))
    return np.array(pred_probs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True,
                        help="Data prefix (stored in numpy arrays)")
    parser.add_argument("--seed", type=int, default=data.DEFAULT_SEED,
                        help="Random seed, default=" + str(data.DEFAULT_SEED))
    parser.add_argument("-k", type=int, required=True,
                        help="How many similar vectors to use")
    parser.add_argument("-m", "--mode", choices=('numpy', 'faiss'),
                        default="numpy", help="Mode of search: numpy or faiss")
    args = parser.parse_args()

    emb, labels = data.load_embeddings(args.data)
    print(emb.shape, labels.shape)

    emb_train, emb_test, labels_train, labels_test = train_test_split(
        emb, labels, random_state=args.seed, test_size=0.2)
    print(emb_train.shape)

    with TimerContext() as timer:
        if args.mode == "numpy":
            pred_probs = predict_numpy(
                emb_train, labels_train, emb_test, labels_test,
                args.k
            )
        elif args.mode == "faiss":
            pred_probs = predict_faiss(
                emb_train, labels_train, emb_test, labels_test,
                args.k
            )
        print(f"Test took {timer.duration}")
    binary_preds = pred_probs > 0.5

    data.show_confusion_matrix(labels_test, binary_preds)
    out_prefix = f"top-{args.k}-{args.data}"
    data.make_pr_plot(labels_test, pred_probs, out_prefix + "-pr.svg", baseline=0.5)
    data.make_roc_plot(labels_test, pred_probs, out_prefix + "-roc.svg")
    best_thr = data.make_f1_plot(labels_test, pred_probs, out_prefix + "-f1.svg")

    binary_preds = pred_probs > best_thr
    data.show_confusion_matrix(labels_test, binary_preds)
