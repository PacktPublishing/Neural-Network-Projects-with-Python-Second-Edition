import argparse
import data
import numpy as np
import tqdm
from sentence_transformers import SentenceTransformer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True,
                        help="Name of the model")
    parser.add_argument("-d", "--device", default="cpu",
                        help="Device to use")
    parser.add_argument("-o", "--output", required=True,
                        help="Output file prefix")
    parser.add_argument("--batch", default=32, type=int,
                        help="Size of encoder batch, default=32")
    args = parser.parse_args()

    data_texts, data_labels = data.load_dataset()
    print(f"Loaded {len(data_texts)} rows")

    model = SentenceTransformer(args.model, device=args.device)
    print(model)
    long_texts_indices = data.get_long_texts_indices(
        data_texts, model.max_seq_length, model.tokenizer)
    print(f"Have {len(long_texts_indices)} long texts to be filled after encoding")

    emb = model.encode(data_texts, batch_size=args.batch,
                       show_progress_bar=True)
    print(f"Got embeddings of shape {emb.shape}")

    if long_texts_indices:
        print("Filling long text embeddings...")
        for idx in tqdm.tqdm(long_texts_indices):
            e = data.get_long_text_embedding(data_texts[idx], model)
            emb[idx] = e

    np.save(args.output + "-emb.npy", emb, allow_pickle=False)
    labels = np.array(data_labels, dtype=np.uint8)
    np.save(args.output + "-labels.npy", labels, allow_pickle=False)
    print(f"Saved {args.output}-emb.npy and {args.output}-labels.npy")
