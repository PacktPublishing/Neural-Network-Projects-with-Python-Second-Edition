import data

import time
import tqdm
import openai
import numpy as np
import argparse

MODEL = "text-embedding-3-small"
BATCH_SIZE = 50


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=MODEL,
                        help="Name of the model, default=" + MODEL)
    parser.add_argument("-o", "--output", required=True,
                        help="Output file prefix")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE,
                        help="Size of one request")
    args = parser.parse_args()

    data_texts, data_labels = data.load_dataset()
    print(f"Loaded {len(data_texts)} rows")

    client = openai.OpenAI()
    batch_size = args.batch
    emb_data = []
    total_tokens = 0

    for start_idx in tqdm.tqdm(range(len(data_texts) // batch_size)):
        texts = data_texts[start_idx*batch_size:(start_idx+1)*batch_size]
        resp = None
        for retry in range(10):
            try:
                resp = client.embeddings.create(input=texts, model=MODEL)
                break
            except openai.RateLimitError:
                print("Rate limit, sleeping 30 seconds and retrying...")
                time.sleep(30)
        total_tokens += resp.usage.total_tokens
        for e in resp.data:
            emb_data.append(e.embedding)
    print(f"Used tokens: {total_tokens}")
    emb = np.array(emb_data, dtype=np.float32)
    print(f"Embeddings shape: {emb.shape}")

    np.save(args.output + "-emb.npy", emb, allow_pickle=False)
    labels = np.array(data_labels, dtype=np.uint8)
    np.save(args.output + "-labels.npy", labels, allow_pickle=False)
    print(f"Saved {args.output}-emb.npy and {args.output}-labels.npy")
