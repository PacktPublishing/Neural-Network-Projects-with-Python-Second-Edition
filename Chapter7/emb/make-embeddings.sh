#!/usr/bin/env sh
if test ! -e glove-emb.npy; then
  uv run ../03-make-embeddings.py -m sentence-transformers/average_word_embeddings_glove.6B.300d -o glove -d cuda:0
fi

if test ! -e deberta-emb.npy; then
  uv run ../03-make-embeddings.py -m microsoft/deberta-v3-small -o deberta -d cuda:0 --batch 128
fi

if test ! -e bge-emb.npy; then
  uv run ../03-make-embeddings.py -m BAAI/bge-small-en-v1.5 -o bge -d cuda:0 --batch 128
fi
