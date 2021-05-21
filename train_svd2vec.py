#!/usr/bin/env python3

import argparse
from pathlib import Path

import nltk
from svd2vec import svd2vec

parser = argparse.ArgumentParser(description="Train a svd2vec model on an unannotated dataset.")

parser.add_argument("--tokenization", default="nltk", type=str, help="")
parser.add_argument("--window", default=10, type=int, help="")
parser.add_argument("--min-count", default=20, type=int, help="")

parser.add_argument("--input-data", type=str, help="")
parser.add_argument("--output-model", type=str, help="")

args = parser.parse_args()


if args.tokenization == "nltk":
    documents = [nltk.word_tokenize(line) for line in open(args.input_data, "r").readlines()]
elif args.tokenization == "space":
    documents = [open(args.input_data, "r").read().split(" ")]
elif args.tokenization == "space2":
    documents = [line.split(" ") for line in open(args.input_data, "r").readlines()]
else:
    raise ValueError(f"Unknown tokenization: {args.tokenization}")

svd = svd2vec(documents, window=args.window, min_count=args.min_count)

Path(args.output_model).parent.mkdir(parents=True, exist_ok=True)
svd.save(args.output_model)


# Use/Debug:
# print(svd.similarity("good", "bad"))
