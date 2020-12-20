# Multilingual Visually Grounded Neural Syntax Acquisition

This code is based on [the repo for visually grounded neural syntax acquisition](https://github.com/ExplorerFreda/VGNSL).

## Requirements

Doing `pip install -r requirements.txt` should get you setup with all you need.


# Data
The data to reproduce can be
[found here](https://drive.google.com/drive/folders/1Hk_AJ6yhZj98uQj5KAj4jo3I1uf8pDh1?usp=sharing).
For each language, we explored both BERT embeddings and tokenization, and just using the
pre-tokenized words and fasttext embeddings. 

Every path that is passed to `--data_path` below refers to the data that is downloaded
above.


# Commands to reproduce

Currently, the code takes in either subword tokenized input (and subword embeddings), or
word tokenized input (and word embeddings). **In the commands below, please use the
flags `--init_embeddings_key fasttext --init-embeddings_type partial-fixed`** if you'd
like to use FastText embeddings(this is of course, in addition, to changing what
`--data_dir` points to as described in the "Data" section above).

The code to reproduce the results in the paper is
in different commits. Therefore, one has to checkout specific Git commits and then 
run the commands.

## Reproduce "separate parser, separate aligner" results
This has to be run for each language(change `multi30k_en` to something else for another
language)

Git commit: `4e691851899157d50163376e8d1706fd3fb4b18c`

```bash
python src/train.py \
  --data_path subword/multi30k_en/ \
  --logger_name /path/to/model/outputs/dir/ \
  --init_embeddings_key bert \
  --init_embeddings_type subword \
  --init_embeddings 1 --word_dim 768 --embed_size 768 --batch_size 128 --workers 1
```

## Reproduce "separate parser, common aligner" results
Git commit: `4e691851899157d50163376e8d1706fd3fb4b18c`

```bash
python src/train.py \
  --data_path subword/ \
  --logger_name /path/to/model/outputs/dir/ \
  --init_embeddings_key bert --init_embeddings_type subword --init_embeddings 1 \
  --word_dim 768 --embed_size 768 --batch_size 128 --workers 1
```

## Reproduce "Common parser, common aligner" results 
Git commit: `0b3e6897da1d46e24c8760b9db8bd5db0179b15f`

```bash
python src/train.py \
  --data_path subword/ \
  --logger_name /path/to/model/outputs/dir/ \
  --init_embeddings_key bert --init_embeddings_type subword --init_embeddings 1 \
  --word_dim 768 --embed_size 768 --batch_size 128
```
