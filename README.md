# "Software product <--> Requirements specification" verification model

The repository contains the model intended to solve the problem of automatic verification of accomplishment of software requirements. We employ the cosine similarity between latent representations of Java methods and functional requirements to estimate the measure of accomplishment. The repository contains full implementation of three source code encoders of different granularity, being the subject of research of Innopolis 2020 master thesis project, along with one natural language encoder. Additional details on problem, related work, model structure and demo can be found in [presentation](https://github.com/LeviiBereg/reqsumm/blob/master/Thesis%20presentation.pdf).


## Requirements

 - Python >= 3.6
 - Tensorflow >= 2.0
 - h5py
 - dpu-utils

## Training

Training script `train.py` performs all the essential operations including:
 - preparation and generation of data with `--data-path` and optional `--data-folder` or loading of already preprocessed data using `--data-path` and optional `--preprocessed-data-folder` arguments
 - specification of model structure with a set of hyperparameters along with selection of source code encoder with `--model` argument from `ngram`, `api` and `bert`
 - model training with ability to continue training from checkpoint `--load-cp`

The list of all training options and arguments can be retrieved using the following command:

```sh
python train.py --help
```

## Evaluation

Evaluation script `evaluate.py` scores performance of trained model retrieved from provided checkpoint. We evaluate the retrieval abilities of model to recover Java methods provided description of functional requirement. For this purpose we exploit the Mean Reciprocal Rank, Relevance@k and First-Rank scoring metrics. To evaluate an ability of model to distinguish between relevant and irrelevant pairs of Java methods and functional requirements we score their cosine similarity.

Additional evaluation arguments can be retrieved using the command:

```sh
python evaluate.py --help
```

## Model structure

We propose the Siamese Artificial Neural Network able to learn joint embeddings of Java methods and functional requirements written in natural language. The experimental results demonstrate that cosine similarity with empirically calculated threshold is an adequate measure to verify an accomplishment of functional requirements. 

<img src="https://github.com/LeviiBereg/reqsumm/blob/master/images/Model_Template.png" height="310">

## N-gram encoder

N-gram encoder treats the Java methods as an unordered bag-of-contexts.

<img src="https://github.com/LeviiBereg/reqsumm/blob/master/images/sc_branch.png" height="270">

## API encoder

API encoder builds representations based on an extracted sequence of API calls augmented with function name and body tokens.

<img src="https://github.com/LeviiBereg/reqsumm/blob/master/images/api_encoder.png" height="270">

## BERT Encoder 

For embedding of functional requirements we exploit the [BERT](https://github.com/kpe/bert-for-tf2) small.

<img src="https://github.com/LeviiBereg/reqsumm/blob/master/images/nl_branch.png" height="90">

## Data details

Altered [CodeSearchNet challenge](https://github.com/github/CodeSearchNet) dataset augmented with [Github-Data-Collector](https://github.com/LeviiBereg/Github-Data-Collector) repositories.
Addition processing applied to the original dataset contains the steps of:
- removal of exotic symbols
- filtering of `@params`, `@link` and other description references
- denial of 5% outlying long descriptions
