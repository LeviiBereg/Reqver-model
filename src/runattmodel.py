import re
import os
import time
import h5py
import math
import string
import pickle
import datetime
import javalang
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as tfh
import tensorflow_datasets as tfds
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from dpu_utils.mlutils import Vocabulary
from gensim.models import KeyedVectors as word2vec
from sklearn.model_selection import train_test_split

import bert
FullTokenizer = bert.bert_tokenization.FullTokenizer

MODEL_TYPE = 'att'
LANGUAGE = "java" #"python"
DATA_PATH = "/home/vkarpov"
DATA_FOLDER = f"{LANGUAGE}/short"
TRAIN_FILE  = f"{LANGUAGE}_train_0.jsonl"
TEST_FILE   = f"{LANGUAGE}_test_0.jsonl"
VALID_FILE  = f"{LANGUAGE}_valid_0.jsonl"

use_cols = ["code", "docstring"] # code_tokens

def read_data(scope, n_splits):
    def get_file(scope, split_ind):
        return pd.read_json(f"{DATA_PATH}/{DATA_FOLDER}/{LANGUAGE}_{scope}_{split_ind}.jsonl", lines=True)[use_cols]

    train_df = get_file(scope, split_ind=0)
    for i in range(1, n_splits):
        train_df = train_df.append(get_file(scope, split_ind=i), ignore_index=True)
    return train_df

model_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"
bert_vocab = tfh.KerasLayer(model_url, trainable=False)
bert_layer_desc = bert.BertModelLayer(**bert.BertModelLayer.Params(
  vocab_size               = 30522,        # embedding params
  use_token_type           = False,
  use_position_embeddings  = True,

  num_layers               = 4,           # transformer encoder params
  hidden_size              = 128,
  num_heads=4,
  hidden_dropout           = 0.1,
  intermediate_size        = 4*128,
  intermediate_activation  = "gelu",

  adapter_size             = None,         # see arXiv:1902.00751 (adapter-BERT)

  shared_layer             = False,        # True for ALBERT (arXiv:1909.11942)
  embedding_size           = None,         # None for BERT, wordpiece embedding size for ALBERT

  # name                     = "bert"        # any other Keras layer params
))


bert_layer_sc = bert.BertModelLayer(**bert.BertModelLayer.Params(
  vocab_size               = 30522,        # embedding params
  use_token_type           = False,
  use_position_embeddings  = True,

  num_layers               = 4,           # transformer encoder params
  hidden_size              = 128,
  num_heads=4,
  hidden_dropout           = 0.1,
  intermediate_size        = 4*128,
  intermediate_activation  = "gelu",

  adapter_size             = None,         # see arXiv:1902.00751 (adapter-BERT)

  shared_layer             = False,        # True for ALBERT (arXiv:1909.11942)
  embedding_size           = None,         # None for BERT, wordpiece embedding size for ALBERT

  # name                     = "bert"        # any other Keras layer params
))


vocab_file = bert_vocab.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_vocab.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)


def cleaning(text):
    '''Performs cleaning of text of unwanted symbols,
    excessive spaces and transfers to lower-case
    '''

    # {@link FaultMessageResolver} => link
    text = re.sub(r"\{?@(\w+)\s+\S+\}?", r'\1', text)
    # delete XML tags
    text = re.sub(r'<[\/a-zA-Z]+>', "", text)
    # remove excessive spaces
    #     text = re.sub(r'\s+', " ", text)

    text = ''.join(character for character in text if character in string.printable)
    text = text.lower().strip()

    return text


def generate_bert_input(text, max_seq_length):

    tokenized_text = [["[CLS]"] + tokenizer.tokenize(seq)[:max_seq_length-2] + ["[SEP]"] for seq in text]
    input_ids   = [tokenizer.convert_tokens_to_ids(tokens_seq) for tokens_seq in tokenized_text]
    input_mask  = [[1] * len(input_seq) for input_seq in input_ids]
    segment_ids = [[0] * max_seq_length for _ in range(len(input_ids))]
    input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids, maxlen=max_seq_length, padding='post', truncating='post')
    input_mask = tf.keras.preprocessing.sequence.pad_sequences(input_mask, maxlen=max_seq_length, padding='post', truncating='post')
    segment_ids = tf.keras.preprocessing.sequence.pad_sequences(segment_ids, maxlen=max_seq_length, padding='post', truncating='post')

    return input_ids, input_mask, segment_ids


def write_hdf5_data(dataset, dataset_name, data_folder='preprocessed_data'):
    with h5py.File(f'{DATA_PATH}/{data_folder}/{MODEL_TYPE}_encoder/{dataset_name}.h5', 'w') as hf:
        hf.create_dataset(dataset_name, data=dataset)


def read_hdf5_data(dataset_name, data_folder='preprocessed_data', start_index=0, end_index=-1):
    with h5py.File(f'{DATA_PATH}/{data_folder}/{MODEL_TYPE}_encoder/{dataset_name}.h5', "r") as f:
        dataset = f[dataset_name]
        end_index = end_index if end_index > 0 else dataset.size
        res = dataset[start_index:end_index]
    return res


def generate_inputs(scope='train', n_splits=1):
    pddf = read_data(scope=scope, n_splits=n_splits)
    pddf.docstring = pddf.docstring.apply(cleaning)

    desc_word_ids, desc_input_mask, desc_segment_ids = generate_bert_input(pddf.docstring, desc_max_seq_length)
    sc_word_ids, sc_input_mask, sc_segment_ids = generate_bert_input(pddf.code, sc_max_seq_length)
    return desc_word_ids, desc_input_mask, desc_segment_ids, sc_word_ids, sc_input_mask, sc_segment_ids, pddf

sc_max_seq_length = 200

desc_max_seq_length = 180 # 0.95 quantile == 178

generate_data_flag = False
if generate_data_flag:

    desc_word_ids, desc_input_mask, desc_segment_ids, sc_word_ids, sc_input_mask, sc_segment_ids, train_df = \
        generate_inputs(scope='train', n_splits=16)

    print("Train dataset size", len(train_df))

    v_desc_word_ids, v_desc_input_mask, v_desc_segment_ids, v_sc_word_ids, v_sc_input_mask, v_sc_segment_ids = \
        generate_inputs(scope='valid', n_splits=1)[:-1]
    print("Validation dataset size", len(v_desc_word_ids))

    assert np.all((desc_word_ids > 0).sum(axis=1) == desc_input_mask.sum(axis=1)), 'wrong bert input mask'
    assert desc_word_ids.shape == desc_input_mask.shape, 'bert inputs shape mismatch'
    assert desc_word_ids.shape == desc_segment_ids.shape, 'bert inputs shape mismatch'
    assert len(desc_word_ids) == len(sc_word_ids), 'nl and sc branches inputs mismatch'

    write_hdf5_data(sc_word_ids, 'sc_word_ids')
    write_hdf5_data(sc_input_mask, 'sc_input_mask')
    write_hdf5_data(desc_word_ids, 'desc_word_ids')
    write_hdf5_data(desc_input_mask, 'desc_input_mask')

    write_hdf5_data(v_sc_word_ids, 'v_sc_word_ids')
    write_hdf5_data(v_sc_input_mask, 'v_sc_input_mask')
    write_hdf5_data(v_desc_word_ids, 'v_desc_word_ids')
    write_hdf5_data(v_desc_input_mask, 'v_desc_input_mask')
else:

    n_samples = -1
    n_val_samples = -1

    sc_word_ids = read_hdf5_data('sc_word_ids', end_index=n_samples)
    sc_input_mask = read_hdf5_data('sc_input_mask', end_index=n_samples)
    sc_segment_ids = np.zeros(sc_word_ids.shape, dtype=np.int32)
    desc_word_ids = read_hdf5_data('desc_word_ids', end_index=n_samples)
    desc_input_mask = read_hdf5_data('desc_input_mask', end_index=n_samples)
    desc_segment_ids = np.zeros(desc_word_ids.shape, dtype=np.int32)

    v_sc_word_ids = read_hdf5_data('v_sc_word_ids', end_index=n_val_samples)
    v_sc_input_mask = read_hdf5_data('v_sc_input_mask', end_index=n_val_samples)
    v_sc_segment_ids = np.zeros(v_sc_word_ids.shape, dtype=np.int32)
    v_desc_word_ids = read_hdf5_data('v_desc_word_ids', end_index=n_val_samples)
    v_desc_input_mask = read_hdf5_data('v_desc_input_mask', end_index=n_val_samples)
    v_desc_segment_ids = np.zeros(v_desc_word_ids.shape, dtype=np.int32)

    print("Train dataset size", len(sc_word_ids))
    print("Validation dataset size", len(v_sc_word_ids))

dense_units = 400

input_word_ids = tf.keras.layers.Input(shape=(desc_max_seq_length,),
                                       dtype=tf.int32,
                                       name="desc_input_word_ids")

desc_pooling = tf.keras.layers.Lambda(lambda x: tf.math.reduce_max(x, axis=1))# tf.keras.layers.Lambda(lambda seq: seq[:, 0, :])
desc_dense = tf.keras.layers.Dense(dense_units, activation='tanh', name="desc_dense")


code_input_word_ids = tf.keras.layers.Input(shape=(sc_max_seq_length,),
                                       dtype=tf.int32,
                                       name="sc_input_word_ids")

sc_pooling = tf.keras.layers.Lambda(lambda x: tf.math.reduce_max(x, axis=1)) #tf.keras.layers.Lambda(lambda x: tf.math.reduce_max(x, axis=1))
sc_dense = tf.keras.layers.Dense(dense_units, activation='tanh', name="sc_dense")


similarity_mode = 'cosine' # 'cosine' 'dense'

desc_sequence_output = bert_layer_desc(input_word_ids)
desc_pooled_output = desc_pooling(desc_sequence_output)
desc_pooled_output = tf.keras.layers.Dropout(0.5)(desc_pooled_output)
desc_output = desc_dense(desc_pooled_output)

desc_embedding_model = tf.keras.Model(inputs=input_word_ids,
                                      outputs=desc_output,
                                      name='desc_embedding_model')


sc_sequence_output = bert_layer_sc(code_input_word_ids)
sc_pooled_output = sc_pooling(sc_sequence_output)
sc_pooled_output = tf.keras.layers.Dropout(0.5)(sc_pooled_output)
sc_output = sc_dense(sc_pooled_output)

code_embedding_model = tf.keras.Model(inputs=code_input_word_ids,
                                    outputs=sc_output,
                                    name='sc_embedding_model')


norm_desc = tf.norm(desc_output, axis=-1, keepdims=True) + 1e-10
norm_sc   = tf.norm(sc_output, axis=-1, keepdims=True)   + 1e-10
cos_similarity = tf.matmul(desc_output/norm_desc,
                                sc_output/norm_sc,
                                transpose_a=False,
                                transpose_b=True,
                                name='code_query_cooccurrence_logits')  # (batch_size, batch_size)

train_model = tf.keras.Model(inputs=[input_word_ids,
                                     code_input_word_ids],
                             outputs=cos_similarity,#cos_similarity,
                             name=f'train_{MODEL_TYPE}_model')

def create_learning_rate_scheduler(max_learn_rate=5e-5,
                                   end_learn_rate=1e-7,
                                   warmup_epoch_count=10,
                                   total_epoch_count=90):

    def lr_scheduler(epoch):
        if epoch < warmup_epoch_count:
            res = (max_learn_rate/warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate * math.exp(math.log(end_learn_rate/max_learn_rate)*(epoch-warmup_epoch_count+1)/(total_epoch_count-warmup_epoch_count+1))
        return float(res)
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    return learning_rate_scheduler

def cos_loss(dummy, cosine_similarities):
    loss_margin = 0.5
    # A max-margin-like loss, but do not penalize negative cosine similarities.
    neg_matrix = tf.linalg.diag(-tf.linalg.diag_part(cosine_similarities))
    per_sample_loss = tf.maximum(1e-6, loss_margin
                                     - tf.linalg.diag_part(cosine_similarities)
                                     + tf.reduce_mean(tf.nn.relu(cosine_similarities + neg_matrix), axis=-1))
    loss = tf.reduce_mean(per_sample_loss)
    return loss

def mrr(dummy, cosine_similarities):
    correct_scores = tf.linalg.diag_part(cosine_similarities)
    compared_scores = cosine_similarities >= tf.expand_dims(correct_scores, axis=-1)
    mrr = 1 / tf.reduce_sum(tf.cast(compared_scores, dtype=tf.float32), axis=1)
    return mrr

def frank(dummy, cosine_similarities):
    correct_scores = tf.linalg.diag_part(cosine_similarities)
    retrieved_before = cosine_similarities > tf.expand_dims(correct_scores, axis=-1)
    rel_ranks = tf.reduce_sum(tf.cast(retrieved_before, dtype=tf.float32), axis=1) + 1
    return rel_ranks

def relevantatk(cosine_similarities, k):
    correct_scores = tf.linalg.diag_part(cosine_similarities)
    compared_scores = cosine_similarities > tf.expand_dims(correct_scores, axis=-1)
    compared_scores = tf.reduce_sum(tf.cast(compared_scores, dtype=tf.float32), axis=1)
    compared_scores = tf.cast(compared_scores < k, dtype=tf.float32)
    return compared_scores

def relevantat10(dummy, cosine_similarities):
    return relevantatk(cosine_similarities, k=10)

def relevantat5(dummy, cosine_similarities):
    return relevantatk(cosine_similarities, k=5)

def relevantat1(dummy, cosine_similarities):
    return relevantatk(cosine_similarities, k=1)

optimizer = 'adam'

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"{DATA_PATH}/logs/fit/{MODEL_TYPE}/" + current_time
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = f"{DATA_PATH}/model_checkpoints/{MODEL_TYPE}/cp.ckpt"

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_best_only=True,
                                                 save_weights_only=True,
                                                 monitor='val_mrr',
                                                 mode='max')

train_model.compile(loss=cos_loss, optimizer=optimizer, metrics=[mrr])
code_embedding_model.compile(loss=cos_loss, optimizer=optimizer)
desc_embedding_model.compile(loss=cos_loss, optimizer=optimizer)

batch_size = 256
valid_batch_size = 256
train_samples = len(desc_word_ids)
valid_samples = len(v_desc_word_ids)
train_steps_per_epoch = train_samples // batch_size
valid_steps_per_epoch = valid_samples // valid_batch_size
epochs = 50

train_data = tf.data.Dataset.from_tensor_slices(((desc_word_ids,
                                                  sc_word_ids),
                                                 np.ones((len(desc_word_ids),1))))\
                            .shuffle(len(desc_word_ids), reshuffle_each_iteration=True)\
                            .batch(batch_size, drop_remainder=True)\
                            .repeat()
valid_data = tf.data.Dataset.from_tensor_slices(((v_desc_word_ids,
                                                  v_sc_word_ids),
                                                 np.ones((len(v_desc_word_ids),1))))\
                            .shuffle(len(v_desc_word_ids), reshuffle_each_iteration=True)\
                            .batch(valid_batch_size, drop_remainder=True)\
                            .repeat()

learning_scheduler = create_learning_rate_scheduler(max_learn_rate=1e-3,
                                                    end_learn_rate=1e-7,
                                                    warmup_epoch_count=10,
                                                    total_epoch_count=epochs)

load_checkpoint_path = "" # f"{DATA_PATH}/model_checkpoints/best_models/{MODEL_TYPE}/cp.ckpt"
if load_checkpoint_path:
    print("Load model weights:", load_checkpoint_path)
    train_model.load_weights(load_checkpoint_path)

train_hist = train_model.fit(train_data,
                           epochs=epochs,
                           validation_data=valid_data,
                           callbacks=[tb_callback, cp_callback, learning_scheduler],
                           steps_per_epoch=train_steps_per_epoch,
                           validation_steps=valid_steps_per_epoch)

with open(f"/home/vkarpov/train_history_{MODEL_TYPE}_{current_time}.pkl", 'wb') as f:
    pickle.dump(train_hist.history, f)

eval_flag= True
if eval_flag:
    t_desc_word_ids, t_desc_input_mask, t_desc_segment_ids, t_sc_word_ids, t_sc_input_mask, t_sc_segment_ids = \
        generate_inputs(scope='test', n_splits=1)[:-1]
    test_batch_size = 50
    test_samples = len(t_desc_word_ids)
    test_steps_per_epoch = test_samples // test_batch_size
    test_data = tf.data.Dataset.from_tensor_slices(((t_desc_word_ids,
                                                     t_sc_word_ids),
                                                    np.ones((len(t_desc_word_ids), 1)))) \
        .shuffle(len(t_desc_word_ids), reshuffle_each_iteration=True) \
        .batch(test_batch_size, drop_remainder=True) \
        .repeat()

    write_hdf5_data(t_sc_word_ids, 't_sc_word_ids')
    # write_hdf5_data(t_desc_word_ids, 't_desc_word_ids')
    # write_hdf5_data(t_desc_input_mask, 't_desc_input_mask')

    train_model.compile(loss=cos_loss, optimizer=optimizer,
                        metrics=[mrr, frank, relevantat1, relevantat5, relevantat10])

    eval_res = train_model.evaluate(test_data, steps=test_steps_per_epoch)
    print(np.round(eval_res, 2))