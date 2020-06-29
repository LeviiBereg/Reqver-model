import re
import os
import time
import h5py
import string
import pickle
import datetime
import javalang
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as tfh
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from dpu_utils.mlutils import Vocabulary
from bert.tokenization import FullTokenizer
from gensim.models import KeyedVectors as word2vec
from sklearn.model_selection import train_test_split

MODEL_TYPE = 'bow'
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

def cleaning(text):
    '''Performs cleaning of text of unwanted symbols,
    excessive spaces and transfers to lower-case
    '''
    # {@link FaultMessageResolver} => link
    text = re.sub(r"\{?@(\w+)\s+\S+\}?", r'\1', text)
    # delete XML tags
    text = re.sub(r'<[\/a-zA-Z]+>', "", text)
    # remove punctuation
    text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+', " ", text)
    # remove excessive spaces
    text = re.sub(r'\s+', " ", text)

    text = ''.join(character for character in text if character in string.printable)
    text = text.lower().strip()

    return text


def check_and_fix_code_validity(train_df):
    valid_inds = []
    for i, s_code in enumerate(train_df.code):
        try:
            javalang.parse.parse_member_signature(s_code)
            valid_inds.append(True)
        except javalang.parser.JavaSyntaxError:
            try:
                modified_s_code = s_code + '\n}'
                javalang.parse.parse_member_signature(modified_s_code)
                valid_inds.append(True)
                train_df.code[i] = modified_s_code
            except javalang.parser.JavaSyntaxError:
                valid_inds.append(False)
    return valid_inds


def split_java_token(cstr, camel_case=True, split_char='_'):
    res_split = []

    if camel_case:
        res_split = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', cstr)).split()

    if split_char:
        char_splt = []
        if not camel_case:
            res_split = [cstr]
        for token in res_split:
            char_splt += token.split(split_char)
        res_split = char_splt
    return [token for token in res_split if len(token) > 0]


def tokenize_java_code(cstr, stop_tokens=[]):
    return [token for plain_token in javalang.tokenizer.tokenize(cstr) \
            if not plain_token.value in stop_tokens \
            for token in split_java_token(plain_token.value, camel_case=True, split_char='_')]


def generate_desc_input(desc_inputs, max_seq_length, max_vocab_size=10000, use_vocab=None):
    input_ids = [desc_input.split()[:max_seq_length] for desc_input in desc_inputs]

    all_ids = [input_ids]
    all_tokens = []
    for token_list in all_ids:
        for lst in token_list:
            all_tokens += lst

    if use_vocab:
        desc_vocab = use_vocab
    else:
        desc_vocab = Vocabulary.create_vocabulary(all_tokens,
                                                  max_size=max_vocab_size,
                                                  count_threshold=int(len(desc_inputs) * 0.00025),
                                                  add_pad=True)

    input_ids = [desc_vocab.get_id_or_unk_multiple(desc_input, pad_to_size=max_seq_length) for desc_input in input_ids]
    return desc_vocab, input_ids


def generate_sc_input(sc_inputs, max_seq_length, max_vocab_size=10000, use_vocab=None):
    stop_tokens = {'{', '}', ';', ':', ',', '(', ')', '.'}
    input_ids = [tokenize_java_code(sc_input, stop_tokens)[:max_seq_length] \
                 for sc_input in sc_inputs]

    all_ids = [input_ids]
    all_tokens = []
    for token_list in all_ids:
        for lst in token_list:
            all_tokens += lst

    if use_vocab:
        sc_vocab = use_vocab
    else:
        sc_vocab = Vocabulary.create_vocabulary(all_tokens,
                                                max_size=max_vocab_size,
                                                count_threshold=int(len(sc_inputs) * 0.00025),
                                                add_pad=True)

    input_ids = [sc_vocab.get_id_or_unk_multiple(sc_input, pad_to_size=max_seq_length) for sc_input in input_ids]
    return sc_vocab, input_ids


def write_hdf5_data(dataset, dataset_name, data_folder='preprocessed_data'):
    with h5py.File(f'{DATA_PATH}/{data_folder}/{MODEL_TYPE}_encoder/{dataset_name}.h5', 'w') as hf:
        hf.create_dataset(dataset_name, data=dataset)


def read_hdf5_data(dataset_name, data_folder='preprocessed_data', start_index=0, end_index=-1):
    with h5py.File(f'{DATA_PATH}/{data_folder}/{MODEL_TYPE}_encoder/{dataset_name}.h5', "r") as f:
        dataset = f[dataset_name]
        end_index = end_index if end_index > 0 else dataset.size
        res = dataset[start_index:end_index]
    return res


def generate_inputs(scope='train', n_splits=1, use_desc_vocab=None, use_sc_vocab=None):
    pddf = read_data(scope=scope, n_splits=n_splits)
    valid_inds = check_and_fix_code_validity(pddf)
    pddf = pddf[valid_inds]
    pddf.docstring = pddf.docstring.apply(cleaning)

    desc_vocab, desc_tok_ids = generate_desc_input(pddf.docstring,
                                                   desc_max_seq_length,
                                                   desc_max_vocab_size,
                                                   use_vocab=use_desc_vocab)
    sc_vocab, sc_tok_ids = generate_sc_input(pddf.code,
                                             sc_max_seq_length,
                                             sc_max_vocab_size,
                                             use_vocab=use_sc_vocab)
    return desc_tok_ids, sc_tok_ids, pddf, desc_vocab, sc_vocab


sc_max_seq_length = 180
sc_max_vocab_size = 15000

desc_max_seq_length = 100  # 0.95 quantile == 178
desc_max_vocab_size = 15000

generate_data_flag = False
if generate_data_flag:
    # acquire tokenized source code and plain docstrings.
    # BERT uses its own 'FullTokenizer' for inputs.

    desc_tok_ids, sc_tok_ids, train_df, desc_vocab, sc_vocab = \
        generate_inputs(scope='train',
                        n_splits=16,
                        use_desc_vocab=None,
                        use_sc_vocab=None)

    sc_vocab_size = len(sc_vocab.token_to_id)
    desc_vocab_size = len(desc_vocab.token_to_id)
    print("Train dataset size", len(train_df))

    v_desc_tok_ids, v_sc_tok_ids = generate_inputs(scope='valid', n_splits=1, use_desc_vocab=desc_vocab,
                                                   use_sc_vocab=sc_vocab)[:-3]
    print("Validation dataset size", len(v_desc_tok_ids))

    assert len(desc_tok_ids) == len(sc_tok_ids), 'train inputs shape mismatch'
    assert len(v_desc_tok_ids) == len(v_sc_tok_ids), 'valid inputs shape mismatch'

    with open(f"cs_{MODEL_TYPE}_vocab_{sc_vocab_size}.pkl", 'wb') as pickle_file:
        pickle.dump(sc_vocab, pickle_file)
    with open(f"desc_{MODEL_TYPE}_vocab_{desc_vocab_size}.pkl", 'wb') as pickle_file:
        pickle.dump(desc_vocab, pickle_file)

    write_hdf5_data(desc_tok_ids, 'desc_tok_ids')
    write_hdf5_data(sc_tok_ids, 'sc_tok_ids')

    write_hdf5_data(v_desc_tok_ids, 'v_desc_tok_ids')
    write_hdf5_data(v_sc_tok_ids, 'v_sc_tok_ids')
else:
    n_samples = -1
    n_val_samples = -1

    with open(f"cs_{MODEL_TYPE}_vocab_10704.pkl", 'rb') as pickle_file:
        sc_vocab = pickle.load(pickle_file)
    with open(f"desc_{MODEL_TYPE}_vocab_5394.pkl", 'rb') as pickle_file:
        desc_vocab = pickle.load(pickle_file)

    sc_vocab_size = len(sc_vocab.token_to_id)
    desc_vocab_size = len(desc_vocab.token_to_id)

    desc_tok_ids = read_hdf5_data('desc_tok_ids', end_index=n_samples)
    sc_tok_ids = read_hdf5_data('sc_tok_ids', end_index=n_samples)

    v_desc_tok_ids = read_hdf5_data('v_desc_tok_ids', end_index=n_val_samples)
    v_sc_tok_ids = read_hdf5_data('v_sc_tok_ids', end_index=n_val_samples)

    print("Train dataset size", len(sc_tok_ids))
    print("Validation dataset size", len(v_sc_tok_ids))

dense_units = 400

desc_emb_size = 128
desc_dropout_rate = 0.25

desc_input_ids = tf.keras.layers.Input(shape=(desc_max_seq_length,),
                                           dtype=tf.int32,
                                           name='desc_input_ids')

desc_tok_embedding = tf.keras.layers.Embedding(desc_vocab_size,
                                               desc_emb_size,
                                               mask_zero=True,
                                               name='desc_tok_embedding')

desc_tok_dropout = tf.keras.layers.Dropout(desc_dropout_rate, name='desc_tok_dropout')
desc_tok_maxpool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=1, keepdims=False),
                                 output_shape=lambda x: (x[0], x[1]),
                                 name='desc_tok_maxpool')
desc_tok_maxpool_activation = tf.keras.layers.Activation('tanh', name='desc_tok_maxpool_activation')
desc_dense = tf.keras.layers.Dense(dense_units, activation='tanh',name='desc_dense')

sc_emb_size = 128
sc_dropout_rate = 0.25

sc_input_ids = tf.keras.layers.Input(shape=(sc_max_seq_length,),
                                           dtype=tf.int32,
                                           name='sc_input_ids')

sc_tok_embedding = tf.keras.layers.Embedding(sc_vocab_size,
                                               sc_emb_size,
                                               mask_zero=True,
                                               name='sc_tok_embedding')

sc_tok_dropout   = tf.keras.layers.Dropout(sc_dropout_rate, name='sc_tok_dropout')
sc_tok_maxpool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=1, keepdims=False),
                                 output_shape=lambda x: (x[0], x[1]),
                                 name='sc_tok_maxpool')
sc_tok_maxpool_activation = tf.keras.layers.Activation('tanh', name='sc_tok_maxpool_activation')
sc_dense = tf.keras.layers.Dense(dense_units, activation='tanh',name='sc_dense')

desc_tok_embedded = desc_tok_embedding(desc_input_ids)
desc_tok_embedded = desc_tok_dropout(desc_tok_embedded)
desc_tok_embedded = desc_tok_maxpool(desc_tok_embedded)
desc_tok_embedded = desc_tok_maxpool_activation(desc_tok_embedded)
desc_output = desc_dense(desc_tok_embedded)

desc_embedding_model = tf.keras.Model(inputs=desc_input_ids,
                                      outputs=desc_output,
                                      name=f'desc_{MODEL_TYPE}_embedding_model')

sc_tok_embedded   = sc_tok_embedding(sc_input_ids)
sc_tok_embedded   = sc_tok_dropout(sc_tok_embedded)
sc_tok_embedded = sc_tok_maxpool(sc_tok_embedded)
sc_tok_embedded = sc_tok_maxpool_activation(sc_tok_embedded)
sc_output = sc_dense(sc_tok_embedded)

code_embedding_model = tf.keras.Model(inputs=sc_input_ids,
                                      outputs=sc_output,
                                      name=f'code_{MODEL_TYPE}_embedding_model')

norm_desc = tf.norm(desc_output, axis=-1, keepdims=True) + 1e-10
norm_sc   = tf.norm(sc_output, axis=-1, keepdims=True)   + 1e-10
cos_similarity = tf.matmul(desc_output/norm_desc,
                            sc_output/norm_sc,
                            transpose_a=False,
                            transpose_b=True,
                            name='code_query_cooccurrence_logits')  # (batch_size, batch_size)

train_model = tf.keras.Model(inputs=[desc_input_ids,
                                     sc_input_ids],
                             outputs=cos_similarity,
                             name=f'train_{MODEL_TYPE}_model')

def cos_loss(dummy, cosine_similarities):
    loss_margin = 0.5
    neg_matrix = tf.linalg.diag(-tf.linalg.diag_part(cosine_similarities))
    per_sample_loss = tf.maximum(1e-6, loss_margin
                                     - tf.linalg.diag_part(cosine_similarities)
                                     + tf.reduce_mean(tf.nn.relu(cosine_similarities + neg_matrix), axis=-1))

    loss = tf.reduce_mean(per_sample_loss)
    return loss

def mrr(dummy, cosine_similarities):
    # extract the logits from the diagonal of the matrix, which are the logits corresponding to the ground-truth
    correct_scores = tf.linalg.diag_part(cosine_similarities)
    # compute how many queries have bigger logits than the ground truth (the diagonal) -> which will be incorrectly ranked
    compared_scores = cosine_similarities >= tf.expand_dims(correct_scores, axis=-1)
    # for each row of the matrix (query), sum how many logits are larger than the ground truth
    # ...then take the reciprocal of that to get the MRR for each individual query (you will need to take the mean later)
    mrr = 1 / tf.reduce_sum(tf.cast(compared_scores, dtype=tf.float32), axis=1)
    return mrr


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
train_samples = len(desc_tok_ids)
valid_samples = len(v_desc_tok_ids)
train_steps_per_epoch = train_samples // batch_size
valid_steps_per_epoch = valid_samples // valid_batch_size
epochs = 15

train_data = tf.data.Dataset.from_tensor_slices(((desc_tok_ids,
                                                  sc_tok_ids),
                                                 np.ones((len(desc_tok_ids),1))))\
                            .shuffle(len(desc_tok_ids), reshuffle_each_iteration=True)\
                            .batch(batch_size, drop_remainder=True)\
                            .repeat()
valid_data = tf.data.Dataset.from_tensor_slices(((v_desc_tok_ids,
                                                  v_sc_tok_ids),
                                                 np.ones((len(v_desc_tok_ids),1))))\
                            .shuffle(len(v_desc_tok_ids), reshuffle_each_iteration=True)\
                            .batch(valid_batch_size, drop_remainder=True)\
                            .repeat()

load_checkpoint_path = "" # f"{DATA_PATH}/model_checkpoints/best_models/{MODEL_TYPE}/cp.ckpt"
if load_checkpoint_path:
    print("Load model weights:", load_checkpoint_path)
    train_model.load_weights(load_checkpoint_path)

train_hist = train_model.fit(train_data,
                           epochs=epochs,
                           validation_data=valid_data,
                           callbacks=[tb_callback, cp_callback],
                           steps_per_epoch=train_steps_per_epoch,
                           validation_steps=valid_steps_per_epoch)

with open(f"/home/vkarpov/train_history_{current_time}.pkl", 'wb') as f:
    pickle.dump(train_hist.history, f)
