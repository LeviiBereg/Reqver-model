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
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from dpu_utils.mlutils import Vocabulary
from bert.tokenization import FullTokenizer
from gensim.models import KeyedVectors as word2vec
from sklearn.model_selection import train_test_split

MODEL_TYPE = 'cnn'
LANGUAGE = "java" #"python"
DATA_PATH = "/home/vkarpov"
DATA_FOLDER = f"{LANGUAGE}/short"
TRAIN_FILE  = f"{LANGUAGE}_train_0.jsonl"
TEST_FILE   = f"{LANGUAGE}_test_0.jsonl"
VALID_FILE  = f"{LANGUAGE}_valid_0.jsonl"

use_cols = ["code", "docstring"]


def read_data(scope, n_splits):
    def get_file(scope, split_ind):
        return pd.read_json(f"{DATA_PATH}/{DATA_FOLDER}/{LANGUAGE}_{scope}_{split_ind}.jsonl", lines=True)[use_cols]

    train_df = get_file(scope, split_ind=0)
    for i in range(1, n_splits):
        train_df = train_df.append(get_file(scope, split_ind=i), ignore_index=True)
    return train_df

model_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"
bert_layer = tfh.KerasLayer(model_url, trainable=False)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)

# EMBEDDINGS_FOLDER = "source-code-embeddings"
# TOKEN_EMBEDDINGS  = "token_vecs.txt"
# TARGET_EMBEDDINGS = "target_vecs.txt"
#
# vectors_text_path = f'{DATA_PATH}/{EMBEDDINGS_FOLDER}/{TOKEN_EMBEDDINGS}'
# pretrained_sc_emb = word2vec.load_word2vec_format(vectors_text_path, binary=False)


def cleaning(text):
    '''Performs cleaning of text of unwanted symbols,
    excessive spaces and transfers to lower-case
    '''

    # {@link FaultMessageResolver} => link
    text = re.sub(r"\{?@(\w+)\s+\w+\}?", r'\1', text)
    text = re.sub(r'\s+', " ", text)

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


# def generate_sc_input(sc_inputs, emb_model, max_seq_length):
#     def word_to_index(word):
#         word_val = emb_model.vocab.get(word, None)
#         word_index = word_val.index if word_val else None
#         return word_index
#
#     input_ids = [[word_to_index(word) for word in sc_input[:max_seq_length] if word in emb_model.vocab.keys()] \
#                  for sc_input in sc_inputs]
#     input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids,
#                                                               dtype='int32',
#                                                               maxlen=max_seq_length,
#                                                               padding='post',
#                                                               truncating='post')
#     return input_ids

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


def tokenize_java_code(cstr, stop_tokens=None):
    if not stop_tokens:
        stop_tokens = []
    return [token for plain_token in javalang.tokenizer.tokenize(cstr) \
            if not plain_token.value in stop_tokens \
            for token in split_java_token(plain_token.value, camel_case=True, split_char='_')]


def generate_sc_input(sc_inputs, max_seq_length, max_vocab_size=10000, use_vocab=None):

    stop_tokens = {'{', '}', ';', ',', '(', ')', '.'}
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


def generate_inputs(scope='train', n_splits=1, use_vocab=None):
    pddf = read_data(scope=scope, n_splits=n_splits)
    valid_inds = check_and_fix_code_validity(pddf)
    pddf = pddf[valid_inds]
    pddf.docstring = pddf.docstring.apply(cleaning)

    desc_word_ids, desc_input_mask, desc_segment_ids = generate_bert_input(pddf.docstring, desc_max_seq_length)
    sc_vocab, sc_tok_ids = generate_sc_input(pddf.code,
                                             sc_max_seq_length,
                                             sc_max_vocab_size,
                                             use_vocab=use_vocab)
    return desc_word_ids, desc_input_mask, desc_segment_ids, sc_tok_ids, pddf, sc_vocab


def write_hdf5_data(dataset, dataset_name, data_folder='preprocessed_data'):
    with h5py.File(f'{DATA_PATH}/{data_folder}/{MODEL_TYPE}_encoder/{dataset_name}.h5', 'w') as hf:
        hf.create_dataset(dataset_name, data=dataset)


def read_hdf5_data(dataset_name, data_folder='preprocessed_data', start_index=0, end_index=-1):
    with h5py.File(f'{DATA_PATH}/{data_folder}/{MODEL_TYPE}_encoder/{dataset_name}.h5', "r") as f:
        dataset = f[dataset_name]
        end_index = end_index if end_index > 0 else dataset.size
        res = dataset[start_index:end_index]
    return res

# sc_max_seq_length = 256 # train_df.code_tokens.apply(len).quantile(0.9) == 225
# sc_vocab_size = len(pretrained_sc_emb.vocab)

sc_max_seq_length = 180 # train_df.code_tokens.apply(len).quantile(0.9) == 225
sc_max_vocab_size = 15000

desc_max_seq_length = 180 # 0.95 quantile == 178

# generate_data_flag = False
# if generate_data_flag:
#     # acquire tokenized source code and plain docstrings.
#     # BERT uses its own 'FullTokenizer' for inputs.
#     train_df = read_data(scope='train', n_splits=16)
#     train_df.docstring = train_df.docstring.apply(cleaning)
#     print("Dataset size", len(train_df))
#
#     sc_ids = generate_sc_input(train_df.code_tokens, pretrained_sc_emb, sc_max_seq_length)
#     desc_word_ids, desc_input_mask, desc_segment_ids = generate_desc_input(train_df.docstring, desc_max_seq_length)
#
#     assert np.all((desc_word_ids > 0).sum(axis=1) == desc_input_mask.sum(axis=1)), 'wrong bert input mask'
#     assert desc_word_ids.shape == desc_input_mask.shape, 'bert inputs shape mismatch'
#     assert desc_word_ids.shape == desc_segment_ids.shape, 'bert inputs shape mismatch'
#     assert len(desc_word_ids) == len(sc_ids), 'nl and sc branches inputs mismatch'
#
#     write_hdf5_data(sc_ids, 'sc_ids', data_folder='preprocessed_data/cnn_encoder')
#     write_hdf5_data(desc_word_ids, 'desc_word_ids', data_folder='preprocessed_data/cnn_encoder')
#     write_hdf5_data(desc_input_mask, 'desc_input_mask', data_folder='preprocessed_data/cnn_encoder')
# else:
#     n_samples = 330000
#
#     sc_ids = read_hdf5_data('sc_ids', end_index=n_samples, data_folder='preprocessed_data/cnn_encoder')
#     desc_word_ids = read_hdf5_data('desc_word_ids', end_index=n_samples, data_folder='preprocessed_data/cnn_encoder')
#     desc_input_mask = read_hdf5_data('desc_input_mask', end_index=n_samples, data_folder='preprocessed_data/cnn_encoder')
#     desc_segment_ids = np.zeros(desc_word_ids.shape, dtype=np.int32)
#
#     sc_ids = np.array([tok_lst[:sc_max_seq_length] for tok_lst in sc_ids])
#
#     print("Dataset size", len(sc_ids))

generate_data_flag = False
if generate_data_flag:

    desc_word_ids, desc_input_mask, desc_segment_ids, sc_tok_ids, train_df, sc_vocab = \
        generate_inputs(scope='train', n_splits=16, use_vocab=None)

    sc_vocab_size = len(sc_vocab.token_to_id)
    print("Train dataset size", len(train_df))

    v_desc_word_ids, v_desc_input_mask, v_desc_segment_ids, v_sc_tok_ids = \
        generate_inputs(scope='valid', n_splits=1, use_vocab=sc_vocab)[:-2]

    print("Validation dataset size", len(v_desc_word_ids))
    #     sc_ids = generate_sc_input(train_df.code_tokens, pretrained_sc_emb, sc_max_seq_length)
    #     desc_word_ids, desc_input_mask, desc_segment_ids = generate_bert_input(train_df.docstring, desc_max_seq_length)

    assert np.all((desc_word_ids > 0).sum(axis=1) == desc_input_mask.sum(axis=1)), 'wrong bert input mask'
    assert desc_word_ids.shape == desc_input_mask.shape, 'bert inputs shape mismatch'
    assert desc_word_ids.shape == desc_segment_ids.shape, 'bert inputs shape mismatch'
    assert len(desc_word_ids) == len(sc_tok_ids), 'nl and sc branches inputs mismatch'

    with open(f"cs_{MODEL_TYPE}_vocab_{sc_vocab_size}.pkl", 'wb') as pickle_file:
        pickle.dump(sc_vocab, pickle_file)

    write_hdf5_data(sc_tok_ids, 'sc_tok_ids')
    write_hdf5_data(desc_word_ids, 'desc_word_ids')
    write_hdf5_data(desc_input_mask, 'desc_input_mask')

    write_hdf5_data(v_sc_tok_ids, 'v_sc_tok_ids')
    write_hdf5_data(v_desc_word_ids, 'v_desc_word_ids')
    write_hdf5_data(v_desc_input_mask, 'v_desc_input_mask')
else:
    n_samples = -1
    n_val_samples = -1
    with open(f"cs_{MODEL_TYPE}_vocab_11264.pkl", 'rb') as pickle_file:
        sc_vocab = pickle.load(pickle_file)

    sc_vocab_size = len(sc_vocab.token_to_id)

    sc_tok_ids = read_hdf5_data('sc_tok_ids', end_index=n_samples)
    desc_word_ids = read_hdf5_data('desc_noparam_word_ids', end_index=n_samples)
    desc_input_mask = read_hdf5_data('desc_noparam_input_mask', end_index=n_samples)
    desc_segment_ids = np.zeros(desc_word_ids.shape, dtype=np.int32)

    v_sc_tok_ids = read_hdf5_data('v_sc_tok_ids', end_index=n_val_samples)
    v_desc_word_ids = read_hdf5_data('v_desc_noparam_word_ids', end_index=n_val_samples)
    v_desc_input_mask = read_hdf5_data('v_desc_noparam_input_mask', end_index=n_val_samples)
    v_desc_segment_ids = np.zeros(v_desc_word_ids.shape, dtype=np.int32)

    print("Train dataset size", len(sc_tok_ids))
    print("Validation dataset size", len(v_sc_tok_ids))

dense_units = 400

input_word_ids = tf.keras.layers.Input(shape=(desc_max_seq_length,),
                                       dtype=tf.int32,
                                       name="desc_input_word_ids")
input_mask  = tf.keras.layers.Input(shape=(desc_max_seq_length,),
                                   dtype=tf.int32,
                                   name="desc_input_mask")
segment_ids = tf.keras.layers.Input(shape=(desc_max_seq_length,),
                                    dtype=tf.int32,
                                    name="desc_segment_ids")

desc_dense = tf.keras.layers.Dense(dense_units, activation='tanh', name="desc_dense")

sc_model = 'convolutional' # 'lstm'
sc_emb_size = 128
sc_dropout_rate = 0.25
conv_kernel_sizes = [2,3,5]
conv_n_filters = 100

sc_input_tok_ids = tf.keras.layers.Input(shape=(sc_max_seq_length,),
                                       dtype=tf.int32,
                                       name="sc_tok_ids")

sc_tok_embedding = tf.keras.layers.Embedding(sc_vocab_size,
                                               sc_emb_size,
                                               mask_zero=True,
                                               # len(model.vocab), model.vector_size, weights=[model.vectors],
                                               name='sc_tok_embedding')

sc_convs = []
sc_max_pools = []
sc_conv_dropouts = []
sc_conv_flatten = tf.keras.layers.Flatten()
for kernel_size in conv_kernel_sizes:
    sc_convs.append(tf.keras.layers.Conv1D(conv_n_filters, kernel_size, activation='relu', name=f'conv_{kernel_size}'))
    sc_max_pools.append(tf.keras.layers.MaxPooling1D(sc_max_seq_length - kernel_size + 1, 1, name=f'max_pool_{kernel_size}'))
    sc_conv_dropouts.append(tf.keras.layers.Dropout(sc_dropout_rate, name=f'dropout_{kernel_size}'))

sc_dense = tf.keras.layers.Dense(dense_units, activation='tanh', name="sc_dense")

pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
desc_output = desc_dense(pooled_output)

desc_embedding_model = tf.keras.Model(inputs=[input_word_ids,
                                              input_mask,
                                              segment_ids],
                                      outputs=desc_output,
                                      name='desc_embedding_model')

sc_embedded_input = sc_tok_embedding(sc_input_tok_ids) # (batch_size, sc_max_seq_length, emb_vec_size)

conv_outputs = []
for sc_conv, sc_max_pool, sc_dropout in zip(sc_convs, sc_max_pools, sc_conv_dropouts):
    sc_conv_out = sc_conv(sc_embedded_input)
    sc_conv_out = sc_max_pool(sc_conv_out)
    sc_conv_out = sc_dropout(sc_conv_out)
    conv_outputs.append(sc_conv_out)
sc_output = tf.concat(conv_outputs, 2) # (batch_size, 1, n_convs * conv_n_filters)
sc_output = sc_conv_flatten(sc_output) # (batch_size, n_convs * conv_n_filters)
sc_output = sc_dense(sc_output)

code_embedding_model = tf.keras.Model(inputs=sc_input_tok_ids,
                                      outputs=sc_output,
                                      name=f'code_{MODEL_TYPE}_embedding_model')

norm_desc = tf.norm(desc_output, axis=-1, keepdims=True) + 1e-10
norm_sc   = tf.norm(sc_output, axis=-1, keepdims=True)   + 1e-10
cos_similarity = tf.matmul(desc_output/norm_desc,
                            sc_output/norm_sc,
                            transpose_a=False,
                            transpose_b=True,
                            name='code_query_cooccurrence_logits')  # (batch_size, batch_size)

train_model = tf.keras.Model(inputs=[input_word_ids,
                                     input_mask,
                                     segment_ids,
                                     sc_input_tok_ids],
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
log_dir=f"{DATA_PATH}/logs/fit/{MODEL_TYPE}/" + current_time
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

# splitted_data = train_test_split(desc_word_ids, desc_input_mask, desc_segment_ids, sc_ids)
# train_desc_word_ids, test_desc_word_ids = splitted_data[:2]
# train_desc_input_mask, test_desc_input_mask = splitted_data[2:4]
# train_desc_segment_ids, test_desc_segment_ids = splitted_data[4:6]
# train_sc_ids, test_sc_ids = splitted_data[6:8]

batch_size = 256
valid_batch_size = 256
train_samples = len(desc_word_ids) # or change to desc_word_ids
valid_samples = len(v_desc_word_ids)
train_steps_per_epoch = train_samples // batch_size
valid_steps_per_epoch = valid_samples // valid_batch_size
epochs = 20

train_data = tf.data.Dataset.from_tensor_slices(((desc_word_ids,
                                                  desc_input_mask,
                                                  desc_segment_ids,
                                                  sc_tok_ids),
                                                 np.ones((len(desc_word_ids),1))))\
                            .shuffle(len(desc_word_ids), reshuffle_each_iteration=True)\
                            .batch(batch_size, drop_remainder=True)\
                            .repeat()
valid_data = tf.data.Dataset.from_tensor_slices(((v_desc_word_ids,
                                                  v_desc_input_mask,
                                                  v_desc_segment_ids,
                                                  v_sc_tok_ids),
                                                 np.ones((len(v_desc_word_ids),1))))\
                            .shuffle(len(v_desc_word_ids), reshuffle_each_iteration=True)\
                            .batch(valid_batch_size, drop_remainder=True)\
                            .repeat()

load_checkpoint_path = ""#f"./model_checkpoints/best_models/{MODEL_TYPE}/20200524-181110/cp.ckpt"
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
