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
# from bert.tokenization import FullTokenizer
from gensim.models import KeyedVectors as word2vec
from sklearn.model_selection import train_test_split

import bert
FullTokenizer = bert.bert_tokenization.FullTokenizer

MODEL_TYPE = 'api'
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
bert_layer = tfh.KerasLayer(model_url, trainable=False)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
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


def tokenize_java_code(cstr):
    stop_tokens = {'{', '}', ';'}
    return [token for plain_token in javalang.tokenizer.tokenize(cstr) \
            if not plain_token.value in stop_tokens \
            for token in split_java_token(plain_token.value, camel_case=True, split_char='_')]


def get_api_sequence(cstr, split_api_tokens=False):
    def find_method(method_node, filter):
        sub_api_seq = []
        for node in method_node.arguments:
            if isinstance(node, javalang.tree.MethodInvocation):
                api = [filter.get(node.qualifier, node.qualifier),
                       node.member]
                sub_api_seq.append(api)
                sub_api_seq.extend(find_method(node, filter))

            if isinstance(node, javalang.tree.ClassCreator):
                api = [get_last_sub_type(node.type).name, 'new']
                sub_api_seq.append(api)
                sub_api_seq.extend(find_method(node, filter))
        return sub_api_seq

    def check_selectors(node, s_filter):
        select_api_seq = []
        if node.selectors is not None:
            for sel in node.selectors:
                if isinstance(sel, javalang.tree.MethodInvocation):
                    if node.qualifier is None:
                        select_api_seq.append([get_last_sub_type(node.type).name, sel.member])
                    else:
                        select_api_seq.append(
                            [s_filter.get(node.qualifier, node.qualifier),
                             sel.member])
        return select_api_seq

    def get_last_sub_type(node):
        if not 'sub_type' in node.attrs or not node.sub_type:
            return node
        else:
            return get_last_sub_type(node.sub_type)

    api_seq = []
    tree = javalang.parse.parse_member_signature(cstr)
    identifier_filter = {}
    this_selectors = []
    for _, node in tree:
        if isinstance(node, javalang.tree.FormalParameter):
            identifier_filter[node.name] = get_last_sub_type(node.type).name

        if isinstance(node, javalang.tree.LocalVariableDeclaration):
            for dec in node.declarators:
                identifier_filter[dec.name] = get_last_sub_type(node.type).name

        if isinstance(node, javalang.tree.ClassCreator):
            api = [get_last_sub_type(node.type).name, 'new']
            api_seq.append(api)
            api_seq.extend(check_selectors(node, identifier_filter))

        if isinstance(node, javalang.tree.MethodInvocation):

            if node.qualifier is None:
                if len(api_seq) != 0:
                    node.qualifier = api_seq[-1][0]
                elif len(this_selectors) != 0:
                    try:
                        node_pos = this_selectors.index(node)
                        if isinstance(this_selectors[node_pos - 1], javalang.tree.MemberReference):
                            node.qualifier = this_selectors[node_pos - 1].member
                    except ValueError:
                        node.qualifier = ''

            sub_api_seq = find_method(node, identifier_filter)
            sub_api_seq.append(
                [identifier_filter.get(node.qualifier, node.qualifier),
                 node.member])
            api_seq.extend(sub_api_seq)
            api_seq.extend(check_selectors(node, identifier_filter))

        if isinstance(node, javalang.tree.This):
            this_selectors = node.selectors
    api_seq = [item for pairs in api_seq for item in pairs if item]
    if split_api_tokens:
        api_seq = [token for item in api_seq for token in split_java_token(item)]
    return api_seq

def generate_desc_input(text, max_seq_length):

    tokenized_text = [["[CLS]"] + tokenizer.tokenize(seq)[:max_seq_length-2] + ["[SEP]"] for seq in text]
    input_ids   = [tokenizer.convert_tokens_to_ids(tokens_seq) for tokens_seq in tokenized_text]
    input_mask  = [[1] * len(input_seq) for input_seq in input_ids]
    segment_ids = [[0] * max_seq_length for _ in range(len(input_ids))]
    input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids, maxlen=max_seq_length, padding='post', truncating='post')
    input_mask = tf.keras.preprocessing.sequence.pad_sequences(input_mask, maxlen=max_seq_length, padding='post', truncating='post')
    segment_ids = tf.keras.preprocessing.sequence.pad_sequences(segment_ids, maxlen=max_seq_length, padding='post', truncating='post')

    return input_ids, input_mask, segment_ids


def generate_sc_input(sc_inputs, max_fname_length, max_api_length, max_seq_length, max_vocab_size=10000,
                      use_vocab=None):
    sc_fname_ids = [split_java_token(javalang.parse.parse_member_signature(sc_input).name)[:max_fname_length] \
                    for sc_input in sc_inputs]

    sc_api_ids = [get_api_sequence(sc_input, split_api_tokens=True)[:max_api_length] \
                  for sc_input in sc_inputs]

    sc_tok_ids = [tokenize_java_code(sc_input)[:max_seq_length] \
                  for sc_input in sc_inputs]

    all_ids = [sc_fname_ids, sc_api_ids, sc_tok_ids]
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

    sc_fname_ids = [sc_vocab.get_id_or_unk_multiple(sc_input, pad_to_size=max_fname_length) for sc_input in
                    sc_fname_ids]
    sc_api_ids = [sc_vocab.get_id_or_unk_multiple(sc_input, pad_to_size=max_api_length) for sc_input in sc_api_ids]
    sc_tok_ids = [sc_vocab.get_id_or_unk_multiple(sc_input, pad_to_size=max_seq_length) for sc_input in sc_tok_ids]

    return sc_vocab, sc_fname_ids, sc_api_ids, sc_tok_ids


def write_hdf5_data(dataset, dataset_name, data_folder='preprocessed_data'):
    with h5py.File(f'{DATA_PATH}/{data_folder}/{MODEL_TYPE}_encoder/{dataset_name}.h5', 'w') as hf:
        hf.create_dataset(dataset_name, data=dataset)


def read_hdf5_data(dataset_name, data_folder='preprocessed_data', start_index=0, end_index=-1):
    with h5py.File(f'{DATA_PATH}/{data_folder}/{MODEL_TYPE}_encoder/{dataset_name}.h5', "r") as f:
        dataset = f[dataset_name]
        end_index = end_index if end_index > 0 else dataset.size
        res = dataset[start_index:end_index]
    return res


def generate_inputs(scope='train', n_splits=1, use_vocab=None):
    pddf = read_data(scope=scope, n_splits=n_splits)
    valid_inds = check_and_fix_code_validity(pddf)
    pddf = pddf[valid_inds]
    pddf.docstring = pddf.docstring.apply(cleaning)

    desc_word_ids, desc_input_mask, desc_segment_ids = generate_desc_input(pddf.docstring, desc_max_seq_length)
    sc_vocab, sc_fname_ids, sc_api_ids, sc_tok_ids = generate_sc_input(pddf.code,
                                                                       sc_max_fname_length,
                                                                       sc_max_api_length,
                                                                       sc_max_seq_length,
                                                                       sc_max_vocab_size,
                                                                       use_vocab=use_vocab)
    return desc_word_ids, desc_input_mask, desc_segment_ids, sc_fname_ids, sc_api_ids, sc_tok_ids, pddf, sc_vocab

sc_max_fname_length = 5
sc_max_api_length = 125
sc_max_seq_length = 150
sc_max_vocab_size = 15000

desc_max_seq_length = 180

generate_data_flag = False
if generate_data_flag:

    desc_word_ids, desc_input_mask, desc_segment_ids, sc_fname_ids, sc_api_ids, sc_tok_ids, train_df, sc_vocab = \
        generate_inputs(scope='train', n_splits=16, use_vocab=None)

    sc_vocab_size = len(sc_vocab.token_to_id)
    print("Train dataset size", len(train_df))

    v_desc_word_ids, v_desc_input_mask, v_desc_segment_ids, v_sc_fname_ids, v_sc_api_ids, v_sc_tok_ids = \
        generate_inputs(scope='valid', n_splits=1, use_vocab=sc_vocab)[:-2]
    print("Validation dataset size", len(v_desc_word_ids))

    assert np.all((desc_word_ids > 0).sum(axis=1) == desc_input_mask.sum(axis=1)), 'wrong bert input mask'
    assert desc_word_ids.shape == desc_input_mask.shape, 'bert inputs shape mismatch'
    assert desc_word_ids.shape == desc_segment_ids.shape, 'bert inputs shape mismatch'
    assert len(desc_word_ids) == len(sc_fname_ids), 'nl and sc branches inputs mismatch'

    with open(f"cs_lstm_noparam_vocab_{sc_vocab_size}.pkl", 'wb') as pickle_file:
        pickle.dump(sc_vocab, pickle_file)

    write_hdf5_data(sc_fname_ids, 'sc_noparam_fname_ids')
    write_hdf5_data(sc_api_ids, 'sc_noparam_api_ids')
    write_hdf5_data(sc_tok_ids, 'sc_noparam_tok_ids')
    write_hdf5_data(desc_word_ids, 'desc_noparam_word_ids')
    write_hdf5_data(desc_input_mask, 'desc_noparam_input_mask')

    write_hdf5_data(v_sc_fname_ids, 'v_sc_noparam_fname_ids')
    write_hdf5_data(v_sc_api_ids, 'v_sc_noparam_api_ids')
    write_hdf5_data(v_sc_tok_ids, 'v_sc_noparam_tok_ids')
    write_hdf5_data(v_desc_word_ids, 'v_desc_noparam_word_ids')
    write_hdf5_data(v_desc_input_mask, 'v_desc_noparam_input_mask')
else:

    n_samples = -1
    n_val_samples = -1

    with open("cs_lstm_noparam_vocab_11892.pkl", 'rb') as pickle_file:
        sc_vocab = pickle.load(pickle_file)

    sc_vocab_size = len(sc_vocab.token_to_id)

    sc_fname_ids = read_hdf5_data('sc_noparam_fname_ids', end_index=n_samples)
    sc_api_ids = read_hdf5_data('sc_noparam_api_ids', end_index=n_samples)
    sc_tok_ids = read_hdf5_data('sc_noparam_tok_ids', end_index=n_samples)
    desc_word_ids = read_hdf5_data('desc_noparam_word_ids', end_index=n_samples)
    desc_input_mask = read_hdf5_data('desc_noparam_input_mask', end_index=n_samples)
    desc_segment_ids = np.zeros(desc_word_ids.shape, dtype=np.int32)

    v_sc_fname_ids = read_hdf5_data('v_sc_noparam_fname_ids', end_index=n_val_samples)
    v_sc_api_ids = read_hdf5_data('v_sc_noparam_api_ids', end_index=n_val_samples)
    v_sc_tok_ids = read_hdf5_data('v_sc_noparam_tok_ids', end_index=n_val_samples)
    v_desc_word_ids = read_hdf5_data('v_desc_noparam_word_ids', end_index=n_val_samples)
    v_desc_input_mask = read_hdf5_data('v_desc_noparam_input_mask', end_index=n_val_samples)
    v_desc_segment_ids = np.zeros(v_desc_word_ids.shape, dtype=np.int32)

    print("Train dataset size", len(sc_fname_ids))
    print("Validation dataset size", len(v_sc_fname_ids))

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

sc_lstm_units = 256
sc_model = 'lstm'
sc_emb_size = 128
sc_dropout_rate = 0.25
sc_lstm_rec_dropout_rate = 0.2

sc_input_fname_ids = tf.keras.layers.Input(shape=(sc_max_fname_length,),
                                           dtype=tf.int32,
                                           name='sc_input_fname_ids')
sc_input_api_ids   = tf.keras.layers.Input(shape=(sc_max_api_length,),
                                           dtype=tf.int32,
                                           name='sc_input_api_ids')
sc_input_tok_ids   = tf.keras.layers.Input(shape=(sc_max_seq_length,),
                                           dtype=tf.int32,
                                           name='sc_input_tok_ids')

sc_fname_embedding = tf.keras.layers.Embedding(sc_vocab_size,
                                               sc_emb_size,
                                               mask_zero=False,
                                               name='sc_fname_embedding')
sc_api_embedding   = tf.keras.layers.Embedding(sc_vocab_size,
                                               sc_emb_size,
                                               mask_zero=False,
                                               name='sc_api_embedding')
sc_tok_embedding   = tf.keras.layers.Embedding(sc_vocab_size,
                                               sc_emb_size,
                                               mask_zero=False,
                                               # len(model.vocab), model.vector_size, weights=[model.vectors],
                                               name='sc_tok_embedding')

sc_fname_dropout = tf.keras.layers.Dropout(sc_dropout_rate, name='sc_fname_dropout')
sc_api_dropout   = tf.keras.layers.Dropout(sc_dropout_rate, name='sc_api_dropout')
sc_tok_dropout   = tf.keras.layers.Dropout(sc_dropout_rate, name='sc_tok_dropout')

sc_fname_frnn = tf.keras.layers.LSTM(sc_lstm_units,
                                     recurrent_dropout=sc_lstm_rec_dropout_rate,
                                     return_sequences=True,
                                     name='sc_fname_frnn')
sc_fname_brnn = tf.keras.layers.LSTM(sc_lstm_units,
                                     return_sequences=True,
                                     recurrent_dropout=sc_lstm_rec_dropout_rate,
                                     name='sc_fname_brnn',
                                     go_backwards=True)
sc_api_frnn = tf.keras.layers.LSTM(sc_lstm_units,
                                     recurrent_dropout=sc_lstm_rec_dropout_rate,
                                     return_sequences=True,
                                     name='sc_api_frnn')
sc_api_brnn = tf.keras.layers.LSTM(sc_lstm_units,
                                     return_sequences=True,
                                     recurrent_dropout=sc_lstm_rec_dropout_rate,
                                     name='sc_api_brnn',
                                     go_backwards=True)
sc_tok_frnn = tf.keras.layers.LSTM(sc_lstm_units,
                                     recurrent_dropout=sc_lstm_rec_dropout_rate,
                                     return_sequences=True,
                                     name='sc_tok_frnn')
sc_tok_brnn = tf.keras.layers.LSTM(sc_lstm_units,
                                     return_sequences=True,
                                     recurrent_dropout=sc_lstm_rec_dropout_rate,
                                     name='sc_tok_brnn',
                                     go_backwards=True)

sc_fname_rnn_dropout = tf.keras.layers.Dropout(sc_dropout_rate, name='sc_fname_rnn_dropout')
sc_api_rnn_dropout   = tf.keras.layers.Dropout(sc_dropout_rate, name='sc_api_rnn_dropout')
sc_tok_rnn_dropout   = tf.keras.layers.Dropout(sc_dropout_rate, name='sc_tok_rnn_dropout')

sc_fname_maxpool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=1, keepdims=False),
                                 output_shape=lambda x: (x[0], x[2]),
                                 name='sc_fname_maxpool')
sc_fname_maxpool_concat = tf.keras.layers.Concatenate(name='sc_fname_maxpool_concat')
sc_fname_maxpool_activation = tf.keras.layers.Activation('tanh', name='sc_fname_maxpool_activation')
sc_api_maxpool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=1, keepdims=False),
                                 output_shape=lambda x: (x[0], x[2]),
                                 name='sc_api_maxpool')
sc_api_maxpool_concat = tf.keras.layers.Concatenate(name='sc_api_maxpool_concat')
sc_api_maxpool_activation = tf.keras.layers.Activation('tanh', name='sc_api_maxpool_activation')
sc_tok_maxpool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=1, keepdims=False),
                                 output_shape=lambda x: (x[0], x[2]),
                                 name='sc_tok_maxpool')
sc_tok_maxpool_concat = tf.keras.layers.Concatenate(name='sc_tok_maxpool_concat')
sc_tok_maxpool_activation = tf.keras.layers.Activation('tanh', name='sc_tok_maxpool_activation')

sc_fname_api_concat = tf.keras.layers.Concatenate(name='sc_fname_api_concat')
sc_fname_api_tok_concat = tf.keras.layers.Concatenate(name='sc_fname_api_tok_concat')
sc_dense = tf.keras.layers.Dense(dense_units, activation='tanh',name='sc_dense')


similarity_mode = 'cosine' # 'cosine' 'dense'

pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
desc_output = desc_dense(pooled_output)

desc_embedding_model = tf.keras.Model(inputs=[input_word_ids,
                                              input_mask,
                                              segment_ids],
                                      outputs=desc_output,
                                      name='desc_embedding_model')


sc_fname_embedded = sc_fname_embedding(sc_input_fname_ids)
sc_api_embedded   = sc_api_embedding(sc_input_api_ids)
sc_tok_embedded   = sc_tok_embedding(sc_input_tok_ids)

sc_fname_embedded = sc_fname_dropout(sc_fname_embedded)
sc_api_embedded   = sc_api_dropout(sc_api_embedded)
sc_tok_embedded   = sc_tok_dropout(sc_tok_embedded)

sc_fname_frnn_out = sc_fname_frnn(sc_fname_embedded) # -->
sc_fname_brnn_out = sc_fname_brnn(sc_fname_embedded) # <--
sc_api_frnn_out   = sc_api_frnn(sc_api_embedded)     # -->
sc_api_brnn_out   = sc_api_brnn(sc_api_embedded)     # <--
sc_tok_frnn_out   = sc_tok_frnn(sc_tok_embedded)     # -->
sc_tok_brnn_out   = sc_tok_brnn(sc_tok_embedded)     # <--

sc_fname_frnn_out = sc_fname_rnn_dropout(sc_fname_frnn_out)
sc_fname_brnn_out = sc_fname_rnn_dropout(sc_fname_brnn_out)
sc_api_frnn_out   = sc_api_rnn_dropout(sc_api_frnn_out)
sc_api_brnn_out   = sc_api_rnn_dropout(sc_api_brnn_out)
sc_tok_frnn_out   = sc_tok_rnn_dropout(sc_tok_frnn_out)
sc_tok_brnn_out   = sc_tok_rnn_dropout(sc_tok_brnn_out)

sc_fname_maxpool_concat_out = sc_fname_maxpool_concat([sc_fname_maxpool(sc_fname_frnn_out),
                                                       sc_fname_maxpool(sc_fname_brnn_out)])
sc_api_maxpool_concat_out   = sc_api_maxpool_concat([sc_api_maxpool(sc_api_frnn_out),
                                                     sc_api_maxpool(sc_api_brnn_out)])
sc_tok_maxpool_concat_out   = sc_tok_maxpool_concat([sc_tok_maxpool(sc_tok_frnn_out),
                                                     sc_tok_maxpool(sc_tok_brnn_out)])

sc_fname_vec = sc_fname_maxpool_activation(sc_fname_maxpool_concat_out)
sc_api_vec   = sc_api_maxpool_activation(sc_api_maxpool_concat_out)
sc_tok_vec   = sc_tok_maxpool_activation(sc_tok_maxpool_concat_out)

sc_fname_api_matr = sc_fname_api_concat([sc_fname_vec, sc_api_vec])
sc_fname_api_tok_matr  = sc_fname_api_tok_concat([sc_fname_api_matr, sc_tok_vec])
sc_output = sc_dense(sc_fname_api_tok_matr)

code_embedding_model = tf.keras.Model(inputs=[sc_input_fname_ids,
                                              sc_input_api_ids,
                                              sc_input_tok_ids],
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
                                     sc_input_fname_ids,
                                     sc_input_api_ids,
                                     sc_input_tok_ids],
                             outputs=cos_similarity,
                             name=f'train_{MODEL_TYPE}_model')


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
valid_batch_size = 50
train_samples = len(desc_word_ids)
valid_samples = len(v_desc_word_ids)
train_steps_per_epoch = train_samples // batch_size
valid_steps_per_epoch = valid_samples // valid_batch_size
epochs = 25

train_data = tf.data.Dataset.from_tensor_slices(((desc_word_ids,
                                                  desc_input_mask,
                                                  desc_segment_ids,
                                                  sc_fname_ids,
                                                  sc_api_ids,
                                                  sc_tok_ids),
                                                 np.ones((len(desc_word_ids),1))))\
                            .shuffle(len(desc_word_ids), reshuffle_each_iteration=True)\
                            .batch(batch_size, drop_remainder=True)\
                            .repeat()
valid_data = tf.data.Dataset.from_tensor_slices(((v_desc_word_ids,
                                                  v_desc_input_mask,
                                                  v_desc_segment_ids,
                                                  v_sc_fname_ids,
                                                  v_sc_api_ids,
                                                  v_sc_tok_ids),
                                                 np.ones((len(v_desc_word_ids),1))))\
                            .shuffle(len(v_desc_word_ids), reshuffle_each_iteration=True)\
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
