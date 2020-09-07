import re
import os
import time
import pickle
import datetime
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfh
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from dpu_utils.mlutils import Vocabulary
from gensim.models import KeyedVectors as word2vec

import bert
FullTokenizer = bert.bert_tokenization.FullTokenizer

from utils.parameters import parse_args
from models.reqver_model import Model
from data_processing.data_generator import DataGenerator

def train(args):

    dat_gen = DataGenerator(args)
    if args.generate_data:

        desc_input, sc_input, train_df, sc_vocab = dat_gen.generate_inputs(scope='train', 
                                                                           n_splits=args.train_splits, 
                                                                           use_vocab=None)

        print("Train dataset size", len(train_df))

        v_desc_input, v_sc_input = dat_gen.generate_inputs(scope='valid', 
                                                           n_splits=args.valid_splits, 
                                                           use_vocab=sc_vocab)[:-2]
        print("Validation dataset size", len(v_desc_input[0]))

        with open(args.sc_vocab_file, 'wb') as pickle_file:
            pickle.dump(sc_vocab, pickle_file)

        dat_gen.write_hdf5_data(desc_input, 'desc_input')
        dat_gen.write_hdf5_data(sc_input, 'sc_input')
        dat_gen.write_hdf5_data(v_desc_input, 'v_desc_input')
        dat_gen.write_hdf5_data(v_sc_input, 'v_sc_input')
    else:

        with open(args.sc_vocab_file, 'rb') as pickle_file:
            sc_vocab = pickle.load(pickle_file)

        desc_input = dat_gen.read_hdf5_data('desc_input')
        sc_input   = dat_gen.read_hdf5_data('sc_input')
        v_desc_input = dat_gen.read_hdf5_data('v_desc_input')
        v_sc_input   = dat_gen.read_hdf5_data('v_sc_input')

        print("Train dataset size", len(desc_input[0]))
        print("Validation dataset size", len(v_desc_input[0]))
    
    model = Model(args, sc_vocab)

    

if __name__ == "__main__":
    args = parse_args()
    train(args)


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

# splitted_data = train_test_split(desc_word_ids,
#                                  desc_input_mask,
#                                  desc_segment_ids,
#                                  sc_fname_ids,
#                                  sc_api_ids,
#                                  sc_tok_ids)
# train_desc_word_ids, v_desc_word_ids = splitted_data[:2]
# train_desc_input_mask, v_desc_input_mask = splitted_data[2:4]
# train_desc_segment_ids, v_desc_segment_ids = splitted_data[4:6]
# train_sc_fname_ids, v_sc_fname_ids = splitted_data[6:8]
# train_sc_api_ids, v_sc_api_ids = splitted_data[8:10]
# train_sc_tok_ids, v_sc_tok_ids = splitted_data[10:12]

batch_size = 256
valid_batch_size = 256
train_samples = len(desc_word_ids)
valid_samples = len(v_desc_word_ids)
train_steps_per_epoch = train_samples // batch_size
valid_steps_per_epoch = valid_samples // valid_batch_size
epochs = 15

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
