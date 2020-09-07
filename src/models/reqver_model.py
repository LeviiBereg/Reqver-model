import datetime
import numpy as np
import tensorflow as tf

from models.api_encoder import ApiEncoder
from models.cnn_encoder import CNNEncoder
from models.bert_encoder import BertEncoder
from utils.metrics import cos_loss, mrr, frank, relevantat1, relevantat5, relevantat10

class Model:

    def __init__(self, params, sc_vocab):
        self.params = params
        self.sc_vocab = sc_vocab
        self.desc_encoder = self._create_desc_encoder()
        self.sc_encoder = self._create_sc_encoder()
        self.model = self._make_graph()
        self._compile_models()
        self.checkpoint_path = f"{self.params.data_path}/model_checkpoints/{self.params.model}/cp.ckpt"
        

    def _create_desc_encoder(self):
        scope = "desc"
        return BertEncoder(scope=scope,
                           max_seq_length=self.params.desc_max_seq_len,
                           num_layers=self.params.desc_bert_layers,
                           hidden_size=self.params.desc_bert_hidden_size,
                           att_heads=self.params.desc_bert_heads,
                           hidden_dropout=self.params.sc_dropout_rate,
                           output_units=self.params.output_units)
    
    def _create_sc_encoder(self):
        scope = "sc"
        new_model = None
        sc_vocab_size = len(self.sc_vocab.token_to_id)
        if self.params.model == "n-gram":
            
            conv_kernel_sizes = []
            for key, value in self.params._get_kwargs():
                if key == "sc-add-conv" and value is not None:
                    conv_kernel_sizes.append(value)
            
            new_model = CNNEncoder(scope=scope, 
                                   max_seq_length=self.params.sc_max_tok_len,
                                   vocab_size=sc_vocab_size,
                                   emb_size=self.params.emb_size,
                                   conv_kernel_sizes=conv_kernel_sizes,
                                   conv_n_filters=self.params.sc_conv_n_filters,
                                   dropout_rate=self.params.sc_dropout_rate,
                                   output_units=self.params.output_units)
        elif self.params.model == "api":
            new_model = ApiEncoder(scope=scope,
                                   max_fname_length=self.params.sc_max_fname_len,
                                   max_api_length=self.params.sc_max_api_len,
                                   max_tok_length=self.params.sc_max_tok_len,
                                   vocab_size=sc_vocab_size,
                                   emb_size=self.params.emb_size,
                                   dropout_rate=self.params.sc_dropout_rate,
                                   lstm_units=self.params.sc_rnn_units,
                                   lstm_rec_dropout_rate=self.params.sc_rnn_dropout_rate,
                                   output_units=self.params.output_units)
        elif self.params.model == "bert":
            new_model = BertEncoder(scope=scope,
                                    max_seq_length=self.params.sc_max_tok_len,
                                    num_layers=self.params.sc_bert_layers,
                                    hidden_size=self.params.sc_bert_hidden_size,
                                    att_heads=self.params.sc_bert_heads,
                                    hidden_dropout=self.params.sc_dropout_rate,
                                    output_units=self.params.output_units)
        return new_model

    def _make_graph(self):
        eps = 1e-10
        norm_desc = tf.norm(self.desc_encoder.outputs, axis=-1, keepdims=True) + eps
        norm_sc   = tf.norm(self.sc_encoder.outputs, axis=-1, keepdims=True)   + eps
        self.outputs = tf.matmul(self.desc_encoder.outputs/norm_desc,
                                 self.sc_encoder.outputs/norm_sc,
                                 transpose_a=False,
                                 transpose_b=True,
                                 name='desc_sc_cos_sim_logits')  # (batch_size, batch_size)
        self.inputs = [*self.desc_encoder.inputs, *self.sc_encoder.inputs]
        reqver_model = tf.keras.Model(inputs=self.inputs,
                                      outputs=self.outputs,
                                      name=f'reqver_{self.params.model}_model')
        
        return reqver_model

    def _compile_models(self):
        self.model.compile(loss=cos_loss, optimizer=self.params.optimizer, metrics=[mrr])
        self.desc_encoder.model.compile(loss=cos_loss, optimizer=self.params.optimizer)
        self.sc_encoder.model.compile(loss=cos_loss, optimizer=self.params.optimizer)

    def train(self, train_data, valid_data):
        callbacks = []
        if self.params.tb_callback:
            callbacks.append(self.get_tb_callback())
        if self.params.cp_callback:
            callbacks.append(self.get_cp_callback())
        callbacks = callbacks if len(callbacks) > 0 else None

        if self.params.load_cp:
            print("Load model weights from:", self.checkpoint_path)
            self.model.load_weights(self.checkpoint_path)

        train_samples = len(train_data[0])
        valid_samples = len(valid_data[0])
        train_data_ds = tf.data.Dataset.from_tensor_slices((train_data, np.ones((train_samples,1)))) \
                                       .shuffle(train_samples, reshuffle_each_iteration=True)        \
                                       .batch(self.params.batch_size, drop_remainder=True)           \
                                       .repeat()
        valid_data_ds = tf.data.Dataset.from_tensor_slices((valid_data, np.ones((valid_samples,1)))) \
                                       .shuffle(valid_samples, reshuffle_each_iteration=True)        \
                                       .batch(self.params.valid_batch_size, drop_remainder=True)     \
                                       .repeat()
        train_steps_per_epoch = train_samples // self.params.batch_size
        valid_steps_per_epoch = valid_samples // self.params.valid_batch_size

        train_hist = self.model.fit(train_data_ds,
                                    epochs=self.params.epochs,
                                    validation_data=valid_data_ds,
                                    callbacks=callbacks,
                                    steps_per_epoch=train_steps_per_epoch,
                                    validation_steps=valid_steps_per_epoch)
        return train_hist

    def get_tb_callback(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"{self.params.data_path}/logs/fit/{self.params.model}/" + current_time
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        return tb_callback

    def get_cp_callback(self):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_best_only=True,
                                                         save_weights_only=True,
                                                         monitor='val_mrr',
                                                         mode='max')
        return cp_callback

    def evaluate(self, test_data):
        if self.params.load_cp:
            print("Load model weights from:", self.checkpoint_path)
            self.model.load_weights(self.checkpoint_path)
        self.model.compile(loss=cos_loss, 
                           optimizer=self.params.optimizer, 
                           metrics=[mrr, frank, relevantat1, relevantat5, relevantat10])

        test_samples = len(test_data[0])
        test_data_ds = tf.data.Dataset.from_tensor_slices((test_data, np.ones((test_samples,1)))) \
                                       .shuffle(test_samples, reshuffle_each_iteration=True)      \
                                       .batch(self.params.batch_size, drop_remainder=True)        \
                                       .repeat()
        test_steps_per_epoch = test_samples // self.params.batch_size

        eval_res = self.model.evaluate(test_data_ds, steps=test_steps_per_epoch)
        return eval_res
