import tensorflow as tf


class ApiEncoder:

    def __init__(self, scope, max_fname_length, max_api_length, max_tok_length, vocab_size, emb_size, dropout_rate, lstm_units, lstm_rec_dropout_rate, output_units):
        self.model_name = "api_encoder"

        self.scope = scope
        self.max_fname_length = max_fname_length
        self.max_api_length = max_api_length
        self.max_tok_length = max_tok_length
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.dropout_rate = dropout_rate
        self.lstm_units = lstm_units
        self.lstm_rec_dropout_rate = lstm_rec_dropout_rate
        self.output_units = output_units

        self._init_layers()
        self.model = self._make_graph()

    def _init_layers(self):
        """
        Initializes the layers of encoder
        """
        self.sc_input_fname_ids = tf.keras.layers.Input(shape=(self.max_fname_length,),
                                                        dtype=tf.int32,
                                                        name=f"{self.scope}_input_fname_ids")
        self.sc_input_api_ids   = tf.keras.layers.Input(shape=(self.max_api_length,),
                                                        dtype=tf.int32,
                                                        name=f"{self.scope}_input_api_ids")
        self.sc_input_tok_ids   = tf.keras.layers.Input(shape=(self.max_tok_length,),
                                                        dtype=tf.int32,
                                                        name=f"{self.scope}_input_tok_ids")

        self.sc_fname_embedding = tf.keras.layers.Embedding(self.vocab_size,
                                                            self.emb_size,
                                                            mask_zero=False,
                                                            name=f"{self.scope}_fname_embedding")
        self.sc_api_embedding   = tf.keras.layers.Embedding(self.vocab_size,
                                                            self.emb_size,
                                                            mask_zero=False,
                                                            name=f"{self.scope}_api_embedding")
        self.sc_tok_embedding   = tf.keras.layers.Embedding(self.vocab_size,
                                                            self.emb_size,
                                                            mask_zero=False,
                                                            name=f"{self.scope}_tok_embedding")

        self.sc_fname_dropout = tf.keras.layers.Dropout(self.dropout_rate, name=f"{self.scope}_fname_dropout")
        self.sc_api_dropout   = tf.keras.layers.Dropout(self.dropout_rate, name=f"{self.scope}_api_dropout")
        self.sc_tok_dropout   = tf.keras.layers.Dropout(self.dropout_rate, name=f"{self.scope}_tok_dropout")

        self.sc_fname_frnn = tf.keras.layers.LSTM(self.lstm_units,
                                                  recurrent_dropout=self.lstm_rec_dropout_rate,
                                                  return_sequences=True,
                                                  name=f"{self.scope}_fname_frnn")
        self.sc_fname_brnn = tf.keras.layers.LSTM(self.lstm_units,
                                                  return_sequences=True,
                                                  recurrent_dropout=self.lstm_rec_dropout_rate,
                                                  name=f"{self.scope}_fname_brnn",
                                                  go_backwards=True)
        self.sc_api_frnn = tf.keras.layers.LSTM(self.lstm_units,
                                                recurrent_dropout=self.lstm_rec_dropout_rate,
                                                return_sequences=True,
                                                name=f"{self.scope}_api_frnn")
        self.sc_api_brnn = tf.keras.layers.LSTM(self.lstm_units,
                                                return_sequences=True,
                                                recurrent_dropout=self.lstm_rec_dropout_rate,
                                                name=f"{self.scope}_api_brnn",
                                                go_backwards=True)
        self.sc_tok_frnn = tf.keras.layers.LSTM(self.lstm_units,
                                                recurrent_dropout=self.lstm_rec_dropout_rate,
                                                return_sequences=True,
                                                name=f"{self.scope}_tok_frnn")
        self.sc_tok_brnn = tf.keras.layers.LSTM(self.lstm_units,
                                                return_sequences=True,
                                                recurrent_dropout=self.lstm_rec_dropout_rate,
                                                name=f"{self.scope}_tok_brnn",
                                                go_backwards=True)

        self.sc_fname_rnn_dropout = tf.keras.layers.Dropout(self.dropout_rate, name=f"{self.scope}_fname_rnn_dropout")
        self.sc_api_rnn_dropout   = tf.keras.layers.Dropout(self.dropout_rate, name=f"{self.scope}_api_rnn_dropout")
        self.sc_tok_rnn_dropout   = tf.keras.layers.Dropout(self.dropout_rate, name=f"{self.scope}_tok_rnn_dropout")

        self.sc_fname_maxpool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=1, keepdims=False),
                                                       output_shape=lambda x: (x[0], x[2]),
                                                       name=f"{self.scope}_fname_maxpool")
        self.sc_fname_maxpool_concat = tf.keras.layers.Concatenate(name=f"{self.scope}_fname_maxpool_concat")
        self.sc_fname_maxpool_activation = tf.keras.layers.Activation("tanh", name=f"{self.scope}_fname_maxpool_activation")
        self.sc_api_maxpool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=1, keepdims=False),
                                                     output_shape=lambda x: (x[0], x[2]),
                                                     name=f"{self.scope}_api_maxpool")
        self.sc_api_maxpool_concat = tf.keras.layers.Concatenate(name=f"{self.scope}_api_maxpool_concat")
        self.sc_api_maxpool_activation = tf.keras.layers.Activation("tanh", name=f"{self.scope}_api_maxpool_activation")
        self.sc_tok_maxpool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=1, keepdims=False),
                                                     output_shape=lambda x: (x[0], x[2]),
                                                     name=f"{self.scope}_tok_maxpool")
        self.sc_tok_maxpool_concat = tf.keras.layers.Concatenate(name=f"{self.scope}_tok_maxpool_concat")
        self.sc_tok_maxpool_activation = tf.keras.layers.Activation("tanh", name=f"{self.scope}_tok_maxpool_activation")

        self.sc_fname_api_concat = tf.keras.layers.Concatenate(name=f"{self.scope}_fname_api_concat")
        self.sc_fname_api_tok_concat = tf.keras.layers.Concatenate(name=f"{self.scope}_fname_api_tok_concat")
        self.sc_dense = tf.keras.layers.Dense(self.output_units, activation="tanh", name=f"{self.scope}_dense")


    def _make_graph(self):
        """
        Creates the computational graph of the encoder
        """
        sc_fname_embedded = self.sc_fname_embedding(self.sc_input_fname_ids)
        sc_api_embedded   = self.sc_api_embedding(self.sc_input_api_ids)
        sc_tok_embedded   = self.sc_tok_embedding(self.sc_input_tok_ids)

        sc_fname_embedded = self.sc_fname_dropout(sc_fname_embedded)
        sc_api_embedded   = self.sc_api_dropout(sc_api_embedded)
        sc_tok_embedded   = self.sc_tok_dropout(sc_tok_embedded)

        sc_fname_frnn_out = self.sc_fname_frnn(sc_fname_embedded) # -->
        sc_fname_brnn_out = self.sc_fname_brnn(sc_fname_embedded) # <--
        sc_api_frnn_out   = self.sc_api_frnn(sc_api_embedded)     # -->
        sc_api_brnn_out   = self.sc_api_brnn(sc_api_embedded)     # <--
        sc_tok_frnn_out   = self.sc_tok_frnn(sc_tok_embedded)     # -->
        sc_tok_brnn_out   = self.sc_tok_brnn(sc_tok_embedded)     # <--

        sc_fname_frnn_out = self.sc_fname_rnn_dropout(sc_fname_frnn_out)
        sc_fname_brnn_out = self.sc_fname_rnn_dropout(sc_fname_brnn_out)
        sc_api_frnn_out   = self.sc_api_rnn_dropout(sc_api_frnn_out)
        sc_api_brnn_out   = self.sc_api_rnn_dropout(sc_api_brnn_out)
        sc_tok_frnn_out   = self.sc_tok_rnn_dropout(sc_tok_frnn_out)
        sc_tok_brnn_out   = self.sc_tok_rnn_dropout(sc_tok_brnn_out)

        sc_fname_maxpool_concat_out = self.sc_fname_maxpool_concat([self.sc_fname_maxpool(sc_fname_frnn_out),
                                                                    self.sc_fname_maxpool(sc_fname_brnn_out)])
        sc_api_maxpool_concat_out   = self.sc_api_maxpool_concat([self.sc_api_maxpool(sc_api_frnn_out),
                                                                  self.sc_api_maxpool(sc_api_brnn_out)])
        sc_tok_maxpool_concat_out   = self.sc_tok_maxpool_concat([self.sc_tok_maxpool(sc_tok_frnn_out),
                                                                  self.sc_tok_maxpool(sc_tok_brnn_out)])

        sc_fname_vec = self.sc_fname_maxpool_activation(sc_fname_maxpool_concat_out)
        sc_api_vec   = self.sc_api_maxpool_activation(sc_api_maxpool_concat_out)
        sc_tok_vec   = self.sc_tok_maxpool_activation(sc_tok_maxpool_concat_out)

        sc_fname_api_matr = self.sc_fname_api_concat([sc_fname_vec, sc_api_vec])
        sc_fname_api_tok_matr = self.sc_fname_api_tok_concat([sc_fname_api_matr, sc_tok_vec])
        self.inputs = [self.sc_input_fname_ids,
                       self.sc_input_api_ids,
                       self.sc_input_tok_ids]
        self.outputs = self.sc_dense(sc_fname_api_tok_matr)


        sc_embedding_model = tf.keras.Model(inputs=self.inputs,
                                            outputs=self.outputs,
                                            name=f"{self.scope}_{self.model_name}")

        return sc_embedding_model