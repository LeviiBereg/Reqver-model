import tensorflow as tf
import bert

class BertEncoder:

    def __init__(self, scope, max_seq_length, bert_params, output_units):
        self.model_name = "bert"

        self.scope = scope
        self.max_seq_length = max_seq_length
        self.bert_params = bert_params
        self.output_units = output_units

        self._init_layers()
        self.model = self._make_graph()

    def _init_layers(self):
        """
        Initializes the layers of encoder
        """
        self.input_word_ids = tf.keras.layers.Input(shape=(self.max_seq_length,),
                                                    dtype=tf.int32,
                                                    name=f"{self.scope}_input_word_ids")
        self.bert_layer = bert.BertModelLayer(**self.bert_params)
        self.bert_pooling = tf.keras.layers.Lambda(lambda x: tf.math.reduce_max(x, axis=1))# tf.keras.layers.Lambda(lambda seq: seq[:, 0, :])
        self.bert_dropout = tf.keras.layers.Dropout(0.5)
        self.bert_dense = tf.keras.layers.Dense(self.output_units, activation='tanh', name=f"{self.scope}_dense")


    def _make_graph(self):
        """
        Creates the computational graph of the encoder
        """
        sc_sequence_output = self.bert_layer(self.input_word_ids)
        sc_pooled_output = self.bert_pooling(sc_sequence_output)
        sc_pooled_output = self.bert_dropout(sc_pooled_output)
        self.inputs  = self.input_word_ids
        self.outputs = self.bert_dense(sc_pooled_output)

        sc_embedding_model = tf.keras.Model(inputs=self.inputs,
                                            outputs=self.outputs,
                                            name=f"{self.scope}_{self.model_name}")

        return sc_embedding_model