import tensorflow as tf


class CNNEncoder:

    def __init__(self, scope, max_seq_length, vocab_size, emb_size, conv_kernel_sizes, conv_n_filters, dropout_rate, output_units):
        self.model_name = 'n-gram_encoder'
        
        self.scope = scope
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_n_filters = conv_n_filters
        self.dropout_rate = dropout_rate
        self.output_units = output_units

        self._init_layers()
        self.model = self._make_graph()

    def _init_layers(self):
        """
        Initializes the layers of encoder
        """
        self.sc_input_tok_ids = tf.keras.layers.Input(shape=(self.max_seq_length,),
                                                      dtype=tf.int32,
                                                      name=f"{self.scope}_tok_ids")

        self.sc_tok_embedding = tf.keras.layers.Embedding(self.vocab_size,
                                                          self.emb_size,
                                                          mask_zero=True,
                                                          name=f"{self.scope}_tok_embedding")

        self.sc_convs = []
        self.sc_max_pools = []
        self.sc_conv_dropouts = []
        self.sc_conv_flatten = tf.keras.layers.Flatten()
        for kernel_size in self.conv_kernel_sizes:
            self.sc_convs.append(tf.keras.layers.Conv1D(self.conv_n_filters, kernel_size, 
                                                        activation='relu', 
                                                        name=f'{self.scope}_conv_{kernel_size}'))
            self.sc_max_pools.append(tf.keras.layers.MaxPooling1D(self.max_seq_length - kernel_size + 1, 1, 
                                                                  name=f'{self.scope}_max_pool_{kernel_size}'))
            self.sc_conv_dropouts.append(tf.keras.layers.Dropout(self.dropout_rate, 
                                                                name=f'{self.scope}_conv_dropout_{kernel_size}'))

        self.sc_dense = tf.keras.layers.Dense(self.output_units, activation='tanh', name=f"{self.scope}_dense")


    def _make_graph(self):
        """
        Creates the computational graph of the encoder
        """
        sc_embedded_input = self.sc_tok_embedding(self.sc_input_tok_ids) # (batch_size, sc_max_seq_length, emb_vec_size)
        conv_outputs = []
        for sc_conv, sc_max_pool, sc_dropout in zip(self.sc_convs, self.sc_max_pools, self.sc_conv_dropouts):
            sc_conv_out = sc_conv(sc_embedded_input)
            sc_conv_out = sc_max_pool(sc_conv_out)
            sc_conv_out = sc_dropout(sc_conv_out)
            conv_outputs.append(sc_conv_out)
        sc_output = tf.concat(conv_outputs, 2) # (batch_size, 1, n_convs * conv_n_filters)
        sc_output = self.sc_conv_flatten(sc_output) # (batch_size, n_convs * conv_n_filters)
        self.inputs  = [self.sc_input_tok_ids]
        self.outputs = self.sc_dense(sc_output)

        sc_embedding_model = tf.keras.Model(inputs=self.inputs,
                                            outputs=self.outputs,
                                            name=f'{self.scope}_{self.model_name}')

        return sc_embedding_model