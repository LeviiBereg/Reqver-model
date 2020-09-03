from utils.java_processing import get_api_sequence, split_java_token, tokenize_java_code, check_and_fix_code_validity
from data_processing.data_extractor import DataExtractor
import re
import string
import tensorflow as tf
import javalang
from dpu_utils.mlutils import Vocabulary


class DataGenerator:

    def __init__(self, params, desc_tokenizer=None, sc_tokenizer=None):
        self.params = params
        self.data_extractor = DataExtractor(params.data_path, params.data_folder, params.use_cols, params.language, params.model_type, params.hdf5_data_folder)
        self.desc_tokenizer = desc_tokenizer
        self.sc_tokenizer = sc_tokenizer

    def create_vocabulary(self, all_ids, max_vocab_size):
        all_tokens = []
        for token_list in all_ids:
            for lst in token_list:
                all_tokens += lst

        vocab = Vocabulary.create_vocabulary(all_tokens,
                                            max_size=max_vocab_size,
                                            count_threshold=int(len(all_tokens[0]) * 0.00025),
                                            add_pad=True)
        return vocab

    def cleaning(self, text):
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

    def generate_bert_input(self, text, max_seq_length, use_vocab):

        tokenized_text = [["[CLS]"] + use_vocab.tokenize(seq)[:max_seq_length-2] + ["[SEP]"] for seq in text]
        input_ids   = [use_vocab.convert_tokens_to_ids(tokens_seq) for tokens_seq in tokenized_text]
        input_mask  = [[1] * len(input_seq) for input_seq in input_ids]
        segment_ids = [[0] * max_seq_length for _ in range(len(input_ids))]
        input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids, maxlen=max_seq_length, padding='post', truncating='post')
        input_mask = tf.keras.preprocessing.sequence.pad_sequences(input_mask, maxlen=max_seq_length, padding='post', truncating='post')
        segment_ids = tf.keras.preprocessing.sequence.pad_sequences(segment_ids, maxlen=max_seq_length, padding='post', truncating='post')

        return [input_ids, input_mask, segment_ids]

    def generate_api_input(self, sc_inputs, max_fname_length, max_api_length, max_tok_length, max_vocab_size=10000,
                        use_vocab=None):
        sc_fname_ids = [split_java_token(javalang.parse.parse_member_signature(sc_input).name)[:max_fname_length] \
                        for sc_input in sc_inputs]
        sc_api_ids = [get_api_sequence(sc_input, split_api_tokens=True)[:max_api_length] for sc_input in sc_inputs]
        sc_tok_ids = [tokenize_java_code(sc_input)[:max_tok_length] for sc_input in sc_inputs]

        all_ids = [sc_fname_ids, sc_api_ids, sc_tok_ids]
        if use_vocab:
            sc_vocab = use_vocab
        else:
            sc_vocab = self.create_vocabulary(all_ids, max_vocab_size)

        sc_fname_ids = [sc_vocab.get_id_or_unk_multiple(sc_input, pad_to_size=max_fname_length) for sc_input in sc_fname_ids]
        sc_api_ids   = [sc_vocab.get_id_or_unk_multiple(sc_input, pad_to_size=max_api_length) for sc_input in sc_api_ids]
        sc_tok_ids   = [sc_vocab.get_id_or_unk_multiple(sc_input, pad_to_size=max_tok_length) for sc_input in sc_tok_ids]

        return sc_vocab, [sc_fname_ids, sc_api_ids, sc_tok_ids]

    def generate_ngram_input(self, sc_inputs, max_seq_length, max_vocab_size=10000, use_vocab=None):

        input_ids = [tokenize_java_code(sc_input)[:max_seq_length] \
                    for sc_input in sc_inputs]

        all_ids = [input_ids]
        if use_vocab:
            sc_vocab = use_vocab
        else:
            sc_vocab = self.create_vocabulary(all_ids, max_vocab_size)

        input_ids = [sc_vocab.get_id_or_unk_multiple(sc_input, pad_to_size=max_seq_length) for sc_input in input_ids]
        return sc_vocab, [input_ids]

    def generate_inputs(self, scope='train', n_splits=1, use_vocab=None):
        pddf = self.data_extractor.read_data(scope=scope, n_splits=n_splits)
        valid_inds = check_and_fix_code_validity(pddf)
        pddf = pddf[valid_inds]
        pddf.docstring = pddf.docstring.apply(self.cleaning)

        desc_input = self.generate_bert_input(pddf.docstring, self.params.desc_max_seq_length, self.desc_tokenizer)

        if self.params.model_type == 'ngram':
            sc_vocab, sc_input = self.generate_ngram_input(pddf.code,
                                                self.params.sc_max_seq_length,
                                                self.params.sc_max_vocab_size,
                                                use_vocab=use_vocab)
        elif self.params.model_type == 'api':
            sc_vocab, sc_input = self.generate_api_input(pddf.code,
                                                    self.params.sc_max_fname_length,
                                                    self.params.sc_max_api_length,
                                                    self.params.sc_max_seq_length,
                                                    self.params.sc_max_vocab_size,
                                                    use_vocab=use_vocab)
        elif self.params.model_type == 'bert':
            sc_input = self.generate_bert_input(pddf.code, self.params.sc_max_seq_length, self.sc_tokenizer)
            sc_vocab = self.sc_tokenizer.vocab
        return desc_input, sc_input, pddf, sc_vocab
