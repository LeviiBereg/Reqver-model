import pickle

from utils.parameters import parse_args
from models.reqver_model import Model
from data_generator import DataGenerator

def train(args):

    dat_gen = DataGenerator(args)
    if args.generate_data:

        desc_input, sc_input, train_df, sc_vocab = dat_gen.generate_inputs(scope='train', 
                                                                           n_splits=args.train_splits, 
                                                                           use_vocab=None)
        v_desc_input, v_sc_input = dat_gen.generate_inputs(scope='valid', 
                                                           n_splits=args.valid_splits, 
                                                           use_vocab=sc_vocab)[:-2]

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

    train_samples = len(desc_input[0])
    valid_samples = len(v_desc_input[0])
    print("Train dataset size", train_samples)
    print("Validation dataset size", valid_samples)
    
    model = Model(args, sc_vocab)

    train_data = (*desc_input, *sc_input)
    valid_data = (*v_desc_input, *v_sc_input)

    train_hist = model.train(train_data, valid_data)
    return train_hist

if __name__ == "__main__":
    args = parse_args()
    train(args)




