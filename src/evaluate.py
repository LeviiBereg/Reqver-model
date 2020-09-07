import pickle

from utils.parameters import parse_args
from models.reqver_model import Model
from data_generator import DataGenerator

def evaluate(args):

    dat_gen = DataGenerator(args)
    with open(args.sc_vocab_file, 'rb') as pickle_file:
        sc_vocab = pickle.load(pickle_file)
    desc_input, sc_input = dat_gen.generate_inputs(scope='test', use_vocab=sc_vocab)[-2]
    test_samples = len(desc_input[0])
    print("Test dataset size", test_samples)
    
    model = Model(args, sc_vocab)

    test_data = (*desc_input, *sc_input)
    test_hist = model.evaluate(test_data)
    return test_hist

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)




