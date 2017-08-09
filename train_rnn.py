import argparse
from parse_reddit_data import parse_reddit_data
from utils import pp_output, save_model_parameters, load_model_parameters
from RNN import RNN

def main(vocab_size, state_size, bptt_truncate, model_path, data_path,
         num_epochs, learning_rate):
    # create an RNN, if possible load pre-existing model parameters
    if model_path:
        model_parameters = load_model_parameters(model_path)
        model = RNN(vocab_size, state_size, bptt_truncate, model_parameters)
    else:
        model = RNN(vocab_size, state_size, bptt_truncate)

    # construct datasets
    training_data, validation_data, test_data, index_to_word = \
    parse_reddit_data(vocab_size, data_path)

    # train the model
    model.sgd(training_data, num_epochs, learning_rate, validation_data)

    # get word predictions for a sample of inputs
    test_inputs = test_data[0]
    outputs = model.get_predictions(test_inputs)
    for o in outputs:
        print(pp_output(o, index_to_word))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, default=1000,
                      help='the size of the model\'s vocabulary')
    parser.add_argument('--state_size', type=int, default=100,
                      help='the size of the model\'s state')
    parser.add_argument('--bptt_truncate', type=int, default=3,
                      help='number of timesteps until bptt truncation')
    parser.add_argument('--model_path', type=str,
                      help='the relative path to saved model parameters')
    parser.add_argument('--data_path', type=str,
                      default='data/reddit_data/reddit-comments-small.csv',
                      help='the path to the training/validation/test data')
    parser.add_argument('--num_epochs', type=int, default=10,
                      help='the number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.05,
                      help='the learning rate')
    args = parser.parse_args()
    main(args.vocab_size,
         args.state_size,
         args.bptt_truncate,
         args.model_path,
         args.data_path,
         args.num_epochs,
         args.learning_rate)
