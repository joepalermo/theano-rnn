"""Most of the parsing code here is copied from Danny Britz:
https://github.com/dennybritz/rnn-tutorial-rnnlm/
"""

import csv
import itertools
import numpy as np
import nltk

unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

def parse_reddit_data(vocab_size, data_path, data_split=[0.8,0.1,0.1]):
    """Parse the reddit comment data and return training, validation and test
    sets.

    Args:
        vocab_size: The size of the vocabulary.
        data_path: The path to the data.
        data_split: A list that determines how to porportion the split of the
        data into training, validation, and test data respectively.

    Returns:
        A 3-tuple containing training, validation and test data. Validation
        and test data may be null.
    """

    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print("Reading CSV file...")
    with open(data_path) as f:
        reader = csv.reader(f, skipinitialspace=True)
        next(reader)
        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print("Parsed %d sentences." % (len(sentences)))

    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    num_sentences = len(tokenized_sentences)

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique words tokens." % len(word_freq.items()))

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocab_size-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

    print("Using vocabulary size %d." % vocab_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    # determine at what indices to split the available data
    training_split = int(num_sentences * data_split[0])
    validation_split = (training_split, int(training_split + num_sentences * data_split[1]))
    test_split = validation_split[1]

    # split the data
    training_sentences = tokenized_sentences[:training_split]
    validation_sentences = tokenized_sentences[validation_split[0]: validation_split[1]]
    test_sentences = tokenized_sentences[test_split:]

    # Create the training, validation and test data
    x_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in training_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in training_sentences])
    training_data = (x_train, y_train)

    if data_split[1] != 0:
        x_validation = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in validation_sentences])
        y_validation = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in validation_sentences])
        validation_data = (x_validation, y_validation)
    else:
        validation_data = None

    if data_split[2] != 0:
        x_test = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in test_sentences])
        y_test = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in test_sentences])
        test_data = (x_test, y_test)
    else:
        test_data = None

    return (training_data, validation_data, test_data)
