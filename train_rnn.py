"""most of the code is copied from Danny Britz
https://github.com/dennybritz/rnn-tutorial-rnnlm/
"""

import csv
import itertools
import numpy as np
import nltk
from utils import pp_output
from MyRNN import RNN

# preprocessing ----------------------------------------------------------------

vocabulary_size = 500
state_size = 100
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print("Reading CSV file...")
with open('data/reddit_data/reddit-comments-small.csv') as f:
    reader = csv.reader(f, skipinitialspace=True)
    next(reader)
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print("Parsed %d sentences." % (len(sentences)))

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print("Found %d unique words tokens." % len(word_freq.items()))

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print("Using vocabulary size %d." % vocabulary_size)
print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

# Create the training data - numpy arrays of lists containing word indices
x_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
training_data = (x_train, y_train)


# define a model and train it
model = RNN(vocabulary_size, state_size=state_size)
# model.sgd(training_data, learning_rate=0.005, num_epochs=3)

# get word predictions for a sample of inputs
outputs = model.get_predictions(x_test)
for o in outputs:
    print(pp_output(o, index_to_word))
