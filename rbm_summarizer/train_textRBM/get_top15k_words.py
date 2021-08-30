# a file to get a list of top 15k most frequent words in the cnn/dailymail data

from __future__ import unicode_literals, print_function, division

# load modules in other files
import sys
import os


module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from data_util import utils
from data_util import config


## step 1: load the stopwords provided by Stanford CoreNLP
f = open("/home/yang6367/gitrepos/cnn-dailymail/corenlp-stopwords.txt")
stopwords = f.read().split('\n')
new_stopwords = ['!!!', '!!!!', "'''", "**", "***", "****", "*****", "******", "*******", "--", "-rcb-", "/", ]
stopwords = stopwords + new_stopwords

## step 2: create a dictionary containing the lemmatized word count
word_to_count = {}
num_words = 0
max_size = 50000

with open(config.vocab_path, 'r') as vocab_f:
    for line in vocab_f:
        pieces = line.split()
        try:
            # if the word is not a stopword, then add the count to the dictionary
            if pieces[0] not in stopwords:
                w = utils.lem(pieces[0]) # lemmatize the word
                if w in word_to_count:  # if already in, then add up the count
                    word_to_count[w] += int(pieces[1])
                else:
                    word_to_count[w] = int(pieces[1])
                    num_words += 1
        except:
            print(pieces)
        if num_words >= max_size:
            break

## step 3: get top15k words
# sort the dictionary by the count
word_to_count_sorted = sorted(word_to_count.items(), key=lambda x: x[1], reverse=True)
# choose the top 15k words
word_to_count_top15k = word_to_count_sorted[:15000]
words_top15k = [key for key, value in word_to_count_top15k]
# write the list of words to a txt file
with open("/home/yang6367/gitrepos/cnn-dailymail/top15k_words.txt", "w") as f:
    f.writelines("%s\n" % word for word in words_top15k)