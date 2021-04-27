
# read in one of the frequency files so that way you can insert the frequency in 
# with the word automatically

import re
import csv 
import sys
import pprint
import matplotlib
import pandas as pd

import nltk
from nltk.downloader import Downloader
from nltk.corpus import stopwords as stopwords_corpus

# Directed acylic word graph
import dawg
from dawg import DawgNode

try: 
    import termios
except Exception:
    terminos = fcntl = None

ORIGINAL_KEY = 'original_key'

downloader = Downloader()
downloader.download("stopwords")
nltk.download('averaged_perceptron_tagger')

# Reading in the unigram frequency file
csvfile = open('../output/idf_freq.csv', 'r')
reader = csv.reader(csvfile)

words = []
counts = []

for column in reader:
    words.append(column[0])
    counts.append(column[1])

# Manually identify synonyms to be inserted into a DAWG
synonyms = {'love': 'romance', 'man': 'men', 'girl': 'woman','dead': 'death'}
root = DawgNode()

# Inserting the word and the word count into a DAWG
for word, count in zip(words, counts):
    synonym = synonyms.get(word)
    if synonym:
        leaf_node = root.insert_word_branch(synonym, leaf_node=leaf_node, add_word=False, count=count)
    else:
        leaf_node = root.insert_word_branch(word, count=count)

# Test that the words can be retrived from the DAWG (try to make this an optional test script)
found_node = root.traverse(query="love")
print(list(found_node.get_descendant_nodes(size=5)))

# Randomly choose movies to train the naive bayes classifier with


# Demo


