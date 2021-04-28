
# read in one of the frequency files so that way you can insert the frequency in 
# with the word automatically

import os, re, csv, sys, time, pprint, matplotlib
from sys import stdin

import nltk
from nltk.downloader import Downloader
from nltk.corpus import stopwords as stopwords_corpus

# Directed acylic word graph
import dawg
from dawg import DawgNode

# Demo
import demo

try: 
    import termios
except Exception:
    terminos = fcntl = None

ORIGINAL_KEY = 'original_key'

downloader = Downloader()
downloader.download("stopwords")
nltk.download('averaged_perceptron_tagger')

with open("../data/full.csv", "r") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_ALL)
    documents = [t[0] for t in reader]

matplotlib.rcParams.update({'font.size': 22})
stop_words = set(stopwords_corpus.words("english"))

_non_alpha = re.compile("[^a-zA-Z ]")
def normalize(text):
    return _non_alpha.sub("", text.lower()).strip()

def filter_text(text):
    return (text and text not in stop_words)
    
def tokenize(text):
    return text.split(" ")

def analyze(text):
    for token in tokenize(text):
        normalized = normalize(token)
        if filter_text(normalized):
            yield normalized

def populate_dawg(root):
    # Put words into the DAWG
    for d in documents:
        for word in analyze(d):
            synonym = synonyms.get(word)
            if synonym:        
                leaf_node = root.insert_word_branch(synonym, leaf_node=leaf_node, 
                                                    add_word=False)
            else:
                leaf_node = root.insert_word_branch(word)

# Manually identify synonyms to be inserted into the DAWG
synonyms = {'love': 'romance', 'man': 'men', 'girl': 'woman','dead': 'death'}
root = DawgNode()

# Load review and track the time that it takes
t0 = time.time()
populate_dawg(root)
loadtime = "{0:.2f}".format(time.time() - t0)
print('All items loaded. Load time: ' + str(loadtime))

# Test that the words can be retrived from the DAWG (try to make this an optional test script)
found_node = root.traverse(query="john")
print(list(found_node.get_descendant_words(size=3)))

found_node = root.traverse(query="love")
print(list(found_node.get_descendant_words(size=3)))

found_node = root.traverse(query="new")
print(list(found_node.get_descendant_words(size=3)))