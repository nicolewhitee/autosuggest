
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
                leaf_node = root.insert_word_branch(synonym, leaf_node=leaf_node, add_word=False)
            else:
                leaf_node = root.insert_word_branch(word)

# Waits for a single keypress on stdin and returns the character of the key that was pressed
def read_single_keypress():
    if fcntl is None or termios is None:
        raise ValueError('termios and/or fcntl packages are not available in your system. This is possible because you are not on a Linux Distro.')
    fd = sys.stdin.fileno()
    # save old state
    flags_save = fcntl.fcntl(fd, fcntl.F_GETFL)
    attrs_save = termios.tcgetattr(fd)
    # make raw - the way to do this comes from the termios(3) man page.
    attrs = list(attrs_save)  # copy the stored version to update
    # iflag
    attrs[0] &= ~(termios.IGNBRK | termios.BRKINT | termios.PARMRK |
                  termios.ISTRIP | termios.INLCR | termios.IGNCR |
                  termios.ICRNL | termios.IXON)
    # oflag
    attrs[1] &= ~termios.OPOST
    # cflag
    attrs[2] &= ~(termios.CSIZE | termios. PARENB)
    attrs[2] |= termios.CS8
    # lflag
    attrs[3] &= ~(termios.ECHONL | termios.ECHO | termios.ICANON |
                  termios.ISIG | termios.IEXTEN)
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    # turn off non-blocking
    fcntl.fcntl(fd, fcntl.F_SETFL, flags_save & ~os.O_NONBLOCK)
    # read a single keystroke
    try:
        ret = sys.stdin.read(1)  # returns a single character
    except KeyboardInterrupt:
        ret = 0
    finally:
        # restore old state
        termios.tcsetattr(fd, termios.TCSAFLUSH, attrs_save)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags_save)
    return ret

def demo(running_modules, max_cost, size):
    word_list = []

    running_modules = running_modules if isinstance(running_modules, dict) else {running_modules.__class__.__name__: running_modules}

    print('AUTOSUGGEST DEMO')
    print('Press any key to search for. Press ctrl+c to exit')

    while True: 
        pressed = read_single_keypress()
        if pressed == '\x7f':
            if word_list:
                word_list.pop()
        elif pressed == '\x03':
            break
        else:
            word_list.append(pressed)

        joined = ''.join(word_list)
        print(chr(27) + "[2J")
        print(joined)
        results = {}
        for module_name, module in running_modules.items():
            results[module_name] = module.search(word=joined, max_cost=max_cost, size=size)
        pprint(results)
        print('')

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

# Demo
#demo(max_cost=10, size=3)