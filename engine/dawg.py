from collections import deque
from itertools import islice

INF = float('inf')
allmovies = set()

class NodeNotFound(ValueError):
    pass

# The DAWG data structure that keeps a set of words, organized with one node for 
# each letter. The node also holds the count of the word or the frequency.
class DawgNode: 

    SHOULD_INCLUDE_COUNT = True

    __slots__ = ("word", "children", "count")

    def __init__(self):
        self.word = None
        self.children = {}
        self.count = 0

    def __repr__(self):
        return f'< children: {list(self.children.keys())}, word: {self.word} >'

    @property
    def value(self):
        return self.word

    def insert(self, word, add_word=True, count=0, insert_count=True):
        node = self
        for letter in word:
            if letter not in node.children:
                node.children[letter] = DawgNode()
            node = node.children[letter]
        if add_word:
            node.word = word
            if insert_count:
                node.count = int(count) # converts string to integer
        return node

    # Inserts a word into the DAWG.
    def insert_word_branch(self, word, leaf_node=None, add_word=True, count=0):
        if leaf_node:
            temp_leaf_node = self.insert(
                word[:-1],
                add_word=add_word,
                count=count,
                insert_count=self.SHOULD_INCLUDE_COUNT
            )
            temp_leaf_node.children[word[-1]] = leaf_node
        else:
            temp_leaf_node = self.insert(
                word,
                count=count,
                insert_count=self.SHOULD_INCLUDE_COUNT
            )
        self.insert_word_callback(word)
        return leaf_node

    def traverse(self, query):
        node = self
        for letter in query:
            child = node.children.get(letter)
            if child:
                node = child
            else:
                break
        return node

    # Once word is inserted, this is called.
    def insert_word_callback(self, word):
        """
        Once word is inserted, run this.
        """
        pass

    # Breadth-first search 
    def get_descendant_nodes(self, size, should_traverse=True, full_stop_words=None, insert_count=True):
        if insert_count is True:
            size = INF
        
        que = deque()
        unique_nodes = {self}
        found_nodes_set = set()
        full_stop_words = full_stop_words if full_stop_words else set()

        for letter, child_node in self.children.items():
            if child_node not in unique_nodes:
                unique_nodes.add(child_node)    
                que.append((letter, child_node))
        
        while que:
            letter, child_node = que.popleft()
            child_value = child_node.value
            if child_value:
                if child_value in full_stop_words:
                    should_traverse = False
                if child_value not in found_nodes_set:
                    found_nodes_set.add(child_value)
                    yield child_node
                    if len(found_nodes_set) > size:
                        break
            
            if should_traverse:
                for letter, grand_child_node in child_node.children.items():
                    if grand_child_node not in unique_nodes:
                        unique_nodes.add(grand_child_node)
                        que.append((letter, grand_child_node))

    # Add naive bayes classifier to this method
    def get_descendant_words(self, size, should_traverse=True, full_stop_words=None, insert_count=True):
        found_nodes_gen = self.get_descendant_nodes(
            size,
            should_traverse=should_traverse,
            full_stop_words=full_stop_words,
            insert_count=insert_count
        )
        
        if insert_count is True:
            found_nodes = sorted(
                found_nodes_gen, 
                key=lambda node: node.count, 
                reverse=True
            )[:size + 1]
        else:
            found_nodes = islice(found_nodes_gen, size)

        return map(lambda word: word.value, found_nodes)