from __future__ import annotations
from collections import deque
import itertools
from typing import Any

class Leaf(dict):
    def __repr__(self) -> str:
        return "Leaf(%s)" % super().__repr__()

class End:
    def __repr__(self) -> str:
        return "End"
    
    def __lt__(self, other) -> bool:
        return False
    
    def __gt__(self, other) -> bool:
        return True
    
    def __eq__(self, value: object) -> bool:
        return False
    
    def __hash__(self) -> int:
        return id(self)

# Do not instantiate more than one End
end_symb = End()

def insert_branch_into_tree(tree : dict, branch : dict) -> None:
    if tree is branch or not (dict == type(tree) == type(branch)):
        return
    for kb,vb in branch.items():
        vt = tree.get(kb)
        if vt is None:
            tree[kb] = vb
        else:
            insert_branch_into_tree(vt, vb)

def tree_from_list_of_choices(list_of_choices : list[list[str | list[int]]], alphabet : tuple[str | tuple[int]] = None) -> tuple[dict, tuple]:
    root = {}
    alphaset = set()
    common_leaf = root
    any_is_empty = []
    leaves_from_root = []
    len_list_choices = len(list_of_choices)
    for k,l in enumerate(list_of_choices):
        leaves_from_root.append(common_leaf)
        current_tree = common_leaf
        common_leaf = Leaf() if k != len_list_choices - 1 else end_symb
        any_is_empty_k = False
        for ch in l:
            current = current_tree
            last_idx = len(ch) - 1
            # (last_idx == -1) means ch is an empty string
            any_is_empty_k = any_is_empty_k or last_idx == -1
            for i,c in enumerate(ch):
                # Make c hashable
                if isinstance(c, list):
                    c = tuple(c)
                elif not isinstance(c, (str, tuple)):
                    c = (c,)
                alphaset.add(c)
                d = current.get(c)
                
                if d is None:
                    d = {}
                    current[c] = d
                
                current = d
                if i == last_idx:
                    current[''] = common_leaf
        any_is_empty.append(any_is_empty_k)
    else:
        leaves_from_root.append(common_leaf)


    # Handle empty choices
    for i in range(len_list_choices):
        count_successive_empty = 0
        for k in any_is_empty[i:]:
            if not k:
                break
            count_successive_empty += 1

        for j in range(i+1, i+1+count_successive_empty):
            d = leaves_from_root[i].get('')
            if d is None:
                leaves_from_root[i][''] = leaves_from_root[j]
            else:
                insert_branch_into_tree(d, leaves_from_root[j])
    
    if alphabet is not None:
        adapt_to_alphabet(root, alphabet)

    return root, tuple(alphaset) if alphabet is None else alphabet


def adapt_to_alphabet(root : dict, alphabet : tuple[str | tuple[int]]) -> None:
    # Handle characters in alphabet (which have a length > 1)
    alphaset = set(alphabet)
    maxlen = max(len(x) for x in alphaset)
    
    nodes_left = [(root, '')]
    chain = []
    node_pointers : list[dict] = [root]
    chain_length_by_node = [0]

    links_to_delete : dict[tuple, tuple[dict, Any]] = {}

    while len(nodes_left):
        current_node, last_char = nodes_left.pop()
        chain_len = chain_length_by_node.pop()
        chain = chain[:chain_len]
        node_pointers = node_pointers[:chain_len+1]

        if last_char != '':
            chain.append(last_char)
            node_pointers.append(current_node)

            for i in range(2, min(chain_len+1, maxlen)+1):
                # TODO: This for loop is optimizable
                if all(isinstance(x, str) for x in chain[-i:]):
                    multilength_ch = ''.join(itertools.chain(*chain[-i:]))
                else:
                    multilength_ch = tuple(itertools.chain(*chain[-i:]))
                if multilength_ch in alphaset:
                    past_node = node_pointers[-i-1]
                    d = past_node.get(multilength_ch)
                    # print('Inserting link of multi-character "%s"' % str(multilength_ch))
                    if d is None:
                        past_node[multilength_ch] = current_node
                    else:
                        insert_branch_into_tree(d, current_node)

            if len(ch := chain[-1]) == 1 and ch not in alphaset:
                # print("Removing links of character '%s'" % ch)
                link = (id(node_pointers[-2]), ch)
                if link not in links_to_delete:
                    links_to_delete[link] = (node_pointers[-2], ch)


        # Stop DFS when reaching leaf
        if not isinstance(current_node, dict):
            continue

        plus_one = 1 if last_char != '' else 0
        for k, v in current_node.items():            
            chain_length_by_node.append(chain_len + plus_one)
            nodes_left.append((v,k))

    # Delete links in tree because alphabet has shrinked
    for node, ch in links_to_delete.values():
        node.pop(ch, None)
    

def unfold_authorized_characters(where_am_i : dict | None, authorized : set):
    if where_am_i is None:
        return authorized
    if where_am_i is end_symb:
        authorized.add(where_am_i)
        return authorized
    for k,v in where_am_i.items():
        if len(k):
            authorized.add(k)
        else:
            unfold_authorized_characters(v, authorized)
    return authorized

def unfold_where_am_i(where_am_i : dict | None, current : dict) -> dict:
    if where_am_i is None:
        return current
    if where_am_i is end_symb:
        current[end_symb] = 0
        return current
    for k,v in where_am_i.items():
        if k is end_symb or k != '':
            vc = current.get(k)
            if vc is None:
                current[k] = v
            else:
                insert_branch_into_tree(vc, v)
        else:
            unfold_where_am_i(v, current)
    return current

                    

class MultiChoicesParser:
    """A efficient incremental parser for multi-choice grammars. They are defined as grammars of the form:

    start: list1 list2 ... listn

    list1: choice1_1 | choice1_2 | ... | choice1_k1

    list2: choice2_1 | choice2_2 | ... | choice2_k2

    ...
    
    listm: choicem_1 | choicem_2 | ... | choicem_km

    where choicex_y are all literals (strings) and can possibly be empty

    Example:
    start: det noun
    
    det: "the " | "an " | "a " | ""

    noun: "orange" | "apple" | "banana"

    This was particularly optimized when the size of the lists of choices is 
    very large (up to order of millions), which can be helpful
    to represent entities preceeded (or not) by a determinent. 
    For example, in Wikipedia, there are around 7 million entities (one article per entity).

    NOTE: It is possible to use other types of sequences that strings as choices, such as a list of integers.
    """
    def __init__(self, list_of_choices : list[list[str | tuple[int]]] | None, alphabet : list = None) -> None:
        """Initialize the parser using a list of choices (a list of lists) which correspond 
        to the lists introduced in the documentation of the class
        """
        if list_of_choices is not None:
            self.tree, self.alphabet = tree_from_list_of_choices(list_of_choices, alphabet)
        else:
            self.tree, self.alphabet = {}, tuple()
        self.reset()

    @staticmethod
    def init_empty() -> MultiChoicesParser:
        empty = MultiChoicesParser(None)
        return empty

    def next(self) -> tuple:
        """Returns all authorized tokens for the current state

        Returns:
            tuple: A tuple of characters or the End symbol 
        """
        if self.finished:
            return tuple()
        return tuple(unfold_authorized_characters(self.where_am_i, set()))
    
    def step(self, ch : str | int | tuple[int] | End) -> None:
        """Feed the character to the parser.

        Note: Feed the End symbol when the string to parse is finished.
        After this is done, the flag self.success will tell you if the parsed string is correct or not

        Args:
            ch (str): A charachter or End symbol 
        """
        assert isinstance(ch, (str,tuple,int,End))
        if self.finished:
            return
        
        # Format int to tuple
        if isinstance(ch,int):
            ch = (ch,)
        
        where_am_i_unfolded = unfold_where_am_i(self.where_am_i, dict())
        next = where_am_i_unfolded.get(ch)
        if next == 0 and ch is end_symb:
            self.success = True
            self.finished = True
        elif next is None:
            self.success = False
            self.finished = True
        elif ch is not end_symb:
            self.buf.append(ch)
        self.where_am_i = next
    
    def reset(self) -> None:
        """Reset the state of the parser.
        """
        self.finished = False
        self.success = False
        self.where_am_i = self.tree
        self.buf = []


    def copy(self, stateful=True) -> MultiChoicesParser:
        """Return a copy of this parser (stateful or not)"""
        c = MultiChoicesParser.init_empty()
        c.tree = self.tree
        c.alphabet = self.alphabet
        if stateful:
            c.finished = self.finished
            c.success = self.success
            c.where_am_i = self.where_am_i
            c.buf = list(self.buf)
        else:
            c.where_am_i = c.tree
        return c
    
    def accepts(self, string : str) -> bool:
        """Check whether the input string is correct according to this parser"""
        current = self.where_am_i
        for s in string:
            where_am_i_unfolded = unfold_where_am_i(current, dict())
            current = where_am_i_unfolded.get(s, None)
            if current is None:
                return False
            if current is end_symb:
                current = {end_symb : 0 if s is end_symb else None}
        return True

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, MultiChoicesParser):
            return False
        return self.alphabet is value.alphabet and self.tree is value.tree \
            and self.where_am_i is value.where_am_i and self.finished == value.finished and self.success == value.success
    
    def __hash__(self) -> int:
        return sum(hash(x) for x in (id(self.alphabet), id(self.where_am_i), 
                                     id(self.tree), self.finished, self.success))