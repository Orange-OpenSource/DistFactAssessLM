# Multi-choices Parser

## Overview
Multi-choices Parser is an pure-Python efficient incremental parser for multi-choices grammars. These grammars are composed of lists of choices, where each choice is a literal string and can possibly be empty (grammar form below). This parser is optimized for scenarios where the size of the lists of choices is very large, such as representing entities preceded by a determiner.


Here is the type of grammar handled by this parser:

```
start: list1 list2 ... listn
list1: choice1_1 | choice1_2 | ... | choice1_k1
list2: choice2_1 | choice2_2 | ... | choice2_k2
...
listm: choicem_1 | choicem_2 | ... | choicem_km
```

## Installation

```
pip install multi-choices-parser
```

## Features
- Handles large lists of choices efficiently (up to millions of choices).
- Incremental parsing.

## Usage
To use the `MultiChoicesParser`, follow these steps:

1. Initialize the parser with a list of choices.
3. Use the `step` method to feed characters to the parser.
4. Check the `success` flag to determine if the parsed string is correct after feeding the End symbol.
5. Reset the parser state using the `reset` method if needed.

### Example
```python

from multi_choices_parser.parser import MultiChoicesParser, end_symb

# Define your list of choices
l = [
    ['the', 'an', "a", ""],
    ['orange', 'apple', 'banana']
]

# Initialize the parser
p = MultiChoicesParser(l)

# Parse a string (don't forget to add the End symbol)
for i, c in enumerate(tuple("apple") + (end_symb, )):
    print('Step %s' % i)
    print("Authorized characters:", sorted(p.next()))
    print('Adding character:', c)
    p.step(c)
    print("State: Finished=%s, Success=%s" % (p.finished, p.success))
    print()
```

<details> <summary>Example Output</summary>

```
Step 0
Authorized characters: ['a', 'b', 'o', 't']
Adding character: a
State: Finished=False, Success=False

Step 1
Authorized characters: ['a', 'b', 'n', 'o', 'p']
Adding character: p
State: Finished=False, Success=False

Step 2
Authorized characters: ['p']
Adding character: p
State: Finished=False, Success=False

Step 3
Authorized characters: ['l']
Adding character: l
State: Finished=False, Success=False

Step 4
Authorized characters: ['e']
Adding character: e
State: Finished=False, Success=False

Step 5
Authorized characters: [End]
Adding character: End
State: Finished=True, Success=True
```

</details>

