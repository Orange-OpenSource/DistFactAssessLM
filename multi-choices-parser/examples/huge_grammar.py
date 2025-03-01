import numpy as np

from multi_choices_parser import MultiChoicesParser, end_symb

# install numpy if needed
l = np.random.randint(0, 10**9, 1000000).astype(str)
l = [
    ['the', 'an', "a", ""],
    l
]
p = MultiChoicesParser(l)

to_parse = l[1][0]
print('String to parse: "%s"\n\n' % (to_parse,))
for i, c in enumerate(tuple(to_parse) + (end_symb, )):
    print('Step %s' % i)
    print("Authorized characters:", sorted(p.next()))
    print('Adding character:', c)
    p.step(c)
    print("State: Finished=%s, Success=%s" % (p.finished, p.success))
    print()