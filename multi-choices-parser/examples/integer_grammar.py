from multi_choices_parser import MultiChoicesParser, end_symb


l = [
    [[0,1,2,3], [0,1]],
    [[5,6,7,8], [0,1,5], []]
]
p = MultiChoicesParser(l)
print(sorted(p.alphabet))

for i, c in enumerate((0,1,2,3) + (end_symb, )):
    print('Step %s' % i)
    print("Authorized characters:", sorted(p.next()))
    print('Adding character:', c)
    p.step(c)
    print("State: Finished=%s, Success=%s" % (p.finished, p.success))
    print()