from itertools import combinations

a = [1,2,3,4]
b = [3,2,4,1]
c_a = combinations(a, 3)
c_b = combinations(sorted(b), 3)

for x in c_a:
    print(x)

for y in c_b:
    print(y)