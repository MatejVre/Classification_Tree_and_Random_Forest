import numpy as np
import random
from collections import Counter

a = [1,2,3,4]

c = random.choices(range(len(a)), k = 4)
print(c)