import math
from time import time

lower = 3
upper = 2000000

primes = [2]

t1 = time()
if (lower % 2) == 0:
    lower = lower + 1
for num in range(lower, upper + 1, 2):
    # all prime numbers are greater than 1
    for i in range(2, int(math.sqrt(num) + 1)):
        if (num % i) == 0:
            break
    else:
        primes.append(num)
t2 = time()

print(len(primes))
print(primes)
print('The CPU needed ' + str(t2 - t1) + ' seconds')
