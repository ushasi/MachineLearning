import numpy as np
from collections import Counter
import scipy
from scipy import stats 

freq = np.array([3, 6, 20, 10, 8, 3])
ages = np.array([135, 165, 195, 225, 255, 285])




#----------------------MEAN-------------------
print('Mean value is:')
sum_ = 0
for i in range(len(freq)):
   sum_ = sum_ + ages[i] * freq[i]

mean = sum_/sum(freq)
print(mean)

#----------------------MEDIAN-------------------
print('Median value is:')

n = sum(freq)
samples = np.zeros(n)
ctr = 0
for i in range(len(freq)):
   for j in range(freq[i]):
       samples[ctr] = ages[i]
       ctr = ctr + 1
print(samples)

if (n%2 == 0):
   median = 0.5 * (samples[n/2 - 1] + samples[n/2])
else:
   median = samples[(n+1)/2]

print(median)

#----------------------MODE-------------------
print('Mode value is:')

data = max(range(len(freq)), key=freq.__getitem__)
print(ages[data])

#-----------------Geometric mean------------------
print('Geometric mean is:')

gm = scipy.stats.gmean(samples, axis=0, dtype=None)
print(gm)


#-----------------Harmonic mean------------------
print('Harmonic mean is:')
hm = scipy.stats.hmean(samples, axis=0, dtype=None)
print(hm)

