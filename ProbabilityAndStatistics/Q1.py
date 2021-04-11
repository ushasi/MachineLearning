import numpy as np
from collections import Counter
import scipy
from scipy import stats 

arr = np.array([220.1, 219.2, 219, 221, 217, 215.5, 214.12, 215.7, 211.4, 210.9, 211.8, 212.4, 210.6, 209.5, 209.7, 208.9, 209.1, 209, 208.5, 208, 207.7, 207.6, 207.5, 207.4, 208.2, 206, 203.5, 203.7, 202.9, 200])


def selection_sort(nums):
    # This value of i corresponds to how many values were sorted
    for i in range(len(nums)):
        # We assume that the first item of the unsorted segment is the smallest
        lowest_value_index = i
        # This loop iterates over the unsorted items
        for j in range(i + 1, len(nums)):
            if nums[j] < nums[lowest_value_index]:
                lowest_value_index = j
        # Swap values of the lowest unsorted element with the first unsorted
        # element
        nums[i], nums[lowest_value_index] = nums[lowest_value_index], nums[i]


#----------------------MEAN-------------------
print('Mean value is:')
sum_ = 0
for i in range(len(arr)):
   sum_ = sum_ + arr[i]

mean = sum_/len(arr)
print(mean)
# OR
mean = np.mean(arr)
print(mean)
#----------------------MEDIAN-------------------
print('Median value is:')
selection_sort(arr)
print(arr)

n = len(arr)
if (n%2 == 0):
   median = 0.5 * (arr[int(n/2 - 1)] + arr[int(n/2)])
else:
   median = arr[int((n+1)/2)]

print(median)
#OR
median = np.median(arr)
print(median)
#----------------------MODE-------------------

n = len(arr) 

data = Counter(arr) 
get_mode = dict(data) 
mode = [k for k, v in get_mode.items() if v == max(list(data.values()))] 
  
if len(mode) == n: 
    get_mode = "No mode found"
else: 
    get_mode = "Mode is / are: " + ', '.join(map(str, mode)) 
      
print(get_mode)

#-----------------Geometric mean------------------
print('Geometric mean is:')
gm = scipy.stats.gmean(arr, axis=0, dtype=None)
print(gm)

#-----------------Harmonic mean------------------
print('Harmonic mean is:')
hm = scipy.stats.hmean(arr, axis=0, dtype=None)
print(hm)
