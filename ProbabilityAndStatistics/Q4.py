import numpy as np
import math
arr = np.array([220.1, 219.2, 219, 221, 217, 215.5, 214.12, 215.7, 211.4, 210.9, 211.8, 212.4, 210.6, 209.5, 209.7, 208.9, 209.1, 209, 208.5, 208, 207.7, 207.6, 207.5, 207.4, 208.2, 206, 203.5, 203.7, 202.9, 200])
Bin_size = 5
Start = 200

#Code for generating frequency distribution
Table = np.zeros(math.ceil((max(arr)/Bin_size))-int(Start/Bin_size))

for i in range(len(arr)):
   Table[int(arr[i]/Bin_size)-int(Start/Bin_size)] += 1

print("Frequency Distribution is:")
for i in range(len(Table)):
    print("%s-%s" %(str(Start+(Bin_size*(i))),str(Start+(Bin_size*(i+1)))),int(Table[i]))

#Code for Histogram Plot

import numpy as np
import random
from matplotlib import pyplot as plt

bins = np.arange(Start, 5*(math.ceil((max(arr))/5)+1), 5) 
plt.xlim([min(arr), 225])

plt.hist(arr, bins=bins)
plt.title('Histogram of data')
plt.ylabel('count')

plt.savefig("Q4.png")