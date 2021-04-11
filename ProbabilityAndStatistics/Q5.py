import numpy as np
import math
import matplotlib.pyplot as plt

fig = plt.figure()
# fig = plt.figure(figsize = (10, 5))
bins = np.arange(120, 300, 30) 
plt.xlim([120,300])
students = [3,6,20,10,8,3]
plt.bar(bins,students,width=30,align='edge')
plt.title('Histogram of data')
plt.xlabel("Wages(Rs.)") 
plt.ylabel("Number of labourers") 
plt.savefig("Q5_1.png")


fig = plt.figure()
# fig = plt.figure(figsize = (10, 5))
bins = np.arange(120, 300, 30) 
plt.xlim([120,300])
students_cumsum = np.zeros((len(students)))
for i in range(len(students)):
    students_cumsum[i] = (i>0)*students_cumsum[i-1] + students[i]

plt.bar(bins,students_cumsum,width=30,align='edge')
plt.title('cumulative frequency plot')
plt.xlabel("Wages(Rs.)") 
plt.ylabel("Cumulative Number of labourers") 
plt.savefig("Q5_2.png")