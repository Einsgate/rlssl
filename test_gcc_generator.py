import matlab.engine
import numpy as np

import time
print("Starting Matlab...")
#eng = matlab.engine.start_matlab()
print("Matlab engine started.")
#A = matlab.int8([[3, 3, 1], [3, 3, 1]])
#print(eng.size(A, 1))

#res = matlab.double([1.2, 1.3, 1.4])


# start_time = time.time()
# for i in range(10):
# 	start_time2 = time.time()
# 	for j in range(10):
# 		#print("Iteration", i, "starts.")
# 		res = eng.gccGenerator(matlab.double([[2.9, 3.0, 1.5], [3.1, 3.0, 1.5]]),
# 						   0.2,
# 						   matlab.double([10.0, 10.0, 3.0]),
# 						   matlab.double([3.0, 9.0, 1.5]))
# 	res = res[0]
#
# 	elapsed_time = time.time() - start_time2
# 	print(elapsed_time)
#
# elapsed_time = time.time() - start_time
# print(elapsed_time)



# res = eng.gccGenerator(matlab.double([[0.5, 0.5 - 0.1, 1.5], [0.5, 0.5 + 0.1, 1.5]]),
# 						   0.2,
# 						   matlab.double([9.0, 9.0, 3.0]),
# 						   matlab.double([8.5, 4.5, 1.5]))
# res = res[0]
# print(res)
#print(len(res))
#print(res.index(max(res)))

#eng.quit()

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

y_5_5 = [1.080000,
0.880000,
0.870000,
1.650000,
1.730000,
1.280000,
1.410000,
0.980000,
1.700000,
0.980000,
1.280000,
0.990000,
1.890000,
4.490000,
1.380000,
3.120000,
1.150000,
2.040000,
1.620000,
1.040000]

y_7_7 = [
0.420000, 1.920000, 0.800000, 0.230000, 0.740000,
1.770000, 0.830000, 0.510000, 1.300000, 0.380000,
0.770000, 1.290000, 1.110000, 1.290000, 0.200000,
1.950000, 0.430000, 1.400000, 0.650000, 0.600000
]

y_9_9 = [0.6, 0.8, 2.64, 0.56, 1.06, 0.97, 1.54, 0.89, 2.59, 3.93,
		 0.900000, 2.900000, 1.190000, 0.390000, 0.890000,
		 1.190000, 1.450000, 0.650000, 0.600000, 1.460000
]

x = list(range(1,21))
# plt.figure(1)
# plt.subplot(211)

ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.xticks(np.arange(1, 21, 1))

plt.plot(x, y_5_5, color='green', marker='o', linewidth=1, markersize=4, label="5x5")
plt.plot(x, y_7_7, color='blue', marker='o', linewidth=1, markersize=4, label="7x7")
plt.plot(x, y_9_9, color='red', marker='o', linewidth=1, markersize=4, label="9x9")
# , linestyle='dashed'
plt.legend()
#plt.show()

plt.savefig("figures/avg_gaps.pdf")

plt.clf()


import pickle

with open('figures/r_10', 'rb') as f:
	y_10 = np.array(pickle.load(f))[0:3000]
	assert(len(y_10) == 3000)
with open('figures/r_dis', 'rb') as f:
	y_dis = np.array(pickle.load(f))[0:3000]
	assert (len(y_dis) == 3000)
with open('figures/r_us', 'rb') as f:
	y_us = np.array(pickle.load(f))[0:3000]
	assert (len(y_us) == 3000)

# normalize
y_10 = y_10/(max(y_10) - min(y_10))
y_dis = y_dis/(max(y_dis) - min(y_dis))
y_us = y_us/(max(y_us) - min(y_us))

y_10_ = []
y_dis_ = []
y_us_ = []

BATCH = 15
N = 200
x_ = x = list(range(1, N+1))

for i in range(N):
	assert(max(y_10[i*BATCH: i*BATCH+BATCH]) <= 1)
	assert(min(y_10[i*BATCH: i*BATCH+BATCH]) >= -1)
	y_10_.append(sum(y_10[i*BATCH: i*BATCH+BATCH]) / float(BATCH))
for i in range(N):
	y_dis_.append(sum(y_dis[i*BATCH: i*BATCH+BATCH]) / BATCH)
for i in range(N):
	y_us_.append(sum(y_us[i*BATCH: i*BATCH+BATCH]) / BATCH)

# plot figure 2

x = list(range(1, 3001))

from matplotlib.ticker import MaxNLocator
f = plt.figure(figsize=(9, 3))
ax = f.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.xticks(np.arange(0, len(y_10), 1000))

#f, axs = plt.subplots(2,2,figsize=(15,15))
#f.set_figheight(15)
#f.set_figwidth(15)
plt.gcf().subplots_adjust(bottom=0.15)
#plt.gcf().subplots_adjust(hspace=0.20)
plt.gcf().subplots_adjust(wspace=0.25)

ax1 = plt.subplot(1,3,1)
ax1.set_title("(a)")
plt.ylim((-1, 0.25))
plt.ylabel("Cumulated reward")
plt.plot(x_, y_10_, color='green', linewidth=1)

plt.subplot(1,3,2).set_title("(b)")
plt.ylim((-1.0, 0.25))
plt.xlabel("Episode")
plt.plot(x_, y_dis_, color='blue', linewidth=1)

plt.subplot(1,3,3).set_title("(c)")
plt.ylim((-1.0, 0.25))
plt.plot(x_, y_us_, color='red', linewidth=1)

#plt.legend()
#plt.show()

#plt.savefig("figures/reward_comparison.png")
plt.savefig("figures/reward_comparison.pdf")