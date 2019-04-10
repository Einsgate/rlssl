import matlab.engine
import time
print("Starting Matlab...")
eng = matlab.engine.start_matlab()
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

res = eng.gccGenerator(matlab.double([[0.5, 0.5 - 0.1, 1.5], [0.5, 0.5 + 0.1, 1.5]]),
						   0.2,
						   matlab.double([9.0, 9.0, 3.0]),
						   matlab.double([8.5, 4.5, 1.5]))
res = res[0]
print(res)
#print(len(res))
#print(res.index(max(res)))

eng.quit()