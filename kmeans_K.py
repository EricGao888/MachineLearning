# python kmeans_analysis1.py /Users/ericgao/Documents/2019Spring/CS573/DataSet/MNIST/digits-embedding.csv 
import numpy as np
from numpy import linalg as LA
import math
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import threading
from kmeans import clustering, compute_wc, compute_sc, compute_nmi

def main():
	start = time.time()
	dataPath = sys.argv[1]
	data = np.loadtxt(open(dataPath, "rb"),delimiter=",") # Load data 

	np.random.seed(0)

	data1 = data.copy()
	data3 = data[np.where(data[:,1] == 6)]
	data3 = np.concatenate((data[np.where(data[:,1] == 7)], data3), axis=0)
	data2 = np.concatenate((data[np.where(data[:,1] == 2)], data3), axis=0)
	data2 = np.concatenate((data[np.where(data[:,1] == 4)], data2), axis=0)
	dataList = [data1,data2,data3]
	# Analysis 1
	K_list = [2,4,8,16,32]

	for idx in range(3):
		wc_list = []
		sc_list = []
		for K in K_list:
			np.random.seed(0)
			classIdArr, repArr = clustering(dataList[idx],K)
			dataArr = np.insert(dataList[idx],[4],classIdArr,axis = 1) 
			wc = compute_wc(dataArr,K,repArr)
			sc = compute_sc(dataArr,K)

			# print wc
			wc_list.append(wc)
			sc_list.append(sc)

		fig, ax = plt.subplots()
		plt.plot(K_list, wc_list,"C0-o", ms = 3)
		ax.grid()
		ax.legend()
		ax.set(xlabel="K", ylabel="WC-SSD",
		       title="WC against K on dataset%d"%(idx+1))
		plt.figure(figsize=(200,50))
		plt.show()
		fig.savefig("wc%d.png"%(idx+1))

		fig, ax = plt.subplots()
		plt.plot(K_list, sc_list, "C1-o", ms = 3)
		ax.grid()
		ax.legend()
		ax.set(xlabel="K", ylabel="SC",
		       title="SC against K on dataset%d"%(idx+1))
		plt.figure(figsize=(200,50))
		plt.show()
		fig.savefig("sc%d.png"%(idx+1))

	end = time. time()
	runTime = end - start
	print("Run Time: %d min %f sec." % (runTime/60,runTime-int(runTime/60)*60))


if __name__ == '__main__':
	main()

