# python kmeans.py /Users/ericgao/Documents/2019Spring/CS573/DataSet/MNIST/digits-embedding.csv 10
import numpy as np
from numpy import linalg as LA
import math
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import threading
import multiprocessing as mp

def clustering(data, K):
	repArr = data[np.random.randint(0, data.shape[0], size = K), 2:4]
	# new_repArr = np.zeros((K,2))
	# cntArr = np.zeros((K,1))
	new_repArr = repArr.copy()
	cntArr = np.ones((K,1))
	classIdArr = np.zeros((data.shape[0],1))

	for it in range(50):

		# flag = False
		# for i in range(data.shape[0]): 
		# 	disArr = LA.norm(data[i:i+1,2:4] - repArr, axis = 1) #Actually not distance but won't affect result
		# 	classId = np.argmin(disArr, axis = 0)
		# 	if classIdArr[i] != classId:
		# 		flag = True
		# 		classIdArr[i] = classId
		# 	cntArr[classId][0] += 1
		# 	new_repArr[classId][0] += data[i][2]
		# 	new_repArr[classId][1] += data[i][3]

##
		disMat = np.zeros((data.shape[0],K))
		for i in range(K):
			disArr = LA.norm(data[:,2:4] - repArr[i:i+1,:], axis = 1).reshape(data.shape[0],1)
			# print disArr.shape
			disMat[:,i:i+1] += disArr
		tmp_classIdArr = np.argmin(disMat, axis = 1).reshape(data.shape[0],1)
		if np.array_equal(classIdArr, tmp_classIdArr):
			break
		else:
			classIdArr = tmp_classIdArr.copy()

		for i in range(K):
			cntArr[i][0] += data[np.where(classIdArr[:,0] == i)].shape[0]
			new_repArr[i][0] = np.sum(data[np.where(classIdArr[:,0] == i)][:,2:3])
			new_repArr[i][1] = np.sum(data[np.where(classIdArr[:,0] == i)][:,3:4])
##

		# if flag == False:
			# break
		new_repArr = new_repArr / cntArr
		repArr = new_repArr.copy()
		# new_repArr = tmpArr
		cntArr = np.ones((K,1))
		# new_repArr = np.zeros((K,2))
		# cntArr = np.zeros((K,1))
		# classIdArr = np.zeros((data.shape[0],1))
	return classIdArr, repArr

# Compute wc
def compute_wc(wcArr, K, repArr):
	wc = 0
	# wcArr = np.insert(data,[4],classIdArr,axis = 1) 	

	for i in range(K):
		wc += np.sum(np.power(wcArr[np.where(wcArr[:,4] == i)][:,2:4] - repArr[i:i+1,:], 2))

	# print("WC-SSD: %.3f" % wc)
	return wc

# Compute sc
def compute_sc_single(scArr, K, results, index, comArr, split):
	sc = 0
	for i in range(scArr.shape[0]):
		aArr = np.delete(comArr,i+index*split,0)
		# aArr = np.delete(comArr,0,0)
		aArr = aArr[np.where(aArr[:,4] == scArr[i][4])] 
		# aArr = comArr[np.where(comArr[:,4] == scArr[i][4])] 
		if aArr.shape[0] == 0:
			sc += 0
		else: 
			bArr = comArr[np.where(comArr[:,4] != scArr[i][4])]
			A = np.sum(np.sqrt(np.sum(np.power(scArr[i:i+1,2:4] - aArr[:,2:4],2),axis = 1)),axis = 0) / aArr.shape[0] 
			B = np.sum(np.sqrt(np.sum(np.power(scArr[i:i+1,2:4] - bArr[:,2:4],2),axis = 1)),axis = 0) / bArr.shape[0]
			sc += (B - A)*1.0 / max(A,B)
	
	results[index] = sc
	# return sc
	# print results


def compute_sc(scArr, K):
	# print mp.cpu_count()
	threadNum = 32
	# results = [0] * threadNum
	manager = mp.Manager()
	results = manager.dict()

	split = scArr.shape[0] / threadNum
	threads = []
	for i in range(threadNum):
		if i != threadNum - 1:
			# process = threading.Thread(target=compute_sc_single, args=[scArr[i*split:(i+1)*split,:], K, results, i, scArr, split])
			process = mp.Process(target=compute_sc_single, args=[scArr[i*split:(i+1)*split,:], K, results, i, scArr, split])
			process.start()
			threads.append(process)
		else:
			# process = threading.Thread(target=compute_sc_single, args=[scArr[i*split:scArr.shape[0],:], K, results, i, scArr, split])
			process = mp.Process(target=compute_sc_single, args=[scArr[i*split:scArr.shape[0],:], K, results, i, scArr, split])
			process.start()
			threads.append(process)
	for process in threads:
		process.join()
	# return sum(results)*1.0 / scArr.shape[0]
	ans = 0
	for i in range(threadNum):
		ans += results[i]
	return ans*1.0 / scArr.shape[0] 	

# Compute nmi
def compute_nmi(nmiArr, labelList, K):
	# nmiArr = np.insert(data,[4],classIdArr,axis = 1) 
	ICG = 0
	HC = 0
	HG = 0

	for i in labelList:
		for j in range(K):
			tmpArr = nmiArr[np.where(nmiArr[:,1] == i)]
			tmpArr = tmpArr[np.where(tmpArr[:,4] == j)]
			pcg = tmpArr.shape[0]*1.0 / nmiArr.shape[0]
			pc = nmiArr[np.where(nmiArr[:,1] == i)].shape[0]*1.0 / nmiArr.shape[0]
			pg = nmiArr[np.where(nmiArr[:,4] == j)].shape[0]*1.0 / nmiArr.shape[0]
			if pcg == 0:
				ICG += 0
			else:
				ICG +=  pcg * math.log(pcg / (pc * pg),2)

	for i in range(10):
		pc = nmiArr[np.where(nmiArr[:,1] == i)].shape[0]*1.0 / nmiArr.shape[0]
		if(pc == 0):
			HC += 0
		else:
			HC += - pc * math.log(pc,2)

	for j in range(K):
		pg = nmiArr[np.where(nmiArr[:,4] == j)].shape[0]*1.0 / nmiArr.shape[0]
		if(pg == 0):
			HG += 0
		else:
			HG += - pg * math.log(pg,2)

	nmi = ICG*1.0 / (HC + HG) 

	# print("NMI: %.3f" % nmi)
	return nmi

def main():
	start = time.time()

	dataPath = sys.argv[1]
	K = int(sys.argv[2])
	data = np.loadtxt(open(dataPath, "rb"),delimiter=",") # Load data 

	np.random.seed(0)

	labelList = [0,1,2,3,4,5,6,7,8,9]
	classIdArr, repArr = clustering(data,K)
	dataArr = np.insert(data,[4],classIdArr,axis = 1) 
	print "WC: %.3f" % compute_wc(dataArr,K,repArr)
	# do not insert
	print "SC: %.3f" % compute_sc(dataArr,K)
	print "NMI: %.3f" % compute_nmi(dataArr, labelList, K)
	end = time. time()
	runTime = end - start
	print("Run Time: %d min %f sec." % (runTime/60,runTime-int(runTime/60)*60))

if __name__ == '__main__':
	main()