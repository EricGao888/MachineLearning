# python knn.py /Users/ericgao/Documents/2019Spring/CS573/Output/proj4/trainingSet.csv /Users/ericgao/Documents/2019Spring/CS573/Output/proj4/testSet.csv
import random
import pandas as pd
import numpy as np
from numpy import linalg as LA
import math
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dataPath1 = sys.argv[1]
dataPath2 = sys.argv[2]
trainingDataFilename = dataPath1
testDataFilename = dataPath2

df_train = pd.read_csv(trainingDataFilename)
df_test = pd.read_csv(testDataFilename)

df_train = df_train.sample(frac=1,random_state=18).reset_index(drop=True)
df_test = df_train.sample(frac=1,random_state=18).reset_index(drop=True)


# Function Definition
def compute_accuracy(compArr):
    m = compArr.shape[0]
    n = 0
    for i in range(m):
        if compArr[i][0] == compArr[i][1]:
            n += 1
    return float(n)/m

def predict(distanceSet, K, distanceCol):
	classNum = 2
	distanceSorted = np.argsort(distanceSet[:,distanceCol].reshape(distanceSet.shape[0],1), axis=0)
	buckList = []
	maxNum = -1
	predictSorce = -1
	for i in range(0,classNum):
	    buckList.append(0)
	for i in range(0,K):
	    buckList[int(distanceSet[distanceSorted[i][0],distanceCol-1])] += 1
	for i in range(0,classNum):
	    if maxNum < buckList[i]:
	        maxNum = buckList[i]
	        predictScore = i
	return predictScore

def euclidean_predict(trainingSet, testSet, K):
	featureNum = trainingSet.shape[1] - 1
	compArr = np.zeros((testSet.shape[0],2))
	cnt = 0
	for testSetEle in testSet:
	    tmp1 = LA.norm((trainingSet[:,:featureNum]-testSetEle.reshape(1,featureNum+1)[:,:featureNum]), axis=1)
	    tmp2 = np.insert(trainingSet, [featureNum+1], tmp1.reshape(trainingSet.shape[0],1), axis=1)
	    predictScore = predict(tmp2,K,featureNum+1)
	    compArr[cnt][0] = int(testSet[cnt][featureNum])
	    compArr[cnt][1] = int(predictScore)
	    cnt += 1
	return compArr

def manhattan_predict(trainingSet, testSet, K):
	featureNum = trainingSet.shape[1] - 1
	compArr = np.zeros((testSet.shape[0],2))
	cnt = 0
	for testSetEle in testSet:
		tmp1 = np.sum(np.abs(trainingSet[:,:featureNum]-testSetEle.reshape(1,featureNum+1)[:,:featureNum]),axis=1)
		tmp2 = np.insert(trainingSet, [featureNum+1], tmp1.reshape(trainingSet.shape[0],1), axis=1)
		predictScore = predict(tmp2,K,featureNum+1)
		compArr[cnt][0] = int(testSet[cnt][featureNum])
		compArr[cnt][1] = int(predictScore)
		cnt += 1
	return compArr

def cosine_predict(trainingSet, testSet, K):
	featureNum = trainingSet.shape[1] - 1
	compArr = np.zeros((testSet.shape[0],2))
	cnt = 0
	for testSetEle in testSet:
	    tmp1 = 1 - np.dot(trainingSet[:,:featureNum],testSetEle.reshape(1,featureNum+1)[:,:featureNum].T).reshape(trainingSet.shape[0],1)/((LA.norm(trainingSet[:,:featureNum],axis=1))*(LA.norm(testSetEle.reshape(1,featureNum+1)[:,:featureNum],axis=1))).reshape(trainingSet.shape[0],1)
	    tmp2 = np.insert(trainingSet, [featureNum+1], tmp1.reshape(trainingSet.shape[0],1), axis=1)
	    predictScore = predict(tmp2,K,featureNum+1)
	    compArr[cnt][0] = int(testSet[cnt][featureNum])
	    compArr[cnt][1] = int(predictScore)
	    cnt += 1
	return compArr

# 10-Fold CV
# df_list = []
# split_num = df_train.shape[0]/10
# for i in range(10):
#     df_list.append(df_train.iloc[split_num*i:split_num*(i+1)])

# Tune Distance Function
# eAccuracyList = []
# mAccuracyList = []
# cAccuracyList = []
# for idx in range(10):
# 	# print("Run fold%d" % idx)
# 	df_test = df_list[idx].reset_index(drop=True)
# 	df_list_copy = df_list[:]
# 	del df_list_copy[idx]
# 	df_train = pd.concat(df_list_copy).reset_index(drop=True)

# 	trainingSet = df_train.values
# 	testSet = df_test.values

# 	compArr = euclidean_predict(trainingSet, testSet, 7)
# 	accuracy = compute_accuracy(compArr)
# 	eAccuracyList.append(accuracy)
# 	# print("Accuracy: %.2f" % accuracy) 

# 	compArr = manhattan_predict(trainingSet, testSet, 7)
# 	accuracy = compute_accuracy(compArr)
# 	mAccuracyList.append(accuracy)
# 	# print("Accuracy: %.2f" % accuracy) 

# 	compArr = cosine_predict(trainingSet, testSet, 7)
# 	accuracy = compute_accuracy(compArr)
# 	cAccuracyList.append(accuracy)

# eAccuracy = sum(eAccuracyList)/10.0
# mAccuracy = sum(mAccuracyList)/10.0	
# cAccuracy = sum(cAccuracyList)/10.0

# print("Accuracy of Euclidean: %.2f" % eAccuracy) 
# print("Accuracy of Manhattan: %.2f" % mAccuracy) 
# print("Accuracy of Cosine: %.2f" % cAccuracy) 

# # Tune k
# kList = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]
# yList = []
# for k in kList:
# 	# print("Run k = %d" % k)
# 	mAccuracyList = []
# 	for idx in range(10):
# 		# print("Run fold%d" % idx)
# 		df_test = df_list[idx].reset_index(drop=True)
# 		df_list_copy = df_list[:]
# 		del df_list_copy[idx]
# 		df_train = pd.concat(df_list_copy).reset_index(drop=True)

# 		trainingSet = df_train.values
# 		testSet = df_test.values

# 		# compArr = euclidean_predict(trainingSet, testSet, 7)
# 		# accuracy = compute_accuracy(compArr)
# 		# eAccuracyList.append(accuracy)
# 		# print("Accuracy: %.2f" % accuracy) 

# 		compArr = manhattan_predict(trainingSet, testSet, k)
# 		accuracy = compute_accuracy(compArr)
# 		mAccuracyList.append(accuracy)
# 		# print("Accuracy: %.2f" % accuracy) 

# 		# compArr = cosine_predict(trainingSet, testSet, 7)
# 		# accuracy = compute_accuracy(compArr)
# 		# cAccuracyList.append(accuracy)
# 	mAccuracy = sum(mAccuracyList)/10.0	
# 	yList.append(mAccuracy)

# fig, ax = plt.subplots()
# plt.plot(kList, yList)
# # ax.set_ylim(min(Y1+Y2), max(Y1+Y2))
# # ax.set_xlim(min(X),max(X))
# ax.grid()
# ax.legend()
# ax.set(xlabel="K", ylabel="Accuracy",
#        title="Accuracy against K")
# plt.figure(figsize=(200,50))
# plt.show()
# # fig.savefig("5_3.png")
# fig.savefig("knn.png")

trainingSet = df_train.values
testSet = df_test.values
compArr = manhattan_predict(trainingSet, testSet, 1)
accuracy = compute_accuracy(compArr)
# mAccuracyList.append(accuracy)
print("Test Accuracy with Manhattan Distance and k = 1: %.2f" % accuracy) 
# print("Test Accuracy with Manhattan Distance and k = 2: %.2f" % accuracy) 

