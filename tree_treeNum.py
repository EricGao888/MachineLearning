# python cv_numtrees.py /Users/ericgao/Documents/2019Spring/CS573/Output/proj4/trainingSet.csv
import random
import pandas as pd
import numpy as np
import math
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dataPath1 = sys.argv[1]
trainingDataFilename = dataPath1
df_train = pd.read_csv(trainingDataFilename)

f = open("treeNum_log.txt",'w')


featureSet = set(df_train)
featureSet.remove("decision")
depthThres = 8
numThre = 50
depth = 0
treeNum = 30

class Node:
	def __init__(self, example_df):
		self.left = None
		self.right = None
		self.feature = None
		self.example_df = example_df

def compute_accuracy(y_hat, y_label):
	cnt = 0
	N = len(y_label)
	for i in range(N):
		if y_hat[i] == y_label[i]:
			cnt += 1
	return 1.0*cnt/N

def standard_error(A):
    A_len = len(A)
    A_mean = 1.0*sum(A)/A_len
    A_sd = 0
    for i in A:
        A_sd += (i-A_mean)**2
    A_sd /= (1.0*A_len)
    A_sd = math.sqrt(A_sd)
    A_se = A_sd / math.sqrt(A_len)
    return A_se

def compute_gini(example_df):
	cnt0 = example_df[example_df["decision"] == 0].shape[0]
	cnt1 = example_df[example_df["decision"] == 1].shape[0]
	total = example_df.shape[0]
	gini = 1 - ((cnt0*1.0/total)**2 + (cnt1*1.0/total)**2)
	return gini 


def choose_feature(example_df):
	giniMap = dict()
	total = example_df.shape[0]
	for feature in featureSet:
		mask = example_df[feature] == 0
		example_df0 = example_df[mask]
		example_df1 = example_df[~mask]
		cnt0 = example_df[example_df[feature] == 0].shape[0]
		cnt1 = example_df[example_df[feature] == 1].shape[0]
		if example_df0.shape[0] == 0 or example_df1.shape[0] == 0:
			gain = compute_gini(example_df)
		else:
			gain = compute_gini(example_df) - (1.0*cnt0/total*compute_gini(example_df0) + 1.0*cnt1/total*compute_gini(example_df1))
		giniMap[feature] = gain
		# print("feature!")

	maxGain = -1
	for key in giniMap:
		if maxGain < giniMap[key]:
			maxGain = giniMap[key]
	for key in giniMap:
		if giniMap[key] == maxGain:
			return key
	print("Error!")


def split_example(feature, example_df):
	mask = example_df[feature] == 0
	example_df0 = example_df[mask]
	example_df1 = example_df[~mask]
	return example_df0, example_df1


def build_tree(example_df):
	global depth
	if example_df.shape[0] < numThre:
		return None
	if depth >= depthThres:
		return None
	if len(set(example_df["decision"])) == 1:
		return Node(example_df)
	if len(featureSet) == 0:
		return Node(example_df)

	node = Node(example_df)
	depth += 1

	feature = choose_feature(example_df)
	node.feature = feature
	# print feature
	featureSet.remove(feature)
	example_dfLeft, example_dfRight = split_example(feature, example_df)

	node.left = build_tree(example_dfLeft)
	node.right = build_tree(example_dfRight)

	featureSet.add(feature)
	depth -= 1
	return node

def train_singleTree(trainingSet):
	root = Node(trainingSet)
	feature = choose_feature(trainingSet)
	root.feature = feature
	featureSet.remove(feature)
	example_dfLeft, example_dfRight = split_example(feature, trainingSet)

	root.left = build_tree(example_dfLeft)
	root.right = build_tree(example_dfRight)	
	featureSet.add(feature)
	return root

def predict_singleTree(root, testSet):
	N = testSet.shape[0]
	y_hat = []
	for i in range(N):
		tmpNode = root
		while(True):
			if tmpNode.left != None and testSet.loc[i,tmpNode.feature] == 0:
				tmpNode = tmpNode.left
			elif tmpNode.right != None and testSet.loc[i,tmpNode.feature] == 1:
				tmpNode = tmpNode.right
			else:
				break
		cnt0 = list(tmpNode.example_df["decision"]).count(0)
		cnt1 = list(tmpNode.example_df["decision"]).count(1)
		res = 0 if cnt0 > cnt1 else 1
		y_hat.append(res)
	return y_hat	



def decisionTree(trainingSet, testSet):
	root = train_singleTree(trainingSet)
	y_hat_test = predict_singleTree(root,testSet)
	y_label_test = list(testSet["decision"])
	accuracy = compute_accuracy(y_hat_test, y_label_test)
	return accuracy


def bagging(trainingSet, testSet):
	global treeNum
	testSet = df_test
	rootList = []
	y_hat_testList = []
	y_hat_test = []
	for i in range(treeNum):
		# print i
		df_train_bagging = trainingSet.sample(frac=1, replace=True)
		root = train_singleTree(df_train_bagging)
		rootList.append(root)
	for root in rootList:
		y_hat_testList.append(predict_singleTree(root,testSet))
	for i in range(df_test.shape[0]):
		tmpList = []
		for j in range(treeNum):
			tmpList.append(y_hat_testList[j][i])
		res = 0 if tmpList.count(0) > tmpList.count(1) else 1
		y_hat_test.append(res)
	y_label_test = list(testSet["decision"])
	accuracy = compute_accuracy(y_hat_test, y_label_test)
	return accuracy

def randomForests(trainingSet, testSet):
	global featureSet
	global treeNum
	tmpSet = set()
	for feature_entry in featureSet:
		tmpSet.add(feature_entry)
	testSet = df_test
	rootList = []
	y_hat_testList = []
	y_hat_test = []
	p = int(math.sqrt(len(featureSet)))
	for i in range(treeNum):
		# print i
		featureSet = set(random.sample(featureSet, p))
		df_train_bagging = trainingSet.sample(frac=1, replace=True)
		root = train_singleTree(df_train_bagging)
		rootList.append(root)
		featureSet = tmpSet
	for root in rootList:
		y_hat_testList.append(predict_singleTree(root,testSet))

	for i in range(df_test.shape[0]):
		tmpList = []
		for j in range(treeNum):
			tmpList.append(y_hat_testList[j][i])
		res = 0 if tmpList.count(0) > tmpList.count(1) else 1
		y_hat_test.append(res)
	y_label_test = list(testSet["decision"])
	accuracy = compute_accuracy(y_hat_test, y_label_test)
	return accuracy

df_train = df_train.sample(frac=1,random_state=18).reset_index(drop=True)
df_train = df_train.sample(frac=0.5,random_state=32).reset_index(drop=True)
# print df_train.shape
# sc = df_train.shape[0]*0.9
# print sc
df_list = []

split_num = df_train.shape[0]/10
# print split_num
for i in range(10):
    df_list.append(df_train.iloc[split_num*i:split_num*i+260])

tList = [10, 20, 40, 50]
# tList = [1, 2]
# resD = []
resB = []
resR = []
# seD = []
seB = []
seR = []
for t_entry in tList:
	print("treeNum: %d" % t_entry)
	f.write(str(t_entry) + '\n')
	treeNum = t_entry
	# accuracyD = []
	accuracyB = []
	accuracyR = []
	for idx in range(10):
		print("Run fold%d" % idx)
		df_test = df_list[idx].reset_index(drop=True)
		df_list_copy = df_list[:]
		del df_list_copy[idx]
		df_train = pd.concat(df_list_copy).reset_index(drop=True)

		# accuracy = decisionTree(df_train, df_test)
		# f.write(str(accuracy) + ',')
		# accuracyD.append(accuracy)

		accuracy = bagging(df_train, df_test)
		f.write(str(accuracy) + ',')
		accuracyB.append(accuracy)

		accuracy = randomForests(df_train, df_test)
		f.write(str(accuracy) + '\n')
		accuracyR.append(accuracy)

	# resD.append(sum(accuracyD)/2.0)
	# seD.append(standard_error(accuracyD))
	resB.append(sum(accuracyB)/10.0)
	seB.append(standard_error(accuracyB))
	resR.append(sum(accuracyR)/10.0)
	seR.append(standard_error(accuracyR))
f.close()


fig, ax = plt.subplots()
# ax.errorbar(depthList, resD, yerr=np.array(seD)*0.2, label = "DT Accuracy")
ax.errorbar(tList, resB, yerr=np.array(seB)*0.2, label = "BT Accuracy")
ax.errorbar(tList, resR, yerr=np.array(seR)*0.2, label = "RF Accuracy")
ax.set(xlabel="Tree Number", ylabel="Accuracy",
       title="Accuracy against Tree Number")
# ax.set_ylim(min(Y1+Y2), max(Y1+Y2))
# ax.set_xlim(min(X),max(X))
ax.grid()
ax.legend()
plt.figure(figsize=(200,50))
plt.show()
# fig.savefig("5_3.png")
fig.savefig("treeNum.png")







