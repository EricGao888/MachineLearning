# python tree.py /Users/ericgao/Documents/2019Spring/CS573/Output/proj4/trainingSet.csv /Users/ericgao/Documents/2019Spring/CS573/Output/proj4/testSet.csv 2
import pandas as pd
import numpy as np
import sys
import math
import random

dataPath1 = sys.argv[1]
dataPath2 = sys.argv[2]
modelIdx = int(sys.argv[3])

trainingDataFilename = dataPath1
testDataFilename = dataPath2

df_train = pd.read_csv(trainingDataFilename)
df_test = pd.read_csv(testDataFilename)

# example_df = set(range(df_train.shape[0]))

featureSet = set(df_train)
featureSet.remove("decision")
depthThres = 8
numThre = 50
depth = 0

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
	# print(maxGain)
	for key in giniMap:
		if giniMap[key] == maxGain:
			return key
	# print("Error!")


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
	# print("Training Complete!")
	y_hat_train = predict_singleTree(root,trainingSet)
	y_hat_test = predict_singleTree(root,testSet)
	return y_hat_train, y_hat_test


def bagging(trainingSet, testSet):
	testSet = df_test
	rootList = []
	y_hat_trainList = []
	y_hat_testList = []
	y_hat_train = []
	y_hat_test = []
	treeNum = 30
	for i in range(treeNum):
		# print i
		df_train_bagging = trainingSet.sample(frac=1, replace=True)
		root = train_singleTree(df_train_bagging)
		rootList.append(root)
	for root in rootList:
		y_hat_trainList.append(predict_singleTree(root,trainingSet))
		y_hat_testList.append(predict_singleTree(root,testSet))

	for i in range(df_train.shape[0]):
		tmpList = []
		for j in range(treeNum):
			tmpList.append(y_hat_trainList[j][i])
		res = 0 if tmpList.count(0) > tmpList.count(1) else 1
		y_hat_train.append(res)

	for i in range(df_test.shape[0]):
		tmpList = []
		for j in range(treeNum):
			tmpList.append(y_hat_testList[j][i])
		res = 0 if tmpList.count(0) > tmpList.count(1) else 1
		y_hat_test.append(res)
	return y_hat_train, y_hat_test

def randomForests(trainingSet, testSet):
	global featureSet
	tmpSet = set()
	for feature_entry in featureSet:
		tmpSet.add(feature_entry)
	testSet = df_test
	rootList = []
	y_hat_trainList = []
	y_hat_testList = []
	y_hat_train = []
	y_hat_test = []
	p = int(math.sqrt(len(featureSet)))
	treeNum = 30
	for i in range(treeNum):
		# print i
		featureSet = set(random.sample(featureSet, p))
		df_train_bagging = trainingSet.sample(frac=1, replace=True)
		root = train_singleTree(df_train_bagging)
		rootList.append(root)
		featureSet = tmpSet
	for root in rootList:
		y_hat_trainList.append(predict_singleTree(root,trainingSet))
		y_hat_testList.append(predict_singleTree(root,testSet))

	for i in range(df_train.shape[0]):
		tmpList = []
		for j in range(treeNum):
			tmpList.append(y_hat_trainList[j][i])
		res = 0 if tmpList.count(0) > tmpList.count(1) else 1
		y_hat_train.append(res)

	for i in range(df_test.shape[0]):
		tmpList = []
		for j in range(treeNum):
			tmpList.append(y_hat_testList[j][i])
		res = 0 if tmpList.count(0) > tmpList.count(1) else 1
		y_hat_test.append(res)
	return y_hat_train, y_hat_test
			

def choose_model(modelIdx):
	if modelIdx == 1:
		y_hat_train, y_hat_test = decisionTree(df_train, df_test)
		y_label_train = list(df_train["decision"])
		y_label_test = list(df_test["decision"])
		trainAccuracy = compute_accuracy(y_hat_train, y_label_train)
		testAccuracy = compute_accuracy(y_hat_test, y_label_test)
		print("Training Accuracy DT: %.2f" % trainAccuracy)
		print("Test Accuracy DT: %.2f" % testAccuracy)

	elif modelIdx == 2:
		y_hat_train, y_hat_test = bagging(df_train, df_test)
		y_label_train = list(df_train["decision"])
		y_label_test = list(df_test["decision"])
		trainAccuracy = compute_accuracy(y_hat_train, y_label_train)
		testAccuracy = compute_accuracy(y_hat_test, y_label_test)
		print("Training Accuracy BT: %.2f" % trainAccuracy)
		print("Test Accuracy BT: %.2f" % testAccuracy)

	elif modelIdx == 3:
		y_hat_train, y_hat_test = randomForests(df_train, df_test)
		y_label_train = list(df_train["decision"])
		y_label_test = list(df_test["decision"])
		trainAccuracy = compute_accuracy(y_hat_train, y_label_train)
		testAccuracy = compute_accuracy(y_hat_test, y_label_test)
		print("Training Accuracy RF: %.2f" % trainAccuracy)
		print("Test Accuracy RF: %.2f" % testAccuracy)
		# test()


choose_model(modelIdx)




