import pandas as pd
import numpy as np
import sys

dataPath1 = sys.argv[1]
dataPath2 = sys.argv[2]
modelIdx = int(sys.argv[3])

trainingDataFilename = dataPath1
testDataFilename = dataPath2

df_train = pd.read_csv(trainingDataFilename)
df_test = pd.read_csv(testDataFilename)
df_train_decision = df_train.pop("decision")
df_test_decision = df_test.pop("decision")
df_train["decision"] = df_train_decision
df_test["decision"] = df_test_decision
# print df_train.shape
# print df_test.shape
# print(df_train)

trainSet = df_train.values
testSet = df_test.values
trainSet = np.insert(trainSet, [trainSet.shape[1]-1], [[1]], axis=1).T
testSet = np.insert(testSet, [testSet.shape[1]-1], [[1]], axis=1).T
# print trainSet[:-1,:].shape
# print trainSet[-1:,:].shape
# print(trainSet.shape)
# print(testSet.shape)
# print testSet[-2:-1,:]

def compute_accuracy(y_hat, y_label):
        cnt = 0
        for i in range(y_hat.shape[1]):
            if(y_hat[0][i] == int(y_label[0][i])):
                cnt += 1
        accuracy = float(cnt)/y_hat.shape[1]
        return accuracy

def lr(trainingSet,testSet):
    trainSet = trainingSet
    testSet = testSet
    def model_lr(trainSet,testSet):
        w = np.zeros((1,trainSet.shape[0]-1))
        lmd = 0.01
        iteration = 1000
        stepSize = 0.01
        tol = 10**(-6)
        w_pre = w.copy()
        for i in range(iteration):
            delta = np.sum((-trainSet[-1:,:]+1/(1+np.power(np.e,np.dot(-w,trainSet[:-1,:]))))*trainSet[:-1,:],axis=1).T + lmd*w
            w -= stepSize*delta
            if(np.sqrt(np.sum((w-w_pre)**2)) < tol):
                break
            w_pre = w.copy()
        
        y_hat = 1 / (1 + np.power(np.e,np.dot(-w,testSet[:-1,:])))
        
        for i in range(y_hat.shape[1]):
            y_hat[0][i] = 1 if y_hat[0][i] > 0.5 else 0
        return y_hat


#     def compute_accuracy(y_hat, y_label):
#         cnt = 0
#         for i in range(y_hat.shape[1]):
#             if(y_hat[0][i] == int(y_label[0][i])):
#                 cnt += 1
#         accuracy = float(cnt)/y_hat.shape[1]
#         return accuracy

    y_hat = model_lr(trainSet,trainSet)
    y_label = trainSet[-1:,:]
    accuracy = compute_accuracy(y_hat,y_label)
    print("Training Accuracy LR: %.2f" % accuracy)

    y_hat = model_lr(trainSet,testSet)
    y_label = testSet[-1:,:]
    accuracy = compute_accuracy(y_hat,y_label)
    print("Testing Accuracy LR: %.2f" % accuracy)
    
# lr(trainSet,testSet)
def svm(trainingSet, testSet):
    trainSet[-1] = trainSet[-1] * 2 - 1
    testSet[-1] = testSet[-1] * 2 - 1 
    def model_svm(trainingSet, testSet):
        trainSet = trainingSet
        testSet = testSet
        lmd = 0.01
        step_size = 0.5
        iteration = 500
        tol = 10**(-6)
        w = np.zeros((1,trainSet.shape[0]-1))
        w_pre = w.copy()
        for i in range(iteration):
            y_hat = np.dot(w,trainSet[:-1,:])
            delta2 = trainSet[-1:,:]*trainSet[:-1,:]
            mask = np.sign(1-trainSet[-1:,:]*y_hat)
            mask = mask + np.ones(mask.shape)
            mask /= 2
            delta2 *= mask
            delta1 = (1.0/trainSet.shape[1]*np.sum(-delta2,axis=1)).reshape(w.shape)
            w_tmp = w.copy()
            w_tmp[0][-1] = 0
            delta1 += (1.0/trainSet.shape[1])*lmd*w_tmp
            w -= step_size * delta1
            if(np.sqrt(np.sum((w-w_pre)**2)) < tol):
                break
            w_pre = w.copy()
        y_hat = np.dot(w,testSet[:-1,:])
        for i in range(y_hat.shape[1]):
            y_hat[0][i] = 1 if y_hat[0][i] > 0 else -1
        return y_hat

    y_hat = model_svm(trainSet,trainSet)
    y_label = trainSet[-1:,:]
    accuracy = compute_accuracy(y_hat,y_label)
    print("Training Accuracy SVM: %.2f" % accuracy)

    y_hat = model_svm(trainSet,testSet)
    y_label = testSet[-1:,:]
    accuracy = compute_accuracy(y_hat,y_label)
    print("Testing Accuracy SVM: %.2f" % accuracy)
# svm(trainSet,testSet)

if modelIdx == 1:
	lr(trainSet,testSet)
elif modelIdx == 2:
	svm(trainSet,testSet)



