import pandas as pd
import numpy as np
import math
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

trainingDataFilename = dataPath1 = sys.argv[1]
# testDataFilename = "/Users/ericgao/Documents/2019Spring/CS573/DataSet/Dating/testSet.csv"
df_train = pd.read_csv(trainingDataFilename)
# df_test = pd.read_csv(testDataFilename)
df_train_decision = df_train.pop("decision")
# df_test_decision = df_test.pop("decision")
df_train["decision"] = df_train_decision
# df_test["decision"] = df_test_decision
# print df_train.shape
# print df_test.shape
# print(df_train)
# df_train.shape[0]

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

#     y_hat = model_lr(trainSet,trainSet)
#     y_label = trainSet[-1:,:]
#     accuracy = compute_accuracy(y_hat,y_label)
#     print("Accuracy on training set: %.2f" % accuracy)

    y_hat = model_lr(trainSet,testSet)
    y_label = testSet[-1:,:]
    accuracy = compute_accuracy(y_hat,y_label)
    return accuracy
#     print("Accuracy on test set: %.2f" % accuracy)
    
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

#     y_hat = model_svm(trainSet,trainSet)
#     y_label = trainSet[-1:,:]
#     accuracy = compute_accuracy(y_hat,y_label)
#     print("Accuracy on training set: %.2f" % accuracy)

    y_hat = model_svm(trainSet,testSet)
    y_label = testSet[-1:,:]
    accuracy = compute_accuracy(y_hat,y_label)
    return accuracy
#     print("Accuracy on test set: %.2f" % accuracy)
# svm(trainSet,testSet)

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

df_train = df_train.sample(frac=1,random_state=18).reset_index(drop=True)
# print df_train.shape
sc = df_train.shape[0]*0.9
# print sc
df_list = []
for i in range(10):
    split_num = df_train.shape[0]/10
    df_list.append(df_train.iloc[split_num*i:split_num*i+520])

t_frac_list = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
lr_accuracy = []
svm_accuracy = []
lr_se = []
svm_se = []

for t_frac in t_frac_list:
    idx_list = list(range(10))
    lr_tmp = []
    svm_tmp = []
    for idx in idx_list:
        df_test = df_list[idx]
        df_list_copy = df_list[:]
        del df_list_copy[idx]
        df_train = pd.concat(df_list_copy)

        df_train = df_train.sample(frac=t_frac,random_state=32).reset_index(drop=True)
        trainSet = df_train.values
        testSet = df_test.values
        trainSet = np.insert(trainSet, [trainSet.shape[1]-1], [[1]], axis=1).T
        testSet = np.insert(testSet, [testSet.shape[1]-1], [[1]], axis=1).T
        lr_tmp.append(lr(trainSet,testSet))
        svm_tmp.append(svm(trainSet,testSet))
    lr_accuracy.append(1.0*sum(lr_tmp)/len(idx_list))
    lr_se.append(standard_error(lr_tmp))
    svm_accuracy.append(1.0*sum(svm_tmp)/len(idx_list))
    svm_se.append(standard_error(svm_tmp))

size_list = []
for i in t_frac_list:
    size_list.append(int(i*sc))
# print size_list
upperlimits = np.array([1, 0] * 3)
lowerlimits = np.array([0, 1] * 3)

fig, ax = plt.subplots()
ax.errorbar(size_list, lr_accuracy, yerr=lr_se, uplims=upperlimits, lolims=lowerlimits, label = "LR Accuracy")
ax.errorbar(size_list, svm_accuracy, yerr=svm_se, uplims=upperlimits, lolims=lowerlimits, label = "SVM Accuracy")
ax.set(xlabel="Training Size", ylabel="Accuracy",
       title="Accuracy against Training Size")
# ax.set_ylim(min(Y1+Y2), max(Y1+Y2))
# ax.set_xlim(min(X),max(X))
ax.grid()
ax.legend()
plt.figure(figsize=(200,50))
plt.show()
# fig.savefig("5_3.png")
fig.savefig("plot.png")

