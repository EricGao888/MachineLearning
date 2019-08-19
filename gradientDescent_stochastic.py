# Import Libraries

import struct 
import numpy as np
import gzip
import sys
# from sklearn.metrics import f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import Data Set and Take Parameters From Cmd

var1 = sys.argv[1] 
var2 = sys.argv[2] 
foldPath = sys.argv[3]
filePath = foldPath + '/' 

if var1 == "True":
	regularization = True
else:
	regularization = False
fType = var2
learningRate = 0.001
lam = 100
criteria = 6

# Read Data

def read_idx(filename):
    with gzip.open(filePath+filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

trainImage = read_idx("train-images-idx3-ubyte.gz")
trainLabel = read_idx("train-labels-idx1-ubyte.gz")
testImage = read_idx("t10k-images-idx3-ubyte.gz")
testLabel = read_idx("t10k-labels-idx1-ubyte.gz")

# # Data Preprocess
# - Method 1: Scaling;
# - Method 2: Sub-sampling by max-pooling;
# - Shuffle data;

# print trainImage.shape
# print trainLabel.shape
# print testImage.shape
# print testLabel.shape
# print 


if fType == "type1":
    trainSet = np.insert((trainImage[:10000].reshape(10000,28*28))/255.0,[28*28],trainLabel[:10000].reshape(10000,1),axis=1)
    testSet = np.insert((testImage.reshape(10000,28*28))/255.0,[28*28],testLabel.reshape(10000,1),axis=1)
    trainSet = np.insert(trainSet,[28*28],1,axis=1)
    testSet = np.insert(testSet,[28*28],1,axis=1)
elif fType == "type2":
    mpList = [0,0,0,0]
    trainSet = np.zeros((10000,14*14))
    testSet = np.zeros((10000,14*14))
    
## Sub-sampling of training set
    for k in range(0,10000):
        cnt = 0
        for i in range(0,28,2):
            for j in range(0,28,2):
                mpList[0] = trainImage[k][i][j]
                mpList[1] = trainImage[k][i][j+1]
                mpList[2] = trainImage[k][i+1][j]
                mpList[3] = trainImage[k][i+1][j+1]
                trainSet[k][cnt] = max(mpList)
                cnt += 1
                
## Sub-sampling of test set    
    for k in range(0,10000):
        cnt = 0
        for i in range(0,28,2):
            for j in range(0,28,2):
                mpList[0] = testImage[k][i][j]
                mpList[1] = testImage[k][i][j+1]
                mpList[2] = testImage[k][i+1][j]
                mpList[3] = testImage[k][i+1][j+1]
                testSet[k][cnt] = max(mpList)
                cnt += 1
                
    trainSet = np.insert((trainSet[:10000].reshape(10000,14*14))/255.0,[14*14],trainLabel[:10000].reshape(10000,1),axis=1)
    testSet = np.insert((testSet.reshape(10000,14*14))/255.0,[14*14],testLabel.reshape(10000,1),axis=1)
    trainSet = np.insert(trainSet,[14*14],1,axis=1)
    testSet = np.insert(testSet,[14*14],1,axis=1)
    
np.random.shuffle(trainSet)
np.random.shuffle(testSet)
# print trainSet.shape
# print testSet.shape
# print trainSet[0]
# print
# print testSet[0]

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

def compute_accuracy(y_true,y_predict):
    size = y_true.shape[0]
    cnt = 0
    for i in range(size):
        if y_true[i] == y_predict[i]:
            cnt += 1
    return 1.0*cnt/size
#     return round(1.0*cnt/size,2)

if regularization == False:
    lam = 0

size = trainSet.shape[1] - 1
sampleNum = trainSet.shape[0]
W = np.random.randn(10,size)*np.sqrt(2.0/size)
# print(W.shape)
trainFeature = trainSet[:,:size]
testFeature = testSet[:,:size]
# print(trainFeature.shape)
trainLabel = trainSet[:,-1].reshape(sampleNum,1)
testLabel = testSet[:,-1].reshape(sampleNum,1)
# print trainFeature.shape
# print(y_true.shape)
trueMatrix = np.zeros((10,sampleNum))
for i in range(sampleNum):
    trueMatrix[int(trainLabel[i][0])][i] = 1

# for epoch in range(1000):
# predictMatrix = np.argmax(sigmoid(np.dot(W,trainFeature.T)).T,axis=1).reshape(sampleNum,1)

X = []
Y = []
Z = []
stopList = []
epoch = 1
flag = 1
while True:
    predictMatrix = sigmoid(np.dot(W,trainFeature.T))
    for i in range(sampleNum):
        sgdPredict = sigmoid(np.dot(W,trainFeature[i,:].reshape(1,size).T))
        W = (1-1.0*learningRate*lam/sampleNum)*W - learningRate*(sgdPredict-trueMatrix[:,i].reshape(10,1))*(trainFeature[i].reshape(1,size))
        W[:,-1] += 1.0*learningRate*lam/sampleNum*W[:,-1]
        
        if epoch % 50000 == 0:
            predictMatrix = sigmoid(np.dot(W,trainFeature.T))
            trainLoss = -np.sum(trueMatrix*np.log(predictMatrix)+(1-trueMatrix)*np.log(1-predictMatrix),axis=1)+lam/(2.0*sampleNum)*np.sum(W[:,:size-1]**2,axis=1)
            trainLoss = 1.0*np.average(trainLoss,axis=0)
            
            if len(stopList) == 20:
                del stopList[0]
            stopList.append(trainLoss)
## Convergency
            if len(stopList) == 20:
                improvement = np.average(stopList[:10])-np.average(stopList[10:20])
#                print("Improvement: ")+str(round(improvement,2))
                if improvement <= criteria:
                    flag = 0
                    break            
            
            trainPredict = np.argmax(sigmoid(np.dot(W,trainFeature.T)).T,axis=1).reshape(sampleNum,1)
            testPredict = np.argmax(sigmoid(np.dot(W,testFeature.T)).T,axis=1).reshape(sampleNum,1)
            trainAccuracy = compute_accuracy(trainLabel,trainPredict)
            testAccuracy = compute_accuracy(testLabel,testPredict)
            X.append(epoch)
            Y.append(trainAccuracy)
            Z.append(testAccuracy)            
            print("epoch: "+str(epoch)+" Training loss: "+str(round(trainLoss,2))+", Training Accuracy: "+str(round(trainAccuracy,2))+", "+"Test Accuracy: "+str(round(testAccuracy,2)))
        epoch += 1
    if flag == 0:
        break
# print trueMatrix.shape
# print predictMatrix.shape

# Data for plotting
fig, ax = plt.subplots()
ax.plot(X, Y, label = "Training Set")
ax.plot(X, Z, label = "Test Set")

ax.set(xlabel='Epoch', ylabel='Accuracy',
       title='Accuracy against Epoch')
ax.set_ylim(min(Y+Z), max(Y+Z))
ax.set_xlim(min(X),max(X))
ax.grid()
ax.legend()
fig.savefig("convergence.png")
# plt.show()
# print(X)