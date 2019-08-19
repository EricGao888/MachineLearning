import struct 
import numpy as np
import sys
import gzip
# from sklearn.metrics import f1_score
# import matplotlib.pyplot as plt

trainingSize = int(sys.argv[1])
epoch = int(sys.argv[2]) 
learningRate = float(sys.argv[3])
foldPath = sys.argv[4]
filePath = foldPath + '/' 

def read_idx(filename):
    with gzip.open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

trainImage = read_idx(filePath + "train-images-idx3-ubyte.gz")
trainLabel = read_idx(filePath + "train-labels-idx1-ubyte.gz")
testImage = read_idx(filePath + "t10k-images-idx3-ubyte.gz")
testLabel = read_idx(filePath + "t10k-labels-idx1-ubyte.gz")

# print trainImage.shape
# print trainLabel.shape
# print testImage.shape
# print testLabel.shape
trainSet = np.insert(np.round((trainImage[:10000].reshape(10000,28*28))/255.0),[28*28],trainLabel[:10000].reshape(10000,1),axis=1)
testSet = np.insert(np.round((testImage.reshape(10000,28*28))/255.0),[28*28],testLabel.reshape(10000,1),axis=1)
trainSet = np.insert(trainSet,[28*28],1,axis=1)
testSet = np.insert(testSet,[28*28],1,axis=1)
np.random.shuffle(trainSet)
np.random.shuffle(testSet)
# print trainSet.shape
# print testSet.shape
# print trainSet[0]
# print
# print testSet[0]

def compute_F1(y_true,y_pred):
    N = len(y_true)
    L = 11
    confusionMatrix = np.zeros((11,11))
    dict_F1 = {}
    for i in range(11):
        dict_F1[i] = -1 
    for i in range(N):
        confusionMatrix[int(y_true[i])][int(y_pred[i])] += 1
    for i in range(11):
        if np.sum(confusionMatrix[i]) != 0:
            dict_F1[i] = 2.0*confusionMatrix[i][i]/\
            (np.sum(confusionMatrix[i])+np.sum(confusionMatrix[:,i]))
        elif np.sum(confusionMatrix[:,i]) != 0:
            dict_F1[i] = 0
        else:
            L -= 1
            
    sum_F1 = 0
    for key in dict_F1:
        if dict_F1[key] != -1:
            sum_F1 += dict_F1[key]
    return float(sum_F1)/L

def train_model(trainSize,epoch,learningRate):
#     W = np.random.randn(10,28*28+1)*np.sqrt(2.0/(28*28+1))
    learningRate = 1 - learningRate
    W = np.ones((10,28*28+1))
#     W = np.zeros((10,28*28+1))
    trainFeature = trainSet[:,0:28*28+1]
#     print trainFeature.shape
    y_true = np.zeros((trainSize,10))    
    for j in range(epoch):
        y_predict = np.zeros((trainSize,10))
        cnt = 0
        for i in range(trainSize):
            signMatrix = np.zeros((10,28*28+1))
            y_true[i][int(trainSet[i][28*28+1])] = 1
            y_predict[i] = y_predict[i] + np.dot(W,trainFeature[i:i+1,:].T).T
            if np.argmax(y_true[i]) != np.argmax(y_predict[i]):
#                 print(np.argmax(y_true[i]),np.argmax(y_predict[i]))
                cnt += 1
                signMatrix += trainFeature[i:i+1,:]
                signMatrix *= learningRate
                signMatrix[np.argmax(y_true[i])] /= (learningRate**2) 
                signMatrix += (np.ones((10,28*28+1)) - trainFeature[i:i+1,:])
                W = W*signMatrix
#         print cnt
    return W

def predict(paraMatrix,testSet):
    testFeature = testSet[:,0:28*28+1] 
    predictMatrix = np.dot(W,testFeature.T).T
    #     print predictMatrix.shape
    y_predict = (np.zeros((testSet.shape[0],1)) + np.argmax(predictMatrix,axis=1).reshape(testSet.shape[0],1)).astype(int)
    y_true = testSet[:,28*28+1:28*28+2].astype(int)

    F1 = compute_F1(y_true,y_predict)
    # F1_ans = f1_score(y_true, y_predict, average='macro')
    return round(F1*100,2)
    #     print("F1 Score: "+str(round(F1*100,2)))
    # print("Ans: "+str(round(F1_ans*100,2)))
    # for i in range(100):
    #     print(str(y_true[i])+"  "+str(y_predict[i]))


# # W = train_model(5000,100,0.0001)
# # predict(W)
# X = []
# Y = []
# Z = []

# for size in range(500,10000+1,250):
#     trainSetTMP = trainSet[:size,:]
#     W = train_model(size,50,0.0001)
#     X.append(size)
#     Y.append(predict(W,testSet))
#     Z.append(predict(W,trainSetTMP))
# # Data for plotting
# fig, ax = plt.subplots()
# ax.plot(X, Y, label = "Test Set")
# ax.plot(X, Z, label = "Training Set")

# ax.set(xlabel='Size of Training Set', ylabel='F1 Score',
#        title='F1-Score against Size of Training Set')
# ax.grid()
# ax.legend()
# #     fig.savefig("test.png")
# plt.show()


# X = []
# Y = []
# Z = []

# for epoch in range(10,100+1,5):
#     trainSetTMP = trainSet
#     W = train_model(10000,epoch,0.00001)
#     X.append(epoch)
#     Y.append(predict(W,testSet))
#     Z.append(predict(W,trainSetTMP))
# # Data for plotting
# fig, ax = plt.subplots()
# ax.plot(X, Y, label = "Test Set")
# ax.plot(X, Z, label = "Training Set")

# ax.set(xlabel='Epoch', ylabel='Macro-F1 Score',
#        title='Macro-F1 Score against Epoch')
# ax.grid()
# ax.legend()
# #     fig.savefig("test.png")
# plt.show()


# X = []
# Y = []
# Z = []

# for learningRate in [0.0001,0.0001,0.001,0.01]:
#     trainSetTMP = trainSet
#     W = train_model(10000,50,learningRate)
#     X.append(learningRate)
#     Y.append(predict(W,testSet))
#     Z.append(predict(W,trainSetTMP))
# # Data for plotting
# fig, ax = plt.subplots()
# ax.plot(X, Y, label = "Test Set")
# ax.plot(X, Z, label = "Training Set")

# ax.set(xlabel='Learning Rate', ylabel='Macro-F1 Score',
#        title='Macro-F1 Score against LearningRate')
# ax.grid()
# ax.legend()
# #     fig.savefig("test.png")
# plt.show()




W = train_model(trainingSize,epoch,learningRate)
trainSetTMP = trainSet
F1_trainSet = predict(W,trainSetTMP)
F1_testSet = predict(W,testSet)
print("Training F1 score: "+str(F1_trainSet))
print("Test F1 score: "+str(F1_testSet))


