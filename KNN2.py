# CS57800 HW 1 KNN Implementation
# from matplotlib import pyplot as plt
# from matplotlib.ticker import MaxNLocator
# from collections import namedtuple
import numpy as np
from numpy import linalg as LA
import math
# from sklearn.metrics import f1_score

f0 = open("winequality-white.csv",'r')
checkList = f0.readlines()
del checkList[0] 
# print(len(checkList))
f0.close()

# # Data Sampling
# - Extract 200 lines of data from 4898

# f1 = open("subset.csv",'w')
# for cnt in range(0,200):
#     f1.write(checkList[cnt])
# f1.close()

# # Load data
# - Load data;
# - Normalization;
# - Split fold;

# Load data and shuffle 
arr = np.loadtxt(open("winequality-white.csv", "rb"),delimiter=";", skiprows=1) # Load data 
np.random.shuffle(arr) # Shuffle data

# Normalize data
arrNorm = LA.norm(arr, axis=0)
arrNorm[11] = 1
arrAverage = np.average(arr, axis=0)
arrAverage[11] = 0
# print("The shape of the dataset: " + str(arr.shape))
arr = (arr-arrAverage)/arrNorm # Normalize data

# Split data into 4 folds
foldNum = arr.shape[0]/4
fold1 = arr[:foldNum]
fold2 = arr[foldNum:2*foldNum]
fold3 = arr[2*foldNum:3*foldNum]
fold4 = arr[3*foldNum:]
# print(fold4[0:3,0:3])

# # Outline
# ## Steps
# - Shuffle data;
# - Normalization(Consider better method when optimizing);
# - Cut data into four folds(using line number and use a loop);
# - Using Validation Set to decide;
# - Establish two lists for Training Set and Validation Set as listTr and listV each time;
# - Establish a list for Test Set as ListTe and set the 12th colon into -1;

# ## Optimization
# - Implementing KNN without stacking by numpy
# - Optimising KNN with numpt through stacking

# # Predict Algorithm
# - Predict algo = distance algo + sort algo + rank algo

# # Three distance algorithm
# - Euclidean;
# - Manhatten;
# - Cosine Similarity;
# ## Parameters
# - Transfer in: Training Set Matrix, Test Set Matrix or Validation Set Matrix;
# - Return: Prediction Set with columns as quality and predicted quality: [quality:predicted quality];

# Construct basic predicting algorithm
def predict(distanceSet, K):
    distanceSorted = np.argsort(distanceSet[:,12].reshape(distanceSet.shape[0],1), axis=0)
    buckList = []
    maxNum = -1
    predictSorce = -1
    for i in range(0,11):
        buckList.append(0)
    for i in range(0,K):
        buckList[int(distanceSet[distanceSorted[i][0],11])] += 1
    for i in range(0,11):
        if maxNum < buckList[i]:
            maxNum = buckList[i]
            predictScore = i
    return predictScore

    # Construct three different predicting algorithms

def euclidean_predict(trainSet, testSet,K):
    compArr = np.zeros((testSet.shape[0],2))
    cnt = 0
    for testSetEle in testSet:
        tmp1 = LA.norm((trainSet[:,:11]-testSetEle.reshape(1,12)[:,:11]), axis=1)
        tmp2 = np.insert(trainSet, [12], tmp1.reshape(trainSet.shape[0],1), axis=1)
        predictScore = predict(tmp2,K)
        compArr[cnt][0] = int(testSet[cnt][11])
        compArr[cnt][1] = int(predictScore)
        cnt += 1
    return compArr

def manhattan_predict(trainSet, testSet, K):
    compArr = np.zeros((testSet.shape[0],2))
    cnt = 0
    for testSetEle in testSet:
        tmp1 = np.sum(np.abs(trainSet[:,:11]-testSetEle.reshape(1,12)[:,:11]),axis=1)
        tmp2 = np.insert(trainSet, [12], tmp1.reshape(trainSet.shape[0],1), axis=1)
        predictScore = predict(tmp2,K)
        compArr[cnt][0] = int(testSet[cnt][11])
        compArr[cnt][1] = int(predictScore)
        cnt += 1
    return compArr

def cosine_predict(trainSet, testSet, K):
    compArr = np.zeros((testSet.shape[0],2))
    cnt = 0
    for testSetEle in testSet:
        tmp1 = 1 - np.dot(trainSet[:,:11],testSetEle.reshape(1,12)[:,:11].T).reshape(trainSet.shape[0],1)/((LA.norm(trainSet[:,:11],axis=1))*(LA.norm(testSetEle.reshape(1,12)[:,:11],axis=1))).reshape(trainSet.shape[0],1)
        tmp2 = np.insert(trainSet, [12], tmp1.reshape(trainSet.shape[0],1), axis=1)
        predictScore = predict(tmp2,K)
        compArr[cnt][0] = int(testSet[cnt][11])
        compArr[cnt][1] = int(predictScore)
        cnt += 1
    return compArr

#     # Error function
# - Precision;
# - F1 Score;

def compute_accuracy(compArr):
    m = compArr.shape[0]
    n = 0
    for i in range(m):
        if compArr[i][0] == compArr[i][1]:
            n += 1
    return float(n)/m
    
    
def compute_F1(compArr):
    dictNum = {}
    dictTP = {}
    dictFP = {}
    dictFN = {}
    dictTN = {}
    dictPre = {}
    dictRec = {}
    dictF1 = {}
    F1 = 0
    for i in range(11):
        dictTP[i] = 0
        dictFP[i] = 0
        dictFN[i] = 0
        dictTN[i] = 0
        dictPre[i] = 0
        dictRec[i] = 0
        dictF1[i] = 0
    
    for key in range(11):
        for i in range(10):
            for j in range(compArr.shape[0]):
                if compArr[j][0] == i and compArr[j][1] == i:
                    dictTP[key] += 1
#                     print(compArr[j][0])
#                     print(i)
                elif compArr[j][0] == i and compArr[j][1] != i:
                    dictFN[key] += 1
                elif compArr[j][0] != i and compArr[j][1] == i:
                    dictFP[key] += 1
                elif compArr[j][0] != i and compArr[j][1] != i:
                    dictTN[key] += 1

    
    m = 12
    for i in range(11):
        if dictTP[i]+dictFP[i] == 0:
            dictPre[i] = 0
            m -= 1
        else:
            dictPre[i] = float(dictTP[i])/(dictTP[i]+dictFP[i]) 
            dictRec[i] = float(dictTP[i])/(dictTP[i]+dictFN[i])

    
    for key in range(11):
        if dictPre[key] == 0:
            dictF1[key] = 0
        else:
            dictF1[key] = 2*dictPre[key]*dictRec[key]/(dictPre[key]+dictRec[key])
            
    for key in dictF1:
        F1 += dictF1[key]
    
#     m = 11
    return F1/m
    
    
# # Choose hyperparameters on validation set
# - Choose distance algo on validation set using K = sqrt(N);
# - Choose K on Validation set using the distance algo chosen above;

def validateFun_distanceAlgo(foldListEle,Kp):
    K = Kp
    splitNum = foldListEle.shape[0]/5
    validationSet = foldListEle[:splitNum]
    trainSet = foldListEle[splitNum:]
    
    compArrE = euclidean_predict(trainSet,validationSet,K)
    compArrM = manhattan_predict(trainSet,validationSet,K)
    compArrC = cosine_predict(trainSet,validationSet,K)
    
    accuracyE = compute_accuracy(compArrE)
    accuracyM = compute_accuracy(compArrM)
    accuracyC = compute_accuracy(compArrC)
    
    F1E = compute_F1(compArrE)
    F1M = compute_F1(compArrM)
    F1C = compute_F1(compArrC)
    return accuracyE,accuracyM,accuracyC,F1E,F1M,F1C,K

def validateFun_K(foldListEle,KMax):
    KList = []
    accuracyList = []
    F1List = []

    splitNum = foldListEle.shape[0]/5
    validationSet = foldListEle[:splitNum]
    trainSet = foldListEle[splitNum:]
    
    for K in range(1,KMax,10):
        compArr = manhattan_predict(trainSet,validationSet,K)
        accuracy = compute_accuracy(compArr)
        F1 = compute_F1(compArr)
        KList.append(K)
        accuracyList.append(accuracy)
        F1List.append(F1)
        
    return KList,accuracyList,F1List
    

def tune_distanceAlgo():
    foldList = [fold1,fold2,fold3,fold4]

    F1E,F1M,F1C = [],[],[]
#     K = int(math.sqrt(min(fold1.shape[0],fold2.shape[0],fold3.shape[0],fold4.shape[0])*0.8))
    K = 21
    for foldListEle in foldList:
        accuracyE0,accuracyM0,accuracyC0,F1E0,F1M0,F1C0,K = validateFun_distanceAlgo(foldListEle,K)
#         accuracyE += accuracyE0
#         accuracyM += accuracyM0
#         accuracyC += accuracyC0
        F1E.append(F1E0)
        F1M.append(F1M0)
        F1C.append(F1C0)

    n_groups = 4

#     means_men = (20, 35, 30, 35, 27)
#     std_men = (2, 3, 4, 1, 2)

#     means_women = (25, 32, 34, 20, 25)
#     std_women = (3, 5, 2, 3, 3)

    fig, ax = plt.subplots(figsize=(8,6))

    index = np.arange(n_groups)
    bar_width = 0.2

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index - 0*bar_width, F1E, bar_width,
                    alpha=opacity, color='b',
                    error_kw=error_config,
                    label='Euclidean')

    rects2 = ax.bar(index + 1*bar_width, F1M, bar_width,
                    alpha=opacity, color='r',
                    error_kw=error_config,
                    label='Manhattan')

    rects3 = ax.bar(index + 2*bar_width, F1C, bar_width,
                    alpha=opacity, color='g',
                    error_kw=error_config,
                    label='Cosine')

    ax.set_xlabel('Distance Algorithm')
    ax.set_ylabel('Macro-F1 Sores')
    ax.set_title('Macro-F1 Score by Distance Algorithm of 4 Folds')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(('Fold1', 'Fold2', 'Fold3', 'Fold4'))
    ax.legend()

    fig.tight_layout()
    plt.show()

#     accuracyE = round((accuracyE/4)*100,1)
#     accuracyM = round((accuracyM/4)*100,1)
#     accuracyC = round((accuracyC/4)*100,1)
#     F1E = round((F1E/4)*100,1)
#     F1M = round((F1M/4)*100,1)
#     F1C = round((F1C/4)*100,1)

    

#     print("While K = "+str(K)+": ")
#     print("Eculidean: ")
#     print("Accuracy: "+str(accuracyE)+", "+"F1: "+str(F1E))
#     print("Manhattan: ")
#     print("Accuracy: "+str(accuracyM)+", "+"F1: "+str(F1M))
#     print("Cosine: ")
#     print("Accuracy: "+str(accuracyC)+", "+"F1: "+str(F1C))

def tune_K():
    accuracyList = []
    F1List = []
    foldList = [fold1,fold2,fold3,fold4]
    KMax = int(min(fold1.shape[0],fold1.shape[0],fold3.shape[0],fold4.shape[0])*0.8)
    for foldListEle in foldList:
        KList,accuracyList0,F1List0 = validateFun_K(foldListEle,KMax)
        accuracyList.append(accuracyList0)
        F1List.append(F1List0)
#     accuracyList1 = accuracyList[0]
#     accuracyList2 = accuracyList[1]
#     accuracyList3 = accuracyList[2]
#     accuracyList4 = accuracyList[3]
    F1List1 = F1List[0]
    F1List2 = F1List[1]
    F1List3 = F1List[2]
    F1List4 = F1List[3]
    plt.plot(KList,F1List1,label="Fold1")
    plt.plot(KList,F1List2,label="Fold2")
    plt.plot(KList,F1List3,label="Fold3")
    plt.plot(KList,F1List4,label="Fold4")
    leg = plt.legend(loc='best', ncol=1, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.show()

# tune_K()


# tune_distanceAlgo()

# Choose K as 35
K = 35
def model():
    foldList = [fold1,fold2,fold3,fold4]
    print("Hyper-parameters:")
    print("K: "+str(K))
    print("Distance measure: Manhattan Distance")
    print
    accuracyTrainA = []
    accuracyValidationA = []
    accuracyTestA = []
    F1TrainA = []
    F1ValidationA = []
    F1TestA = []

    
    for i in range(4):
        tmp1 = foldList[0-i]
        tmp2 = foldList[1-i]
        tmp3 = foldList[2-i]
        tmp4 = foldList[3-i]
        
        trainSet = np.append(tmp1,tmp2,axis=0)
        trainSet = np.append(trainSet,tmp3,axis=0)
        splitNum = int(trainSet.shape[0]*0.8)
        validationSet = trainSet[:splitNum,:]
        trainSet = trainSet[splitNum:,:]
        testSet = tmp4

        compArrTrain = manhattan_predict(trainSet,trainSet,K)
        compArrValidation = manhattan_predict(trainSet,validationSet,K)  
        compArrTest = manhattan_predict(trainSet,testSet,K) 

        accuracyTrain = compute_accuracy(compArrTrain)
        accuracyValidation = compute_accuracy(compArrValidation)
        accuracyTest = compute_accuracy(compArrTest)

        F1Train = compute_F1(compArrTrain)
        F1Validation = compute_F1(compArrValidation)
        F1Test = compute_F1(compArrTest)
        
        accuracyTrainA.append(accuracyTrain)
        accuracyValidationA.append(accuracyValidation)
        accuracyTestA.append(accuracyTest)
        
        F1TrainA.append(F1Train)
        F1ValidationA.append(F1Validation)
        F1TestA.append(F1Test)       
    
#     y_true = []
#     y_pred = []
#     for i in range(compArr.shape[0]):
#         y_true.append(compArr[i][0])
#         y_pred.append(compArr[i][1])
#     F1_ans =  f1_score(y_true, y_pred, average='macro')
        print("Fold-"+str(i+1)+":")
        print("Training: "+"F1 Score: "+str(round(F1Train*100,1))+", Accuracy: "+str(round(accuracyTrain*100,1))) 
        print("Validation: "+"F1 Score: "+str(round(F1Validation*100,1))+", Accuracy: "+str(round(accuracyValidation*100,1)))
        print("Test: "+"F1 Score: "+str(round(F1Test*100,1))+", Accuracy: "+str(round(accuracyTest*100,1)))
        print
    print("Average:")
    print("Training: "+"F1 Score: "+str(round(sum(F1TrainA)/float(4)*100,1))+", Accuracy: "+str(round(sum(accuracyTrainA)/float(4)*100,1))) 
    print("Validation: "+"F1 Score: "+str(round(sum(F1ValidationA)/float(4)*100,1))+", Accuracy: "+str(round(sum(accuracyValidationA)/float(4)*100,1)))
    print("Test: "+"F1 Score: "+str(round(sum(F1TestA)/float(4)*100,1))+", Accuracy: "+str(round(sum(accuracyTestA)/float(4)*100,1)))
    print
# model()
model()


