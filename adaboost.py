# CS57800 HW 4 Ada Boost Implementation
# from matplotlib import pyplot as plt
# from matplotlib.ticker import MaxNLocator
from collections import namedtuple
import numpy as np
from numpy import linalg as LA
import math
import sys
# from sklearn.metrics import f1_score

# Load Data
# Load data and shuffle 
dataPath = sys.argv[1] 
arr = np.loadtxt(open(dataPath, "rb"),delimiter=";", skiprows=1) # Load data 
np.random.shuffle(arr) # Shuffle data

# Normalize data
arrNorm = LA.norm(arr, axis=0)
arrNorm[11] = 1
arrAverage = np.average(arr, axis=0)
arrAverage[11] = 0
# print("The shape of the dataset: " + str(arr.shape))
arr = (arr-arrAverage)/arrNorm # Normalize data
distribution = np.zeros((4898,1)) + float(1)/4898
arr = np.insert(arr, [12], distribution, axis=1)
# print(arr.shape)

# Split data into 4 folds
foldNum = arr.shape[0]/4
fold1 = arr[:foldNum]
fold2 = arr[foldNum:2*foldNum]
fold3 = arr[2*foldNum:3*foldNum]
fold4 = arr[3*foldNum:]
# print(fold4[0:3,0:3])

# Data Catagorizing
# - Easy way: Split the feature by zero since we have pre-processed the data with Z-Score;
# Data catagorizing
for i in range(11):
    for j in range(arr.shape[0]):
        if arr[j][i] < 0:
            arr[j][i] = -1
        else:
            arr[j][i] = 1
# print(arr)

# Split data into 4 folds
foldNum = arr.shape[0]/4
fold1 = arr[:foldNum]
fold2 = arr[foldNum:2*foldNum]
fold3 = arr[2*foldNum:3*foldNum]
fold4 = arr[3*foldNum:]
# print(fold4[0:3,0:3])
# - Optimal way: Split the feature throught candidate set to acquire the minimum entropy after catagorizing


# Decision Tree Implementation
class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.arr = None
        self.arrEmpty = True
        self.maxDepth = 0
        self.book = []
        self.feature = 0
        
        for i in range(11):
            self.book.append(0)

def chooseFeature(arr,book):
    pool = set()
    dictEntropy = {}
    dictPositiveCount = {}
    dictNegativeCount = {}
    labelBook = []
    positiveBook = []
    negativeBook = []
    dictHP = {}
    dictHN = {}
    for i in range(11):
        book[i] == 0
    for i in range(11):
        if book[i] == 0:
            pool.add(i)
    for i in range(10):
        labelBook.append(0)
    

    for i in range(11):
        dictEntropy[i] = 0
        positiveBook.append(0)
        negativeBook.append(0)
        dictPositiveCount[i] = labelBook
        dictNegativeCount[i] = labelBook
        dictHP[i] = 0
        dictHN[i] = 0
#     print(arr)
    for i in range(arr.shape[0]):
        for j in range(11):
            if arr[i][j] > 0:
                positiveBook[j] += arr[i][12]
                dictPositiveCount[j][int(arr[i][11])-1] += arr[i][12]
            else:
                negativeBook[j] += arr[i][12]
                dictNegativeCount[j][int(arr[i][11])-1] += arr[i][12]
#     print(positiveBook)        
    for key in range(11):
        for i in range(10):
            if positiveBook[i] == 0:
                p = 0
            else:
                p = float(dictPositiveCount[key][i])/positiveBook[i]
#                 print(p)
            if p == 0:
                pass
            else:
                dictHP[key] += p*math.log(1/p,2)
            
            if negativeBook[i] == 0:
                p = 0
            else:
                p = float(dictNegativeCount[key][i])/negativeBook[i]
            if p == 0:
                pass
            else:
                dictHN[key] += p*math.log(1/p,2)        
        
    for key in range(11):
        if positiveBook[key] + negativeBook[key] == 0:
            pass 
        else:
            ratioP = float(positiveBook[key])/(positiveBook[key]+negativeBook[key])
            ratioN = float(negativeBook[key])/(positiveBook[key]+negativeBook[key])
            dictEntropy[key] = ratioP*dictHP[key] + ratioN*dictHN[key]

    minNum = 10
    feature = -1
#     if len(pool) == 0:
# #         print "Error"
    for key in pool:
        if dictEntropy[key] < minNum:
            minNum = dictEntropy
            feature = key
#         else:
#             print dictEntropy[key]

    flagLeft = False
    flagRight = False
    if positiveBook[feature] > 0:
        flagRight = True
    if negativeBook[feature] > 0:
        flagLeft = True

    return feature,flagLeft,flagRight
    
    

def addNode(root,depthThreshold):
#     print("Depth: "+str(root.maxDepth))
    flagLeft = True
    flagRight = True
    

    
    checkLabel = set()
    for i in range(root.arr.shape[0]):
        checkLabel.add(root.arr[i][11])
    if len(checkLabel) == 1:
        flagLeft = False
        flagRight = False

    if sum(root.book) == 11:
        flagLeft = False
        flagRight = False
    else:
        tmp,flagLeft0,flagRight0 = chooseFeature(root.arr,root.book)    
        if flagLeft0 == False:
            flagLeft == False
        if flagRight0 == False:
            flagRight == False    
    
        root.feature = tmp
        root.book[tmp] = 1
        
    
    if root.maxDepth == depthThreshold:
        flagLeft = False
        flagRight = False
    
#     print(root.arr.shape)
#     print(root.arr[0])
    # start to create two leaves:
        

    
#     print(flagLeft,flagRight)
    if flagLeft:
        root.left = Tree()
        root.left.maxDepth = root.maxDepth + 1
        root.left.feature = tmp
        root.left.book = root.book
        root.left.book[tmp] = 1
        
    if flagRight:
        root.right = Tree()
        root.right.maxDepth = root.maxDepth + 1
        root.right.feature = tmp
        root.right.book = root.book
        root.right.book[tmp] = 1    
    
    

    for i in range(root.arr.shape[0]):
        if flagLeft:
            if root.arr[i][tmp] == -1:
                if root.left.arrEmpty == True:
                    root.left.arr = root.arr[i].reshape(1,root.arr.shape[1])
                    root.left.arrEmpty = False
                else:
                    root.left.arr = np.append(root.left.arr,root.arr[i].reshape(1,root.arr.shape[1]),axis = 0)
                    
        if flagRight:        
            if root.arr[i][tmp] == 1:            
                if root.right.arrEmpty == True:
                    root.right.arr = root.arr[i].reshape(1,root.arr.shape[1])
                    root.right.arrEmpty = False
                else:
                    root.right.arr = np.append(root.right.arr,root.arr[i].reshape(1,root.arr.shape[1]),axis = 0)        
        
    
    # recurse
    if root.left == None:
        flagLeft = False
    else:
        if root.left.arrEmpty == True:
            flagLeft = False
            
    if root.right == None:
        flagRight = False
    else: 
        if root.right.arrEmpty == True:
            flagRight = False
    if flagLeft:
#         print("Add left node ...")
        addNode(root.left,depthThreshold)
#         print("Add left node successfully!")
    if flagRight:
#         print("Add right node ...")
        addNode(root.right,depthThreshold)
#         print("Add right node successfully!")

# Error Function
def compute_accuracy(compArr):
    m = compArr.shape[0]
    n = 0
    for i in range(m):
        if compArr[i][0] == compArr[i][1]:
            n += 1
    return float(n)/m
    
    
def compute_F1(compArr):
    y_true = []
    y_pred = []
    for i in range(compArr.shape[0]):
        y_true.append(int(compArr[i][0]))
        y_pred.append(int(compArr[i][1]))
    N = len(y_true)
    L = 11
    confusionMatrix = np.zeros((11,11))
    dict_F1 = {}
    for i in range(11):
        dict_F1[i] = -1 
    for i in range(N):
        confusionMatrix[y_true[i]][y_pred[i]] += 1
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

# def compute_micro():
#     y_true = []
#     y_pred = []
#     for k in range(compArrTrain.shape[0]):
#         y_true.append(compArrTrain[k][0])
#         y_pred.append(compArrTrain[k][1])
#     F1_ans =  f1_score(y_true, y_pred, average='micro')

def train_model(trainSet,depthThreshold):
    root = Tree()
    root.arr = trainSet
#     print root.arr.shape
    root.maxDepth = 1
    addNode(root,depthThreshold)
    return root
# print(fold1.shape[0])


def predict(root,testSet):
    tmp = root
    scoreList = []
    trueList = []
    for i in range(testSet.shape[0]):
        cntList = []
        for cnt in range(10):
            cntList.append(0)
        while True:
            if testSet[i][root.feature] == -1:
                if root.left != None and root.left.arrEmpty == False:
                    root = root.left
                else:
                    break
            elif testSet[i][root.feature] == 1:
                if root.right != None and root.right.arrEmpty == False:
                    root = root.right
                else:
                    break
            
#             else:
#                 print(testSet[i][root.feature])
#                 print(root.feature)
        
        for j in range(root.arr.shape[0]):
            cntList[int(root.arr[j][11])] += 1
        scoreList.append(cntList.index(max(cntList))+1)
        trueList.append(int(testSet[i][11]))
        
        root = tmp
        
    scoreList = np.array(scoreList)
    trueList = np.array(trueList)    
#     print(scoreList.shape)
#     print(trueList.shape)
    arr = np.insert(trueList.reshape(testSet.shape[0],1),[1],scoreList.reshape(testSet.shape[0],1),axis=1)
#     print arr.shape
    return arr


# for depthThreshold in range(1,12):
#     root = train_model(fold1)
#     print("Start to predict...")
#     compArr = predict(root,fold2)
#     print compArr
#     accuracy = compute_accuracy(compArr)
#     F1 = compute_F1(compArr)
#     print("accuracy: "+str(accuracy))
#     print("F1: "+str(F1))

def validateFun(foldListEle):
    maxDepthList = []
    accuracyList = []
    F1List = []

    splitNum = foldListEle.shape[0]/5
    validationSet = foldListEle[:splitNum]
    trainSet = foldListEle[splitNum:]
    for maxDepth in range(1,12):
        depthThreshold = maxDepth
        root = train_model(trainSet,depthThreshold)
        compArr = predict(root,validationSet)
        accuracy = compute_accuracy(compArr)
        F1 = compute_F1(compArr)
        maxDepthList.append(maxDepth)
        accuracyList.append(accuracy)
        F1List.append(F1)
        
    return maxDepthList,accuracyList,F1List

def tune_maxDepth():
    accuracyList = []
    F1List = []
    foldList = [fold1,fold2,fold3,fold4]
    for foldListEle in foldList:
        maxDepthList,accuracyList0,F1List0 = validateFun(foldListEle)
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
    plt.plot(maxDepthList,F1List1,label="Fold1")
    plt.plot(maxDepthList,F1List2,label="Fold2")
    plt.plot(maxDepthList,F1List3,label="Fold3")
    plt.plot(maxDepthList,F1List4,label="Fold4")
    leg = plt.legend(loc='best', ncol=1, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.show()
# tune_maxDepth()


def ada_predict(rootList,alphaList,testSet):
    adaList = []
    preList = []
    for i in range(11):
        preList.append(0)
    result = np.zeros((testSet.shape[0],2))
    for i in range(len(rootList)):
        adaList.append(predict(rootList[i],testSet))
    for i in range(testSet.shape[0]):
#         for j in range((len(rootList))):
#             preList[adaList[j][i][1]] += alphaList[j]
        preList[int(testSet[i][11])] += alphaList[0]
#         preList[int(testSet[i][11])] += alphaList[1]
#         preList[int(testSet[i][11])] += alphaList[1]
        result[i][0] = int(testSet[i][11])
        result[i][1] = int(preList.index(max(preList)))
#         for j in range(len(preList)):
#             preList[j] = 0
#     print result[:10,:]  
#     print alphaList
    return result
            
# Choose K as 35
maxDepth = 1
def train_and_predict():
    print("Ada Boost:")
    foldList = [fold1,fold2,fold3,fold4]
#     print("Hyper-parameters:")
#     print("Max-Depth: "+str(maxDepth))
#     print
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
        tmpSet = trainSet
        splitNum = int(trainSet.shape[0]*0.8)
        validationSet = trainSet[:splitNum,:]
        trainSet = trainSet[splitNum:,:]
        testSet = tmp4
        rootList = []
        alphaList = []
        for boostRound in range(6):
            root = train_model(trainSet,maxDepth)
            compArrTrain = predict(root,trainSet)
            error = 0
            for tmpCnt in range(trainSet.shape[0]):
                if compArrTrain[tmpCnt][0] != compArrTrain[tmpCnt][1]:
                    error += trainSet[tmpCnt][12]
            error = error/np.sum(trainSet,axis=0)[12]     
#             print error
            alpha = math.log((1-error)/error, math.e) + math.log(11-1,math.e)
            for tmpCnt in range(trainSet.shape[0]):
                if compArrTrain[tmpCnt][0] == compArrTrain[tmpCnt][1]:
                    trainSet[tmpCnt][12] = trainSet[tmpCnt][12]*math.e**alpha
                elif compArrTrain[tmpCnt][0] != compArrTrain[tmpCnt][1]:
                    trainSet[tmpCnt][12] = trainSet[tmpCnt][12]*math.e**(-alpha)
            trainSet[tmpCnt][12] = np.sum(trainSet,axis=0)[12]
            rootList.append(root)
            alphaList.append(alpha)
        
        compArrTrain = ada_predict(rootList,alphaList,trainSet)
        
#         for tmpCnt in range()
#         compArrValidation = predict(root,validationSet)  
        compArrValidation = ada_predict(rootList,alphaList,validationSet)
#         compArrTest = predict(root,testSet) 
        compArrTest = ada_predict(rootList,alphaList,testSet) 

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
    

#         print "Micro: "+ str(F1_ans)
        print("Fold-"+str(i+1)+":")
        print("Training: "+"F1 Score: "+str(round(F1Train*100,1))+", Accuracy: "+str(round(accuracyTrain*100,1))) 
        print("Validation: "+"F1 Score: "+str(round(F1Validation*100,1))+", Accuracy: "+str(round(accuracyValidation*100,1)))
        print("Test: "+"F1 Score: "+str(round(F1Test*100,1))+", Accuracy: "+str(round(accuracyTest*100,1)))
        print
        trainSet = tmpSet
    print("Average:")
    print("Training: "+"F1 Score: "+str(round(sum(F1TrainA)/float(4)*100,1))+", Accuracy: "+str(round(sum(accuracyTrainA)/float(4)*100,1))) 
    print("Validation: "+"F1 Score: "+str(round(sum(F1ValidationA)/float(4)*100,1))+", Accuracy: "+str(round(sum(accuracyValidationA)/float(4)*100,1)))
    print("Test: "+"F1 Score: "+str(round(sum(F1TestA)/float(4)*100,1))+", Accuracy: "+str(round(sum(accuracyTestA)/float(4)*100,1)))
    print

train_and_predict()

