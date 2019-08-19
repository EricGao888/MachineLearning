import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# exName = ["gender", "race", "race_o", "samerace", "field", "decision"] these columns are not in bins
df_train_origin = pd.read_csv("trainingSet.csv")
df_test_origin = pd.read_csv("testSet.csv")

def nbc(t_frac):
    df_train = df_train_origin.sample(frac=t_frac, replace=False, random_state=47)
    return df_train
# df_train_origin = df_train.copy().reset_index(drop=True)
# df_test_origin = df_test.copy().reset_index(drop=True)


trainSet = nbc(1).reset_index(drop=True)
testSet = df_test_origin

# print(df_train[:][0:1])
# print(list(df_train))
# df_train.apply(pd.value_counts)
# print(res)
    
def train_model(trainSet):
    df_train = trainSet.copy()
    trainSize = df_train.shape[0]
    nameList = list(df_train)
    featureNum = len(nameList) - 1 #exclude the column decision which is used as label 

    proMapList0 = []
    proMapList1 = []

    for i in range(featureNum):
        proMapList0.append(dict())
        proMapList1.append(dict())

    pro0 = 0
    pro1 = 0

    for i in range(trainSize):
        if df_train.loc[i,"decision"] == 0:
            pro0 += 1
            for j in range(featureNum):
                value = str(df_train.loc[i,nameList[j]])
                if value in proMapList0[j]:
                    proMapList0[j][value] += 1
                else:
                    proMapList0[j][value] = 1

        if df_train.loc[i,"decision"] == 1:
            pro1 += 1
            for j in range(featureNum):
                value = str(df_train.loc[i,nameList[j]])
                if value in proMapList1[j]:
                    proMapList1[j][value] += 1
                else:
                    proMapList1[j][value] = 1

    for i in range(len(proMapList0)):
        for key in proMapList0[i]:
            proMapList0[i][key] /= float(pro0)

    for i in range(len(proMapList1)):
        for key in proMapList1[i]:
            proMapList1[i][key] /= float(pro1)

    pro0 /= float(trainSize)
    pro1 /= float(trainSize)
    return(pro0, pro1, proMapList0, proMapList1)

    # print(proMapList0[0])  
    # print(proMapList1[0])

def predict(pro0, pro1, proMapList0, proMapList1, testSet):
    df_test = testSet.copy()
    nameList = list(df_test)
    featureNum = len(nameList) - 1 #exclude the column decision which is used as label 
    testSize = df_test.shape[0]
    cnt = 0

    for i in range(testSize):
        pre0 = 1
        pre1 = 1
        for j in range(featureNum):
            value = str(df_test.loc[i,nameList[j]])
            if value in proMapList0[j]:
                pre0 *= proMapList0[j][value]
            else:
                pre0 *= 0
    #             cntError += 1
            if value in proMapList1[j]:
                pre1 *= proMapList1[j][value]
            else:
                pre1 *= 0
    #             cntError += 1
    #     print(pre0,pre1)
        pre = 0 if pre0 > pre1 else 1
    #     print(pre)
        cnt = cnt + 1 if pre == df_test.loc[i,"decision"] else cnt
    return (cnt/float(testSize))
    # print(df_test[:][-10:-1])

(pro0, pro1, proMapList0, proMapList1) = train_model(trainSet)
trainAccuracy = predict(pro0, pro1, proMapList0, proMapList1, trainSet)
testAccuracy = predict(pro0, pro1, proMapList0, proMapList1, testSet)
# print(trainAccuracy,testAccuracy)
print("Training Accuracy: %.2f" % trainAccuracy)
print("Testing Accuracy: %.2f" % testAccuracy)


