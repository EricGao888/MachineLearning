import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# exName = ["gender", "race", "race_o", "samerace", "field", "decision"] these columns are not in bins
def nbc(t_frac):
    df_train = df_train_origin.sample(frac=t_frac, replace=False, random_state=47).reset_index(drop=True)
    return df_train

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

df_train_origin = pd.read_csv("trainingSet.csv")
df_test_origin = pd.read_csv("testSet.csv")
F = [0.01, 0.1, 0.2, 0.5, 0.6, 0.75, 0.9, 1]
for f in F:
    trainSet = nbc(f)
    testSet = df_test_origin.reset_index(drop=True)
    print("")
    (pro0, pro1, proMapList0, proMapList1) = train_model(trainSet)
    trainAccuracy = predict(pro0, pro1, proMapList0, proMapList1, trainSet)
    testAccuracy = predict(pro0, pro1, proMapList0, proMapList1, testSet)
    # print(trainAccuracy,testAccuracy)
    print("Fraction: %f" % f)
    print("Training Accuracy: %.2f" % trainAccuracy)
    print("Testing Accuracy: %.2f" % testAccuracy)

X = [0.01, 0.1, 0.2, 0.5, 0.6, 0.75, 0.9, 1]
Y1 = [1, 0.86, 0.79, 0.79, 0.78, 0.78, 0.78, 0.78]
Y2 = [0.48, 0.68, 0.71, 0.74, 0.74, 0.75, 0.75, 0.74]

# Data for plotting
fig, ax = plt.subplots()
ax.plot(X, Y1, label = "Training Accuracy")
ax.plot(X, Y2, label = "Test Accuracy")
ax.set(xlabel="Fraction", ylabel="Accuracy",
       title="Accuracy against Fraction")
# ax.set_ylim(min(Y1+Y2), max(Y1+Y2))
# ax.set_xlim(min(X),max(X))
ax.grid()
ax.legend()
plt.figure(figsize=(200,50))
plt.show()
fig.savefig("5_3.png")