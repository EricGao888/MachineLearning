import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def make_bins(binNum, df0):
    df = df0.copy()
    rowNum = df.shape[0]
    exName = ["gender", "race", "race_o", "samerace", "field", "decision"]
    preName = ["attractive_important", "sincere_important", "intelligence_important", "funny_important",\
               "ambition_important", "shared_interests_important", "pref_o_attractive", "pref_o_sincere",\
               "pref_o_intelligence", "pref_o_funny", "pref_o_ambitious", "pref_o_shared_interests"] #[0,1]
    colName = list(df)

    # binList = []
    # for i in range(binNum):
    #     binList.append(dict())

    nameList = []

    split0 = (1 - 0) / float(binNum)
    split1 = (58 - 18) / float(binNum)
    split2 = (1 + 1) / float(binNum)
    split3 = (10 - 0) / float(binNum)
    for name in colName:
        if name in exName:
            continue
        else:   
            nameList.append(name)

        for i in range(rowNum):
            if name in preName:
                flag = df.loc[i,name] / split0
            elif name == "age" or name == "age_o":
                flag = (df.loc[i,name] - 18) / split1
            elif name == "interests_correlate":
                flag = (df.loc[i,name] + 1) / split2
            else:
                flag = df.loc[i,name] / split3


            df.loc[i,name] = int(flag - 0.0001)
    return df

# exName = ["gender", "race", "race_o", "samerace", "field", "decision"] these columns are not in bins


def nbc(t_frac):
    df_train = df_train_origin.sample(frac=t_frac, replace=False, random_state=47).reset_index(drop=True)
    return df_train

# print(df_train[:][0:1])
# print(list(df_train))
# df_train.apply(pd.value_counts)
# print(res)
    
def train_model(trainSet):
    df_train = trainSet.copy()
#     print(df_train)
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
#     print(df_train.loc[9,"decision"])
#     print(df_train.shape)

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
    return (pro0, pro1, proMapList0, proMapList1)

    # print(proMapList0[0])  
    # print(proMapList1[0])

def predict(pro0, pro1, proMapList0, proMapList1, testSet):
    df_test = testSet.copy()
    nameList = list(df_test)
    featureNum = len(nameList) - 1 #exclude the column decision which is used as label 
    testSize = df_test.shape[0]
    cnt = 0
#     print(df_test.loc[6,nameList[0]])

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

binList = [2,5,10,50,100,200]
df0 = pd.read_csv("dating.csv")
for binNum in binList:
    df = make_bins(binNum,df0)
    df_test = df.sample(frac=0.2, replace=False, random_state=47).reset_index(drop=True)
    df_train = df.copy()
    # df_train = pd.concat([df, df_test, df_test]).drop_duplicates(keep=False) 
    df_train.drop(df_test.index, axis=0,inplace=True)
    
    df_train_origin = df_train.copy().reset_index(drop=True)
    df_test_origin = df_test.copy().reset_index(drop=True)

    trainSet = nbc(1)
    testSet = df_test_origin
    
    (pro0, pro1, proMapList0, proMapList1) = train_model(trainSet)
    trainAccuracy = predict(pro0, pro1, proMapList0, proMapList1, trainSet)
    testAccuracy = predict(pro0, pro1, proMapList0, proMapList1, testSet)
    # print(trainAccuracy,testAccuracy)
    print("Bin size: %d" % binNum)
    print("Training Accuracy: %.2f" % trainAccuracy)
    print("Testing Accuracy: %.2f" % testAccuracy)

X = [2, 5, 10, 50, 100, 200]
Y1 = [0.75, 0.78, 0.79, 0.80, 0.80, 0.81]
Y2 = [0.70, 0.73, 0.73, 0.74, 0.74, 0.75]

# Data for plotting
fig, ax = plt.subplots()
ax.plot(X, Y1, label = "Training Accuracy")
ax.plot(X, Y2, label = "Test Accuracy")
ax.set(xlabel="bin_size", ylabel="Accuracy",
       title="Accuracy against bin_size")
# ax.set_ylim(min(Y1+Y2), max(Y1+Y2))
# ax.set_xlim(min(X),max(X))
ax.grid()
ax.legend()
plt.show()
fig.savefig("5_2.png")
