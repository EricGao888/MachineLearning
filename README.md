<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-MML-AM_CHTML' async></script>
# Machine Learning Implementation
- This is a repository of the implementation of basic Machine Learning algorithm from scratch.

## Content
- KNN
- NBC
- SVM
- AdaBoost
- Gradient Descent
- K-Means
- Decision Tree
- Perceptron
- Gradient Descent
- AdaBoost

## Format
- Each Algorithm has both .ipynb and .py format.

## Python Version
- Use Python 2.7 here. 

## Development Diary

### Settings 
- Check modules installed: Enter python shell

```python
help("modules")
```
- [Switch Python Version in Jupyter](https://ipython.readthedocs.io/en/latest/install/kernel_install.html)
- [Conada Switch Python Environment](https://conda.io/docs/user-guide/tasks/manage-environments.html?highlight=environment)

### Command Line
- Take arguments from command line(Python 3):

```python
import sys

# main
param_1= sys.argv[1] 
param_2= sys.argv[2] 
param_3= sys.argv[3]  
print('Params=', param_1, param_2, param_3)
```
- Read idx format file

```python
def read_idx(filename):
    with gzip.open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=numpy.uint8).reshape(shape)


```

- Sort a list with list element:

```python
from operator import itemgetter, attrgetter
recommenderList1 = sorted(recommenderList0,key=itemgetter(1),reverse=True)
```

### Numpy
- axis: 0 down, 1 across;
- Convert float to int: [```astype```](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.astype.html)
- [```numpy.argmax```](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html)
- ```insert```:

```python
A = np.array([[1,2],[3,4],[5,6]])
B = np.array([[1],[2],[3]])
print A
print
print B
C = np.insert(A,[2],B,axis=1)
print
print C
```
```
[[1 2]
 [3 4]
 [5 6]]

[[1]
 [2]
 [3]]

[[1 2 1]
 [3 4 2]
 [5 6 3]]
```
### Normalization

- Apply the same $\sigma$ and $\mu$ calculated on training set to the test set when doing normalization; 

### Macro-F1 Implementation
***Python Implementation of Macro-F1 Score for Multiclass***

#### References
- Use the result computed by sklearn as ans, [f1 computation reference](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
- Use the conception of [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix)

##### Recommended Reference from YouTube
- [Confusion Matrix](https://www.youtube.com/watch?v=t3H0xKjdaJ0&list=PLlmEZCU0l6maux5LkhtKBnpwkmR94WVk9&index=2)
- [Metrices for Multiclassification](https://www.youtube.com/watch?v=HBi-P5j0Kec&list=PLlmEZCU0l6maux5LkhtKBnpwkmR94WVk9&index=3)

#### Import Libraries
```python
import numpy as np
from sklearn.metrics import f1_score
import random
```
#### Implementation
- Assume we have to classify input into 11 labels as 11 continuous integer from 0 to 10
- Assume there may be missing labels during cross validation

```python
def compute_F1(y_true,y_pred):
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
```

```python
for x in range(10):
    y_true = []
    y_pred = []
    for i in range(100):
        y_true.append(random.randint(0,10))
        y_pred.append(random.randint(0,10))
#     print(y_true[:10])
#     print(y_pred[:10])
    f1 = compute_F1(y_true,y_pred)
    f1_ans = f1_score(y_true, y_pred, average='macro')    
#     print("F1: "+str(f1)+", F1 Answer: "+str(f1_ans))
```
###  Machine Learning Code Module(To Be Continued)
- Preprocess Data;
- Split Data Set;
- Train Model;
- Make Prediction;

```python
def preprocess(dataSet):
	# Combine data and label;
	# Normalization;
	# Shuffle dataSet;
	# Return preprocessed dataSet;

def split_dataSet(dataSet,splitRules):
	# Split data set into training set and test set according to split rules;
	# Return training set and test set;
	
def train_model(trainingSet,trainingParameters):
	# Train models
 	# Return parameters of the model;
 	
def predict(modelParameters,testSet):
	# Make prediction and compute evluation metrices

```


# MachineLearning
