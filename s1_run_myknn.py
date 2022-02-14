import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

import myknn

##### Regression example

### Read data
#url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data")
#df = pd.read_csv(url, header=None)

### Categorical columns to numeric
#dfTmp = pd.get_dummies(df[df.columns[0]], drop_first=True)
#df = pd.concat([dfTmp, df.drop(0, axis=1)], axis=1)

### Read X and y
#X = np.array(df.iloc[:,0:-1])
#y = np.array(df.iloc[:,-1])

### Split to training and testing data
#XTr, XTe, yTr, yTe = train_test_split(X, y, test_size = 0.4, random_state = 4)

### Scale data
#scaler = MinMaxScaler()
#scaler.fit(XTr)
#XTrNorm = scaler.transform(XTr)
#XTeNorm = scaler.transform(XTe)

### Find nearest neighbors for test data
#k = 5
#predTe = myknn.myknn_reg(XTrNorm, yTr, XTeNorm, k)

### Evaluation of pred values
#print('Corr of pred values : ' + str(np.corrcoef(yTe, predTe)[0,1]))
#sns.regplot(x=yTe, y=predTe)
#plt.show()


#### Classification example

## Read data
url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
df = pd.read_csv(url, header=None)
df[4] = pd.factorize(df[4])[0]


## Read X and y
X = np.array(df.iloc[:,0:-1])
y = np.array(df.iloc[:,-1])

## Split to training and testing data
XTr, XTe, yTr, yTe = train_test_split(X, y, test_size = 0.4, random_state = 4)

## Scale data
scaler = MinMaxScaler()
scaler.fit(XTr)
XTrNorm = scaler.transform(XTr)
XTeNorm = scaler.transform(XTe)

## Find nearest neighbors for test data
k = 4
predTe = myknn.myknn_class(XTrNorm, yTr, XTeNorm, k)


## Evaluation of pred values
print('Corr of pred values : ' + str(accuracy_score(yTe, predTe)))
print(confusion_matrix(yTe, predTe))

