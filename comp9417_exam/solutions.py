
## STUDENT ID: z5238743
## STUDENT NAME: Yu Zhang


## Question 2

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)       # make sure you run this line for consistency 
x = np.random.uniform(1, 2, 100)
y = 1.2 + 2.9 * x + 1.8 * x**2 + np.random.normal(0, 0.9, 100)
plt.scatter(x,y)
plt.show()


## (c)
c = 2
interation = 100
import math
def loss_function(c,x,y):
    result = math.sqrt((pow((x-y),2)/c/c+1))-1
    return result

def w0_cost_function(c,x,y,w0,w1):
    result =0
    i= 0
    while i < interation:
        cy1 = x[i]*w1 +w0
        ty1 = y[i]
        result = result + (-(1/pow(c,2))*(ty1-cy1)/(math.sqrt((1/pow(c,2))*pow((ty1-cy1),2)+1)))
        i = i +1
    return result

def w1_cost_function(c,x,y,w0,w1):
    result =0
    i = 0
    while i < interation:
        cy2 = x[i]*w1 + w0
        ty2 = y[i]
        result =result + (-(1/pow(c,2))*x[i]*(ty2 -cy2)/(math.sqrt((1/pow(c,2))*pow((ty2-cy2),2)+1)))
        i = i +1
    return result

def update_weight_function(w0,w1,c,x,y,alphy):
    nw0 = -alphy * w0_cost_function(c,x,y,w0,w1) + w0
    nw1 = -alphy * w1_cost_function(c,x,y,w0,w1) + w1
    return nw0,nw1

def run_function():
    lista =[]
    i = 0
    while i < 9:
        lista.append(1*pow(10,-i))
        i = i+1

    temploss =[]
    losses =[]
    for i in lista:
        temploss =[]
        w0 =1
        w1 =1
        j = 0
        while(j<interation):
            w0,w1 = update_weight_function(w0,w1,c,x,y,i)
            two_temploss = []
            k =0
            while(k<interation):
                cy = x[k]*w1 +w0
                two_temploss.append(loss_function(c,cy,y[k]))
                k = k + 1
            temploss.append(np.mean(two_temploss))
            j = j +1
        losses.append(temploss)
    return losses

losses = run_function()


## plotting help
fig, ax = plt.subplots(3,3, figsize=(10,10))
alphas = [10e-1, 10e-2, 10e-3,10e-4,10e-5,10e-6,10e-7, 10e-8, 10e-9]
for i, ax in enumerate(ax.flat):
    # losses is a list of 9 elements. Each element is an array of length 100 storing the loss at each iteration for that particular step size
    ax.plot(losses[i])
    ax.set_title(f"step size: {alphas[i]}")	 # plot titles	
plt.tight_layout()      # plot formatting
plt.show()










## Question 3

# (c)
# YOUR CODE HERE





# Question 5

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import time
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def create_dataset():
    X, y = make_classification( n_samples=1250,
                                n_features=2,
                                n_redundant=0,
                                n_informative=2,
                                random_state=5,
                                n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 3 * rng.uniform(size = X.shape)
    linearly_separable = (X, y)
    X = StandardScaler().fit_transform(X)
    return X, y

def plotter(classifier, X, X_test, y_test, title, ax=None):
    # plot decision boundary for given classifier
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                            np.arange(y_min, y_max, plot_step))
    Z = classifier.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    if ax:
        ax.contourf(xx, yy, Z, cmap = plt.cm.Paired)
        ax.scatter(X_test[:, 0], X_test[:, 1], c = y_test)
        ax.set_title(title)
    else:
        plt.contourf(xx, yy, Z, cmap = plt.cm.Paired)
        plt.scatter(X_test[:, 0], X_test[:, 1], c = y_test)
        plt.title(title)

# (a)
X, label = create_dataset()
X_train, X_test, label_train, label_test = train_test_split(X, label, test_size=0.20, random_state=0)
#print(X_train)
#print(label_train)
#print("   ")
#print(X_test)
#print(label_test)
#sample 画图 6个模型


SVCModel = SVC()
LogisticRegressionModel = LogisticRegression()
AdaBoostClassifierModel = AdaBoostClassifier()
RandomForestClassifiermodel = RandomForestClassifier()
DTmodel = DecisionTreeClassifier()
MLPClassifiermodel = MLPClassifier()
fig, ax = plt.subplots(3,2, figsize=(10,10))
title = ['SVC', 'LogisticRegression', 'AdaBoostClassifier', 'RandomForestClassifier','DecisionTreeClassifier','MLPClassifier']
modles = [SVCModel,LogisticRegressionModel,AdaBoostClassifierModel, RandomForestClassifiermodel, DTmodel, MLPClassifiermodel]
for i, ax in enumerate(ax.flat):
    m =  modles[i].fit(X_train,label_train)
    plotter(m, X, X_test,label_test,title[i],ax)
plt.tight_layout()      # plot formatting
plt.show()


# (b)
List_in_size = [50,100,200,300,400,500,600,700,800,900,1000]
SVCModellist = []
LogisticRegressionModellist = []
AdaBoostClassifierModellist = []
RandomForestClassifiermodellist = []
DTmodellist = []
MLPClassifiermodellist = []
append_list = [SVCModellist,LogisticRegressionModellist,AdaBoostClassifierModellist,RandomForestClassifiermodellist,DTmodellist,MLPClassifiermodellist]
SVCModelresult=[]
LogisticRegressionresult=[]
AdaBoostClassifierresult=[]
RandomForestClassifierresult=[]
DTmodelresult=[]
MLPClassifierresult=[]
resulttime = [SVCModelresult,LogisticRegressionresult,AdaBoostClassifierresult,RandomForestClassifierresult,DTmodelresult,MLPClassifierresult]

length_dataset = X_train.shape[0]
for i in List_in_size:
  #rate = i/length_dataset
  #X,label = create_dataset()
  index_traing =np.random.choice(length_dataset,size=i,replace =True)
  X_train_1 = X_train[index_traing]
  label_train_1 = label_train[index_traing]
  X_test = X_test
  label_test = label_test

  for modle in modles:
    temperate = []

    for k in range(0,10):
      time_record = 0
      strat_time = time.time()
      temperate = []
      modle.fit(X_train_1,label_train_1)
      temp = modle.predict(X_test)
      end_time = time.time()
      time_record = end_time - strat_time
      accuarcy = accuracy_score(label_test,temp)
      temperate.append(accuarcy)
    append_list[modles.index(modle)].append(np.mean(temperate))
    resulttime[modles.index(modle)].append(time_record)

#print(append_list)
l1=plt.plot(List_in_size,SVCModellist,color ='brown',label='SVC')
l2=plt.plot(List_in_size,LogisticRegressionModellist,color ='red',label='LR')
l3=plt.plot(List_in_size,AdaBoostClassifierModellist,color ='green',label='ADB')
l4=plt.plot(List_in_size,RandomForestClassifiermodellist,color ='orange',label='RF')
l5=plt.plot(List_in_size,DTmodellist,color ='blue',label='DT')
l6=plt.plot(List_in_size,MLPClassifiermodellist,color ='purple',label='MLP')
plt.title('Accuracy with training set increasing')
plt.xlabel('train set size')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# (c)
# YOUR CODE HERE
l7=plt.plot(List_in_size,SVCModelresult,color ='brown',label='SVC')
l8=plt.plot(List_in_size,LogisticRegressionresult,color ='red',label='LR')
l9=plt.plot(List_in_size,AdaBoostClassifierresult,color ='green',label='ADB')
l10=plt.plot(List_in_size,RandomForestClassifierresult,color ='orange',label='RF')
l11=plt.plot(List_in_size,DTmodelresult,color ='blue',label='DT')
l12=plt.plot(List_in_size,MLPClassifierresult,color ='purple',label='MLP')
plt.title('Training time with training set increasing')
plt.xlabel('train set size')
plt.ylabel('time cost')
plt.legend()
plt.show()

