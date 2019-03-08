
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
X=pd.read_csv("X.csv",header=None)
Y=pd.read_csv("y.csv",header=None)
df = pd.concat([X, Y], axis=1)
df=df.sample(frac=1)


# In[2]:


def Bayes_classifier(df_train,df_test):
    df_train_y1=df_train.loc[df_train.iloc[:,-1]==1]
    df_train_y0=df_train.loc[df_train.iloc[:,-1]==0]

    lamda_1=[]
    for n in range (0,54):  
        lamda_1.append((sum(df_train_y1.iloc[:,n])+1)/(df_train_y1.shape[0]+1))

    lamda_0=[]
    for n in range (0,54):  
        lamda_0.append((sum(df_train_y0.iloc[:,n])+1)/(df_train_y0.shape[0]+1))

    pi=df_train_y1.shape[0]/4600
    
    
    count_1_1=0
    count_0_0=0
    count_1_0=0
    count_0_1=0
    
    for i in range (0,460):
        term_1=1
        term_0=1
        for d in range (0,54):
            a=math.exp(-lamda_1[d])
            b=lamda_1[d]**df_test.iloc[i,d]
            #c=math.factorial(df_test.iloc[i,d])
            term_1=term_1*a*b
            if d==53:
                term_1=term_1*pi
        
            a=math.exp(-lamda_0[d])
            b=lamda_0[d]**df_test.iloc[i,d]
            #c=math.factorial(df_test.iloc[i,d])
            term_0=term_0*a*b       
            if d==53:
                term_0=term_0*(1-pi)
    
        if term_1>term_0:
            if df_test.iloc[i,-1]==1:
                count_1_1=count_1_1+1
            if df_test.iloc[i,-1]==0:
                count_1_0=count_1_0+1
        
        if term_1<term_0:
            if df_test.iloc[i,-1]==1:
                count_0_1=count_0_1+1
            if df_test.iloc[i,-1]==0:
                count_0_0=count_0_0+1
      
    
    result=np.array([[count_1_1,count_1_0],[count_0_1,count_0_0]])
    return result


# In[3]:


result=np.array([[0,0],[0,0]])

for i in range (0,10):
    df_test=df.iloc[i*460:(i+1)*460,:]
    df_train=df.drop(index=df_test.index.tolist())
    result=result+Bayes_classifier(df_train,df_test)
    


# In[4]:


print("Accuray:"+ str(sum(np.diag(result))/4600))


# In[6]:


result


# In[7]:


def poisson_parameters(df_train,df_test):
    df_train_y1=df_train.loc[df_train.iloc[:,-1]==1]
    df_train_y0=df_train.loc[df_train.iloc[:,-1]==0]

    lamda_1=[]
    for n in range (0,54):  
        lamda_1.append((sum(df_train_y1.iloc[:,n])+1)/(df_train_y1.shape[0]+1))

    lamda_0=[]
    for n in range (0,54):  
        lamda_0.append((sum(df_train_y0.iloc[:,n])+1)/(df_train_y0.shape[0]+1))
        
    return np.array(lamda_1), np.array(lamda_0)


# In[8]:


test_1=[]
test_0=[]
for x in range (0,54):
    test_1.append(0)
    test_0.append(0)
test_1=np.array(test_1)
test_0=np.array(test_0)

for i in range (0,10):
    df_test=df.iloc[i*460:(i+1)*460,:]
    df_train=df.drop(index=df_test.index.tolist())
    lamda_1,lamda_0=poisson_parameters(df_train,df_test)
    test_1=test_1+lamda_1
    test_0=test_0+lamda_0

test_1=test_1/10
test_0=test_0/10 


# In[32]:


import matplotlib.pyplot as plt
x=[]
for i in range (1,55):
    x.append(i)


# In[37]:


fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(111)
ax1 = plt.stem(x, test_1,linefmt='r-',label="y=1")
ax1 = plt.stem(x, test_0,linefmt='b-',label="y=0")
plt.legend()
plt.show()


# In[24]:


def KNN(df_test,df_train,k):
    df_test=df_test.values
    df_train=df_train.values
    
    count_1_1=0
    count_0_0=0

    for i in range (df_test.shape[0]):
        term_1=0
        term_0=0

        xi_distance=np.sum(np.abs(df_test[i,:54]-df_train[:,:54]),axis=1)
        index=np.argsort(xi_distance)[:k]

        for x in index:
                if df_train[x,54]==1:
                    term_1=term_1+1
                if df_train[x,54]==0:
                    term_0=term_0+1


        if term_1>=term_0:
            if df_test[i,-1]==1:
                count_1_1=count_1_1+1

        if term_1<term_0:
            if df_test[i,-1]==0:
                count_0_0=count_0_0+1
    return count_0_0+count_1_1


# In[25]:


KNN_accuray=[]

for k in range (1,21):
    result_KNN=0
    for i in range(0,10):
        df_test=df.iloc[i*460:(i+1)*460,:]
        df_train=df.drop(index=df_test.index.tolist())
        result_KNN=result_KNN+KNN(df_test,df_train,k)
    KNN_accuray.append(result_KNN/4600)
    
k=[]
for x in range(1,21):
    k.append(x)

plt.figure(figsize=(15, 6))
plt.plot(k, KNN_accuray)
plt.xticks(k)
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("Plot of the prediction accuracy of KNN Classifier as a function of k")
plt.show()


# In[15]:


X=pd.read_csv("X.csv",header=None)
Y=pd.read_csv("Y.csv",header=None)

for i in range(4600):
    if Y.iloc[i,0]==0:
        Y.iloc[i,0]=-1
X[55]=1
df = pd.concat([X, Y], axis=1)
df=df.sample(frac=1)


# In[16]:


def log(df_train):
    
    
    df_train=df_train.values

    W_0=[]
    Eta=0.01/4600
    for x in range(55):
        W_0.append(0)
   
    W_0=np.asarray(W_0)
    W_0=W_0.reshape((55, 1))

    lnalpha_all=[]
    
    for k in range(1000):
        spc=0
        lnalpha=0

        for i in range (4140):
            y_i=df_train[i,-1]
            x_i=df_train[i,:-1]
            x_i=x_i.reshape((1, 55))
            y_i=y_i.reshape((1,1))
            yx=np.matmul(y_i,x_i)
            yxw=np.matmul(yx,W_0)
            alpha=math.exp(yxw)/(1+math.exp(yxw))

            lnalpha=lnalpha+math.log(alpha)
            spc=spc+(1-alpha)*np.matmul(y_i,x_i).reshape(55,1)

        lnalpha_all.append(lnalpha)
        W_0=W_0+Eta*spc
    

    return lnalpha_all


# In[28]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(111)
for i in range(0,10):
    df_test=df.iloc[i*460:(i+1)*460,:]
    df_train=df.drop(index=df_test.index.tolist())
    plt.plot(log(df_train))

plt.show()


# In[32]:


import pandas as pd
import numpy as np
import math
from sklearn.utils import shuffle
import random

X=pd.read_csv("X.csv",header=None)
Y=pd.read_csv("Y.csv",header=None)

for i in range(4600):
    if Y.iloc[i,0]==0:
        Y.iloc[i,0]=-1
X[55]=1

df = pd.concat([X, Y], axis=1)
df=df.sample(frac=1)


# In[31]:


def newtown(df_train):
    
    df_train=df_train.values
    lnalpha_all=[]
    w=np.zeros((55,1))
    
    for i in range (100):
        p_all=[]
        lnalpha=0
        spc=0

        for i in range (4140):
                    y_i=df_train[i,-1]
                    x_i=df_train[i,:-1]
                    x_i=x_i.reshape((1, 55))
                    y_i=y_i.reshape((1,1))
                    yx=np.matmul(y_i,x_i)
                    yxw=np.matmul(yx,w)

                    
                    if yxw>=700:
                        lnalpha=lnalpha+0
                    else:
                        alpha=math.exp(yxw)/(1+math.exp(yxw))
                        lnalpha=lnalpha+math.log(alpha)

                    p=alpha*(1-alpha)  
                    p_all.append(p)               
                    spc=spc+(1-alpha)*np.matmul(y_i,x_i).reshape(55,1)
                    
                    
        lnalpha_all.append(lnalpha)
        H=np.diag(p_all)
        before=np.matmul(df_train[:,:-1].T,H)
        later=-np.matmul(before,df_train[:,:-1])
        inverse=np.linalg.inv(later)
        w=w-np.matmul(inverse,spc)
    return lnalpha_all


# In[23]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(111)
for i in range(0,10):
    df_test=df.iloc[i*460:(i+1)*460,:]
    df_train=df.drop(index=df_test.index.tolist())
    plt.plot(newtown(df_train))

plt.show()


# In[33]:


def newtown(df_train):
    
    df_train=df_train.values
    lnalpha_all=[]
    w=np.zeros((55,1))
    
    for i in range (100):
        p_all=[]
        lnalpha=0
        spc=0

        for i in range (4140):
                    y_i=df_train[i,-1]
                    x_i=df_train[i,:-1]
                    x_i=x_i.reshape((1, 55))
                    y_i=y_i.reshape((1,1))
                    yx=np.matmul(y_i,x_i)
                    yxw=np.matmul(yx,w)

                    
                    if yxw>=700:
                        lnalpha=lnalpha+0
                    else:
                        alpha=math.exp(yxw)/(1+math.exp(yxw))
                        lnalpha=lnalpha+math.log(alpha)

                    p=alpha*(1-alpha)  
                    p_all.append(p)               
                    spc=spc+(1-alpha)*np.matmul(y_i,x_i).reshape(55,1)
                    
                    
        lnalpha_all.append(lnalpha)
        H=np.diag(p_all)
        before=np.matmul(df_train[:,:-1].T,H)
        later=-np.matmul(before,df_train[:,:-1])
        inverse=np.linalg.inv(later)
        w=w-np.matmul(inverse,spc)
    return w


# In[34]:


def newton_log(df_test,df_train):
    
    df_test=df_test.values
    df_train=df_train
    w=newtown(df_train)
    
    count_1_1=0
    count_0_0=0
    count_1_0=0
    count_0_1=0

    for i in range (460):
        x=df_test[i,:-1].T
        xw=np.matmul(x,w)
        p=math.exp(xw)/(1+math.exp(xw))
        if p>=0.5:
            if df_test[i,-1]==1:
                count_1_1=count_1_1+1
            if df_test[i,-1]==-1:
                count_1_0=count_1_0+1
        if p<0.5:
            if df_test[i,-1]==-1:
                count_0_0=count_0_0+1
            if df_test[i,-1]==1:
                count_0_1=count_0_1+1

    result=np.array([[count_1_1,count_1_0],[count_0_1,count_0_0]])
    return result


# In[35]:


result=np.array([[0,0],[0,0]])
for i in range (0,10):
    df_test=df.iloc[i*460:(i+1)*460,:]
    df_train=df.drop(index=df_test.index.tolist())
    result=result+newton_log(df_test,df_train)


# In[36]:


result


# In[37]:


print("Accuray:"+ str(sum(np.diag(result))/4600))

