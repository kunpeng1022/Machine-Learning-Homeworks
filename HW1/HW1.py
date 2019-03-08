import pandas as pd
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

file_trainX=pd.read_csv("X_train.csv",header=None)
file_trainY=pd.read_csv("Y_train.csv",header=None)

WRR=[]
df=[]
X=file_trainX.values
Y=file_trainY.values
for i in range (0,5001):
    lamda=i
    xtranspose = np.transpose(X)
    X_transX=np.matmul(xtranspose, X)
    lI=lamda*np.identity(file_trainX.shape[1])
    first=lI+X_transX
    first=inv(first)
    WRR.append(np.matmul(np.matmul(first,xtranspose),Y))
    u, s, vh = np.linalg.svd(X,full_matrices=False)
    df.append(sum(np.square(s)/(lamda+np.square(s))))
    
x1=[]
x2=[]
x3=[]
x4=[]
x5=[]
x6=[]
x7=[]

for i in range (0,5001):
        x1.append(float(WRR[i][0]))
        x2.append(float(WRR[i][1]))
        x3.append(float(WRR[i][2]))
        x4.append(float(WRR[i][3]))
        x5.append(float(WRR[i][4]))
        x6.append(float(WRR[i][5]))
        x7.append(float(WRR[i][6]))

fig = plt.figure(figsize=(9, 6))
ax1 = fig.add_subplot(111)
ax1.plot(df, x1,linewidth=1,label="cylinders")
ax1.plot(df, x2,linewidth=1,label="displacement")
ax1.plot(df, x3,linewidth=1,label="horsepower")
ax1.plot(df, x4,linewidth=1,label="weight")
ax1.plot(df, x5,linewidth=1,label="acceleration")
ax1.plot(df, x6,linewidth=1,label="year made")
ax1.plot(df, x7,linewidth=1,label="w_0")
lgd = ax1.legend(loc=9, bbox_to_anchor=(0.15,0.4))
ax1.set_xlabel("d(λ)")


file_testX=pd.read_csv("X_test.csv",header=None)
file_testY=pd.read_csv("Y_test.csv",header=None)
X_test=file_testX.values
Y_test=file_testY.values

RMSE=[]
for i in range(0,101):
    residual=np.dot(X_test,WRR[i])-Y_test
    residual=np.square(residual)
    RMSE.append(np.sqrt(np.sum(residual)/42))
fig = plt.figure(figsize=(9, 6))
ax1 = fig.add_subplot(111)

lamda_list=[]
for i in range(0,51):
    lamda_list.append(i)

ax1.plot(lamda_list,RMSE[:51])
ax1.set_xlabel("λ")
ax1.set_ylabel("RMSE")


X_2_train=np.power(X[:,:6],2)
X_2_train_record=X_2_train
X_2_train=(X_2_train-np.mean(X_2_train,axis=0))/np.std(X_2_train,axis=0)
X_2_train=np.concatenate((X, X_2_train), axis=1)

X_3_train=np.power(X[:,:6],3)
X_3_train_record=X_3_train
X_3_train=(X_3_train-np.mean(X_3_train,axis=0))/np.std(X_3_train,axis=0)
X_3_train=np.concatenate((X_2_train, X_3_train), axis=1)


WRR_2=[]
X=X_2_train
Y=file_trainY.values
for i in range (0,101):
    lamda=i
    xtranspose = np.transpose(X)
    X_transX=np.matmul(xtranspose, X)
    lI=lamda*np.identity(X_2_train.shape[1])
    first=lI+X_transX
    first=inv(first)
    WRR_2.append(np.matmul(np.matmul(first,xtranspose),Y))
    

WRR_3=[]
X=X_3_train
Y=file_trainY.values
for i in range (0,101):
    lamda=i
    xtranspose = np.transpose(X)
    X_transX=np.matmul(xtranspose, X)
    lI=lamda*np.identity(X_3_train.shape[1])
    first=lI+X_transX
    first=inv(first)
    WRR_3.append(np.matmul(np.matmul(first,xtranspose),Y))

    
file_testX=pd.read_csv("X_test.csv",header=None)
file_testY=pd.read_csv("Y_test.csv",header=None)


X_test=file_testX.values
Y_test=file_testY.values


X_2_test=np.power(X_test[:,:6],2)
X_2_test=(X_2_test-np.mean(X_2_train_record,axis=0))/np.std(X_2_train_record,axis=0)
X_test=np.concatenate((X_test, X_2_test), axis=1)


X_3_test=np.power(X_test[:,:6],3)
X_3_test=(X_3_test-np.mean(X_3_train_record,axis=0))/np.std(X_3_train_record,axis=0)
X_test_3=np.concatenate((X_test, X_3_test), axis=1)




RMSE_2=[]
for i in range(0,101):
    residual=np.dot(X_test,WRR_2[i])-Y_test
    residual=np.square(residual)
    RMSE_2.append(np.sqrt(np.sum(residual)/42))
    
RMSE_3=[]
for i in range(0,101):
    residual=np.dot(X_test_3,WRR_3[i])-Y_test
    residual=np.square(residual)
    RMSE_3.append(np.sqrt(np.sum(residual)/42))



lamda_list=[]
for i in range(0,101):
    lamda_list.append(i)

fig = plt.figure(figsize=(9, 6))
ax1 = fig.add_subplot(111)

ax1.plot(lamda_list,RMSE,label='p=1')
ax1.plot(lamda_list,RMSE_2,label='p=2')
ax1.plot(lamda_list,RMSE_3,label='p=3')


ax1.set_xlabel("λ")
ax1.set_ylabel("RMSE")
lgd = ax1.legend(loc=9, bbox_to_anchor=(0.08,1))

plt.show()