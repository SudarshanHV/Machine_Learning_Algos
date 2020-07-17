#CHANGES FROM PREVIOUS TIME:
#NARROWING DOWN SELECTING FEATURES TO G1,G2, health, absences and failures
#CLEANER DATA EXTRACTION
#OVERALL BETTER AESTHETICS.
#AVERAGE ERROR VARIES AROUND 0.8 to 1.6 grade points around the score on absolute average measured.

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data= pd.read_csv("student-dataset/student-mat.csv")
# data= pd.read_csv("student-dataset/student-mat.csv",sep=";") If delimiter is ; instead of ,
#print(data.head())
data=data[["G1","G2","G3","studytime","failures","absences"]]
#print(data.head())

predict="G3" #Label Or the thing you got to predict

x=np.array(data.drop([predict],1))
y=np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size = 0.1) 

n=len(x_train[0])      #No of Features

m_math=data.shape[0]    #Total no of data points
m= len(x_train)         #No of points used for training
epoch=100
alpha=0.005
theta=np.zeros((n,1))

for i in range(0,epoch):
    err=np.zeros((n,1))
    for j in range (0,m):
        hyp= (x_train[j]).dot(theta) #COMPUTES HYPOTHESIS VALUE FOR EACH ENTRY.
        err= np.add(err,((hyp-y_train[j])*x_train[j].transpose()).reshape(n,1)) #ERROR TERM CALCULATION FOR HYPOTHESIS    
    theta = theta - (alpha/m*(err.transpose().reshape(n,1)))
    print(f"Epoch {i+1} DONE------------")

y_output= np.zeros((m_math-m,1))
final_err= np.zeros((m_math-m,1))
i=0
for (item1,item2) in zip(x_test,y_test): 
    y_output[i]= (item1).dot(theta)
    final_err[i]= y_output[i]-item2
    i=i+1
i=0

print("\nSIDE BY SIDE COMPARISON OF DATA\n")
for (item1,item2) in zip(x_test,y_test):
    print(y_output[i], item1, item2)
    i=i+1

print("\n AVERAGE ABSOLUTE ERROR\n")
print(np.average(np.absolute(final_err)))
