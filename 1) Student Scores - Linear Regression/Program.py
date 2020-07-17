#SLOPPY VERSION 1. G1, G2 ARE NOT CONSIDERED AS ESSENTIAL FEATURES. ACCURACY SEEMS LOW.

import pandas as pd
import numpy as np

print("Long time no code")
# Extracting data from CSV File:
# Feature Numbers considered fairly important (top 10): 14,15,16,17,18,22,24,25,29,30
# Feature Numbers of Grades G1,G2 and G3: 31,32,33 ---- These are outputs.
cols=[13,14,15,16,17,21,23,24,28,29]
df_mat = pd.read_csv('student-dataset/student-mat.csv',) #395 entries
df_por = pd.read_csv('student-dataset/student-por.csv') #649 entries
#print(df_mat)
#print(df_por)

#Storing number of data points available, setting X and Y.
m_math= df_mat.shape[0]
x_math=df_mat[df_mat.columns[cols]]
y_math=df_mat[['G3']].to_numpy()
# print(x_math.iloc[0])
# print(x_math.iloc[0,1])
for col in [2,3,4,5]:
    for count in range(0,m_math):
        if x_math.iloc[count,col] == 'yes':
            x_math.iloc[count,col] = 1
        else:
            x_math.iloc[count,col] = 0
x_math=x_math.to_numpy()
# x_math=x_math[:,[1,2,3,4]]
n=len(x_math[0])
# print(x_math)
# print(y_math)

# m_por=df_por.shape[0]
# x_por=df_por[df_por.columns[cols]]
# y_por=df_por[['G3']]

#Setting some variables.
epoch=100 #No of times you gotta train the data
alpha=0.005 #Learning rates
theta= np.zeros((n,1))

# print(x_math[0])

#-------------------------FEW DIMENSIONS LISTED OUT HERE---------------------
#x_math: m_math x n   x_math[j]= 1 x 10 here
#theta: n x 1 n= no of features =10
#hyp = m_math x 1
#y_math= m_math x 1 = y_math
# print(y_math)
for i in range(0,epoch):
    err=np.zeros((n,1))
    for j in range (0,m_math-100):
        hyp= (x_math[j]).dot(theta) #COMPUTES HYPOTHESIS VALUE FOR EACH ENTRY.
        # print(hyp.shape)
        # print(err)
        # # print((hyp-y_math[j])*x_math[j].transpose()[0])
        # print((hyp-y_math[j])*x_math[j].transpose())
        # print(((hyp-y_math[j])*x_math[j].transpose()).shape)
        # print(err.shape)
        err= np.add(err,((hyp-y_math[j])*x_math[j].transpose()).reshape(n,1))
        # print(err)
        # err = err + (hyp-y_math[j])*x_math[j].transpose()
        # print((hyp-y_math[j]).shape)
        # print(x_math[j].transpose().shape)
        # print(f"This is entry {j}")
        # print("Error term")
        # print(err)

    print(f"Epoch {i+1}------------")    
    theta = theta - (alpha/m_math)*(err.transpose().reshape(n,1))
    print(theta)

# for i in range(0,10):
#     print(theta[i][0])
# print("This is theta")
# print(theta.s)

y_output= np.zeros((100,1))
# print(y_output)
for i in range (m_math-100,m_math):
    # print(y_output[j])
    # print(x_math[i])
    # print((x_math[i]).dot(theta))
    y_output[i-m_math+100]= (x_math[i]).dot(theta)
# print("OUTPUT Y ")
# print(y_output)

# print("FINAL ERROR")
# print(y_math[100][0])
final_err= (y_output-y_math[m_math-100:,:])
# print(y_output.shape)
# print(y_math[m_math-100:,:].shape)
# print(y_math.shape)
print("\n COMPARISON OF INPUT, OUTPUT-------------------------\n")
print(np.concatenate((y_output,y_math[m_math-100:,:],final_err),axis=1))
print("\n AVERAGE ERROR ABSOLUTE-------------------------\n")
print(np.average(np.absolute(final_err)))
# print(np.average(np.absolute(final_err/y_math[m_math-100:,:])))