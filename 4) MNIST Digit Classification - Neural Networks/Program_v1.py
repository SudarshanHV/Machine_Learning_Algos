#Some Notes:1) Hidden layer size:15, alpha = 3, Batchsize=10, 30 epochs. Accuracy: 88 to 90 percent
#2) Hidden layer size: 32, Batch size:10 , alpha=3, 30 epochs. Accuracy: 90 to 92 percent 
#3) Hidden layer size:15, alpha = 1, Batchsize=1000, 10 epochs. Accuracy: 30 to 35 percent
#4) Regularization has not been added to the model.
#5) Model 2, run with a training-validation split of 0.01, and test set of 10000 (Before, it was 51000 training, 9000 cross validation, 10000 test set)
# Accuracy: 88 to 92 percent.
#6) Model no 2, run with alpha = 4.5 10 epochs,Line 142 to change alpha, line 48 to change train test split
# Accuracy:83 to 85 percent.
import idx2numpy
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

#------------------------------STEP 1: UNROLLING AND STORING THE DATA--------------------------------------

file1 = './data/train-images.idx3-ubyte'
file2 = './data/t10k-images.idx3-ubyte'
file3 = './data/train-labels.idx1-ubyte'
file4 = './data/t10k-labels.idx1-ubyte'

arr1 = idx2numpy.convert_from_file(file1)
arr2= idx2numpy.convert_from_file(file2)
arr3 = idx2numpy.convert_from_file(file3)
arr4 = idx2numpy.convert_from_file(file4)

def vectorized_result(j):
    e=np.zeros((10,1))
    e[j]=1.0
    return e

print("hello bois")

training_set = [np.reshape(x, (784,1)) for x in arr1]
training_set= np.array(training_set)

training_input = [vectorized_result(y) for y in arr3]
training_input=np.array(training_input)

x_test = [np.reshape(x, (784,1)) for x in arr2]
x_test = np.array(x_test)
y_test = np.array(arr4).reshape(10000,1)

# print(arr3[0])
# print(training_input[0])
# print(training_set.shape)
# print(training_set[0])

x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(training_set, training_input , test_size = 0.15)

# print(x_train.shape)
# print(y_val.shape)
# print(x_test.shape)
# print(y_test.shape)

#---------------------STEP 2: INITIALIZING THE WEIGHTS AND BIAS VECTORS--------------------------

layer_details = [784, 32, 10]
L= len(layer_details) #Number of layers in the Neural Network

def initialize_weights_bias(layer_details):
    biases = [np.random.randn(x,1) for x in layer_details[1:]] #NO BIAS FOR INPUT LAYER
    weights = [np.random.randn(x,y) for x,y in zip(layer_details[1:], layer_details[:-1])] 
    # l+1 x l neurons for weight matrix from
    # layer l to layer l+1.
    weights=np.array(weights)
    biases=np.array(biases)
    return (biases, weights)

# PLEASE DO NOTE THE SYNTAX FOR np.zeros(). You need double brackets
b , w = initialize_weights_bias(layer_details) #These variables store the weights and biases calculated.
delta_b = [np.zeros((x,1)) for x in layer_details[1:]]
delta_b = np.array(delta_b)
delta_w = [np.zeros((x,y)) for x,y in zip(layer_details[1:], layer_details[:-1])]
delta_w = np.array(delta_w)

# print(w[0].shape)
# print(b[0].shape)

#--------------------------------STEP 3: Defining sigmoid function---------------------
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
    #Returns sigmoid function value

def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))
    #Return derivative of the sigmoid function

# NOTE: The above functions only work when the inputs are numpy arrays. It makes for more convenient operations.



#-----------------------------STEP 4: Initializing z's and activations--------------------------

# print(z)
# print(a)
z = [np.zeros((x,1)) for x in layer_details[0:]] # z values for each layer
a = [sigmoid(x) for x in z]  #Activation values for each layer
#----------------------------STEP 5: Creating the Feed Forward Function--------------------------- 
def feed_forward(x):
    #We need to use a global variable. Basically the weights and bias vectors will be the global one.
    z[0] = x
    a[0] = sigmoid(z[0])
    #Given a single training example, to compute the outputs.
    #print(z[0])
    for i in range(1,L):
        z[i] = w[i-1].dot(a[i-1])+ b[i-1]
        a[i] = sigmoid(z[i])
    return a[L-1]

#---------------------------STEP 6: Back propagation function to set delta w and delta b-------------------------
#We basically run back propagation and calculate gradients to use for 
error = [np.zeros((x,1)) for x in layer_details[0:]]
error = np.array(error)
def back_propagation(y):
    #This function computes the gradient of the cost function with respect to the weights and biases, storing it in the global variable delta_b and delta_w.
    global delta_b
    global delta_w #Tells the function to use the globally defined variables.
    error[L-1] = (a[L-1]-y)*sigmoid_derivative(z[L-1])
    for j in range(L-2,0,-1):
        error[j]= (np.transpose(w[j]).dot(error[j+1]))*sigmoid_derivative(z[j])
    delta_b = error[1:]
    for i in range(0,L-1):
        delta_w[i]=error[i+1].dot(np.transpose(a[i]))
    # print("DELTA B inside the function")
    # print(delta_b[0])

# print("Z before feedforward------------------------")
# print(z[1])
# feed_forward(x_train[0])
# print("Z after feedforward------------------------")
# print(z[1])
# print(" DELTA_B Before Backpropagation -------------------------------------")
# print(delta_b[0])
# back_propagation(y_train[0])
# print(" DELTA_B After Backpropagation -------------------------------------")
# print(delta_b[0])

#---------------------------STEP 7: Gradient descent and calculating weights and biases---------------------------------------
epochs= 30
batch_size_SGD=10
no_of_batches = (x_train.size/784)/batch_size_SGD
alpha= 4.5

# Note: Keep batch size around 10 to 20. and alpha around 3 to 6. Net Learning rate will be around alpha/batch_size_SGD 

def error_printer():
    sum=0
    for i in range (0,10000):
        dummy = feed_forward(x_test[i])
        m = np.where(dummy== np.amax(dummy))
        # print(f"{dummy.size}")
        if(m[0][0] == y_test[i][0]):
            sum = sum+1
    print(f"{sum}/10000 predicted correctly")


# print(no_of_batches)
for i in range(0,epochs):
    print("\n")
    for j in range(0,int(no_of_batches)):
        total_error_w= np.zeros(delta_w.shape)
        total_error_b= np.zeros(delta_b.shape)
        # print(j*1000)
        for k in range (j*batch_size_SGD,j*batch_size_SGD+batch_size_SGD-1):
            dummy2 = feed_forward(x_train[k])
            back_propagation(y_train[k])
            total_error_b= total_error_b + delta_b
            total_error_w= total_error_w + delta_w
        # print(j*1000+999)
        w = w - (alpha/batch_size_SGD)*total_error_w
        # print(b[1])
        b = b - (alpha/batch_size_SGD)*total_error_b
        # print(f"BATCH {j+1} COMPLETED")
    print(f"\nEPOCH {i+1} COMPLETED-----------------------------------------------------------------------")
    error_printer()

 
