#Before the reader begins to read the code, I would like to tell that this is a re implemented version of the Neural
# Networks code found in Nielsen's Chapter 1 based on my understanding. I have re ordered few of the functions as I felt
# it would be more logical in that order to understand the code more thoroughly.
# I have also commented out a few snippets which I believe are essential for any reader to debug and understand the working of the code.
import random
import numpy as np

class NNetwork(object):
	def __init__(self,sizes):
		self.no_of_layers= len(sizes)
		self.sizes=sizes
		self.biases=[np.random.randn(y,1) for y in sizes[1:]]
		self.weights=[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
		# for x in self.weights:
		# 	print(np.shape(x))
		# for w,b in zip(self.weights,self.biases):
		# 	print(np.shape(b))
		# 	print(np.shape(w))
	def forwardprop(self,a):
		"""This function basically performs the Forward propagation part of the Neural Network, and returns the output layer."""
		for w,b in zip(self.weights,self.biases):
			a=sigmoid(np.dot(w,a)+b)
		return a

	def SGD(self, training_data, epochs, mini_batch_size,eta,test_data=None):
		"""NOTE: This is the implementation of Stochastic Gradient Descent Algorithm"""
		if test_data:
			n_test = len(test_data)
		n = len (training_data)
		#No of times the learning is repeated.
		for j in range(epochs):
			random.shuffle(training_data)
			mini_batches=[training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
			for mini_batch in mini_batches:
				self.update(mini_batch,eta)
			if test_data:
				print("Epoch {0}:{1}/{2}".format(j,self.evaluate(test_data),n_test))
			else:
				print("Epoch {0} complete.".format(j))

	def evaluate(self,test_data):
		"""Basically this function returns the number of successful predictions for a given test data set"""
		test_results= [(np.argmax(self.forwardprop(x)),y) for (x,y) in test_data]
		return sum(int(x==y) for (x,y) in test_results)

	def backprop(self,x,y):
		"""This function returns nabla_b, nabla_w, which have stored values of gradient of the cost function,
		with respect to the individual biases and weights."""
		nabla_b= [np.zeros(b.shape) for b in self.biases]
		nabla_w= [np.zeros(w.shape) for w in self.weights]
		activation=x
		activations=[x]#Stores all activations recorded
		zvectors=[]#This will store all the z vectors, layer by layer
		for b,w in zip(self.biases,self.weights):
			# print(w.shape)
			# print(activation.shape)
			# print(b.shape)
			# print("\n")
			z=np.dot(w,activation)+b
			zvectors.append(z)
			activation=sigmoid(z) #This will be the activation for the next iteration
			activations.append(activation)
		#backward pass
		delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zvectors[-1])
		nabla_b[-1]=delta
		nabla_w[-1]=np.dot(delta, activations[-2].transpose())
		for l in range(2,self.no_of_layers):
			z=zvectors[-l]
			sp=sigmoid_prime(z)
			delta=np.dot(self.weights[-l+1].transpose(),delta)*sp
			nabla_b[-l]=delta
			nabla_w[-l]=np.dot(delta,activations[-l-1].transpose())

		return (nabla_b,nabla_w)

	def update(self,mini_batch,eta):
		"""This function basically adjusts the weights and biases by running through training examples in the mini batch."""
		nabla_b=[np.zeros(b.shape) for b in self.biases]
		nabla_w=[np.zeros(w.shape) for w in self.weights]
		for x,y in mini_batch:
			delta_nabla_b, delta_nabla_w=self.backprop(x,y)
			nabla_b= [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
			nabla_w= [nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
		self.weights=[w-(eta/len(mini_batch))*nw for 
		w, nw in zip (self.weights, nabla_w)]
		self.biases=[b-(eta/len(mini_batch))*nb for
		b, nb in zip (self.biases, nabla_b)]


	def cost_derivative(self,output_results,y):
		"""This function may be provided in Nielsen to avoid unnecessary clutter. Also, the error function is 1/2*(y-x)^2. 
		So d(Error function)/d(y)=(y-x), which is what is returned in this function."""
		return (output_results-y)


# UTILITY FUNCTIONS:
def sigmoid(z):
    """Sigmoid Function"""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivatie of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

# NNetwork([3,5,2])
