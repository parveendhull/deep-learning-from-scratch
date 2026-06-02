import numpy as np
X=np.array([0.5,0.1,-0.2])
# activation functions
def ReLU(z):
    return np.maximum(0,z)
def sigmoid(z):
    return 1/(1+np.exp(-z))

# initialize weights and biases for hidden layer and output layer
w1=np.random.randn(3,4)
b1=np.zeros((1,4))
w2=np.random.randn(4,1)
b2=np.zeros((1,1))

def for_prop(X, w1, b1, w2, b2):
    z1=np.dot(X,w1)+b1
    a1=ReLU(z1)
    z2=np.dot(a1,w2)+b2
    a2=sigmoid(z2)
    cache=(z1,a1,z2,a2)
    return a1,a2,cache

hidden_output,final_output,cache=for_prop(X, w1, b1, w2, b2)
print('hidden_output',hidden_output)
print('final_output',final_output)
print('cache',cache)

