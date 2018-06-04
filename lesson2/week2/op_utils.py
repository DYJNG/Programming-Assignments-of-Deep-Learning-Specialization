import numpy as np
import h5py
import matplotlib.pyplot as plt
import h5py
import scipy.io
import sklearn
import sklearn.datasets

# 初始化参数
def initialize_params(layer_dims):
    params = {}
    
    L = len(layer_dims)
    
    for i in range(1, L):
        # random initialization
        # params["W" + str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * 0.3
        # Xavier initiliazation
        params["W" + str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * np.sqrt(2 / layer_dims[i-1])
        params["b" + str(i)] = np.zeros((layer_dims[i], 1))
        
        assert(params["W" + str(i)].shape == (layer_dims[i], layer_dims[i-1]))
        assert(params["b" + str(i)].shape == (layer_dims[i], 1))
    
    return params


# 正向传播
def forward_prop(X, params):
    L = len(params) // 2
    caches = {}
    caches["A0"] = X
    for i in range(1, L):
        caches["Z" + str(i)] = np.dot(params["W" + str(i)], caches["A" + str(i-1)]) + params["b" + str(i)]
        caches["A" + str(i)] = np.maximum(0, caches["Z" + str(i)])
    
    caches["Z" + str(L)] = np.dot(params["W" + str(L)], caches["A" + str(L-1)]) + params["b" + str(L)]
    caches["A" + str(L)] = 1 / (1 + np.exp(-caches["Z" + str(L)]))
    
    AL = caches["A" + str(L)]
    assert(AL.shape == (1, X.shape[1]))
    
    return AL, caches


# 代价函数
def cost_fun(AL, Y):
    m = Y.shape[1]
    cost = (-1/m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log(1-AL)))
    cost = np.squeeze(cost)
    
    return cost


# 反向传播
def backward_prop(params, caches, Y, lambd):
    grads = {}
    L = len(params) // 2
    m = Y.shape[1]
    
    grads["dZ" + str(L)] = caches["A" + str(L)] - Y
    
    for i in range(L):
        grads["dW" + str(L-i)] = (1/m) * np.dot(grads["dZ" + str(L-i)], caches["A" + str(L-i-1)].T) + (lambd/m) * params["W" + str(L-i)]
        grads["db" + str(L-i)] = (1/m) * np.sum(grads["dZ" + str(L-i)], axis=1, keepdims=True)
        if (i < L-1):
            grads["dA" + str(L-i-1)] = np.dot(params["W" + str(L-i)].T, grads["dZ" + str(L-i)])
            grads["dZ" + str(L-i-1)] = grads["dA" + str(L-i-1)] * (caches["Z" + str(L-i-1)] > 0)
        
    return grads


def load_cat_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def load_dataset():
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2) #300 #0.2 
    # Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    
    return train_X, train_Y

def load_2D_dataset():
    data = scipy.io.loadmat('datasets/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral);
    
    return train_X, train_Y, test_X, test_Y

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()
    
def predict_dec(parameters, X):
    """
    Used for plotting decision boundary.
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (m, K)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Predict using forward propagation and a classification threshold of 0.5
    a3, cache = forward_prop(X, parameters)
    predictions = (a3 > 0.5)
    return predictions

def predict(X, Y, params):
    AL, _ = forward_prop(X, params)
    pred = (AL > 0.5)
    
    acc = np.mean(pred == Y)
    print("Accuracy is %f" % acc)
    
    return pred