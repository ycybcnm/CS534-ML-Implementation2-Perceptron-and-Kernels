# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 06:31:24 2020

@author: cyy
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

def loadData( fname_x  = "pa2_train_X.csv", fname_y = "pa2_train_Y.csv"):
    # load data from file
    dataX = []
    dataY = []
    with open(fname_x) as csvfile:
        trainData_reader  = csv.reader(csvfile, delimiter=" ")
        line_count = 0
        for row in trainData_reader:
            if line_count != 0:
                line = ",".join(row).split(",")
                dataX.append(list(map(float, line)))
            line_count += 1
            
    with open(fname_y) as csvfile:
        trainData_reader  = csv.reader(csvfile, delimiter=" ")
        line_count = 0
        for row in trainData_reader:
            if line_count != 0:
                line = ",".join(row).split(",")
                dataY.append(list(map(float, line)))
            line_count += 1
            
    return dataX, dataY

def dataNormalizaton(data_x, data_y, normPos = [1,5,6]):
    train_x =  np.array(data_x)
    
    for i in normPos:
        data_min = train_x[:,i].min(0)
        data_max = train_x[:,i].max(0)
        train_x[:,i] = (train_x[:,i] - data_min) / (data_max - data_min)
        
    train_y = np.array(data_y).reshape(train_x.shape[0],)
    
    
    return train_x, train_y

def training(train_x, train_y, rate, convergence_norm = 10 ** -2, reg_para = 10 ** -1):
     weights = np.zeros(train_x.shape[1])
     weight_recorder = []
     #norm_recorder = []
     
     #while(1):
     for i in range(10000):
         gradient = gradient_helper(weights, train_x, train_y)
         weights = weights + (rate * gradient)
         weights -= rate * reg_para * weights
         #for j in range(weights.shape[0]):
             #weights[j] =  np.sign(weights[j]) * max(abs(weights[j])- rate * reg_para , 0)
         norm_gradient = np.linalg.norm(gradient)
         weight_recorder.append(weights)
         #norm_recorder.append(norm_gradient)
         #print(gradient)
         #print(weight)
         #print(np.linalg.norm(gradient))
         #print(norm_gradient)
         if norm_gradient <= convergence_norm or np.isinf(norm_gradient):
             return weights, weight_recorder
     return weights, weight_recorder

def gradient_helper(w, x, y):   
    gradient = 0
    N = x.shape[0]

    for i in range(N):
        error = y[i] - sigmoid_func(np.dot(x[i],w))
        gradient +=  (error) * x[i]
    return gradient/N

def sigmoid_func(x):   
    """
    Just sigmoid function
    """
    return 1 / (1 + np.exp(-x))

def predict(test_x, weights, boundary = 0.5):
    ret = np.array(sigmoid_func(np.dot(test_x, weights)))
    prediction = np.where(ret >= boundary, 1, 0)
       
    return prediction, ret

if __name__ == "__main__":
    #learning rate and convergence
    learning_rate = pow(10, -1)
    convergence_norm = pow(10, -1)
    reg_para = 10 ** -0
    success_rates = []
                                                       
    # Load data from csv file
    train_x, train_y= loadData()
    # Load data from csv file
    test_x, test_y = loadData("pa2_dev_X.csv", "pa2_dev_Y.csv")
    # Nomalization training Data
    nl_train_data, y = dataNormalizaton(train_x, train_y)
    # Nomalization testing Data
    nl_test_data, yt = dataNormalizaton(test_x, test_y)
    
    
    wr,weight_recorder = training(nl_train_data, y, learning_rate, convergence_norm, reg_para )
    for idx, val in enumerate(weight_recorder):
        prediction, ret = predict(nl_test_data, val)
        success_rates.append(1 - np.count_nonzero(yt - prediction)/prediction.shape[0])
        
    plt.plot(success_rates)
    plt.show()
        
    prediction, ret = predict(nl_test_data, wr)
    success_rate = 1 - np.count_nonzero(yt - prediction)/prediction.shape[0]
    print("the weight:", wr )
    print(success_rate)
    
    #success_rates_col= []
    #wrs = []
    #for i in range(6):
    #    success_rates = []
    #    reg_para = 10 ** -i
    #    wr,weight_recorder = training(nl_train_data, y, learning_rate, convergence_norm, reg_para )
    #    for idx, val in enumerate(weight_recorder):
    #        prediction, ret = predict(nl_train_data, val)
    #        success_rates.append(1 - np.count_nonzero(y - prediction)/prediction.shape[0])
    #    success_rates_col.append(success_rates)
    #    wrs.append(wr)
    
    #x = list(range(len(success_rates_col[0])))
    #y = success_rates_col
    #plt.xlabel("iteration")
    #plt.ylabel("accuracy")
    #for i in range(len(y)):
    #    plt.plot(x,[pt for pt in y[i]],label = '10^-%s'%i)
    #plt.legend()
    #plt.show()
