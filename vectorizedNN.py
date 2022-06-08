#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 15:06:41 2022

@author: nicolecastillo
"""
import numpy as np

#%%
# Neural networks with back propagation

# adds 1 column corresponding to weights of the bias neuron
def random_weights(x_i, connections):
    w_matrix =[]
    for i in range((x_i+1)*connections):
        w_matrix.append(np.random.randn())
    return(np.array(w_matrix).reshape(connections, (x_i+1)))

# tranfer functions - we're using sigmoid here
def forward_sigmoid(x):
    return(1/(1 + np.exp(-x)))
    
def backward_sigmoid(x):
    return x * (1 - x)

# Where A is the inputs (later in the loop, these are sigmoid tranferred activations per neuron)
# Where Z is the raw activation
def forward_propogation(a, weights, nn):
    activations = {}
    activations["A" + str(0)] = np.array([[a]])
    for layer in range(len(nn)-1):
        layer_idx = layer + 1
        prev_act = np.array([np.append([1], a)])
        prev_act = np.swapaxes(prev_act, 0, 1)
        Wi = weights["W" + str(layer_idx)]
        
        z = np.around(np.dot(Wi, prev_act), decimals = 5)
        a = np.around(forward_sigmoid(z), decimals = 5)
        activations["A" + str(layer_idx)] = a
        #activations["Z" + str(layer_idx)] = z
    return activations
        
# cost function j, regularized
# the predicted is going be fp["A"+str(len(nn)-1)]
def cost_function(expected, predicted):
    j = (np.dot(-expected, np.log(predicted).T) - np.dot(1 - expected, np.log(1 - predicted).T))
#    if len(expected) > 1:
#        j = np.sum(j.diagonal())
    return j

def regularized_cost(j, inputs, weights, lambda_r):
    j = np.array([np.sum(j)/len(inputs)])
    squared_sum = []
    for w in weights:
        squared_sum.append(np.sum(np.square(weights[str(w)])))
    s = np.sum(squared_sum) * (lambda_r/(2*len(inputs)))
    return j+s

# append j to vector in loop
# store all of the activations somewhere and all of the true classes too
def backward_propagation(expected, activations, weights, nn):
    gradients = {}
    deltas = {} 
    curr_deltas = activations["A" + str(len(nn)-1)] - expected
    deltas["dE" + str(len(weights))] = curr_deltas
    
    for layer in reversed(range(len(nn)-1)):
        current_layer = layer
        prev_delta = curr_deltas 
        
        curr_w = weights["W" + str(current_layer + 1)] #still includes bias
        curr_a = activations["A" + str(current_layer)]
        if current_layer != 0:
            curr_deltas = np.multiply(curr_w[:,1:].T.dot(prev_delta.T), np.multiply(curr_a, 1-curr_a).T)
            if curr_deltas.shape[0] > 1:
                curr_deltas = curr_deltas.diagonal()
            deltas["dE" + str(current_layer)] = np.array([curr_deltas])
        
        actsWbias = np.append(1, curr_a) #add 1 for bias
        gradients["dW" + str(current_layer)] = np.outer(prev_delta, actsWbias)
    return gradients, deltas

def update(weights, gradients, nn, alpha):
    for layer in range(len(nn)-1):
        weights["W" + str((layer+1))] -= alpha * gradients["dW" + str((layer))]        
    return weights
