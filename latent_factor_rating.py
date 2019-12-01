#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 17:18:53 2019

@author: jenny
"""
import json
import gzip
from collections import defaultdict
import  random
from random import choice
import  numpy
import scipy
import scipy.optimize
from sklearn.model_selection import train_test_split

def prediction(user, item):
    if user not in userBiases: 
        userBias=0
    else:
        userBias=userBiases[user]
    if item not in itemBiases:
        itemBias=0
    else:
        itemBias=itemBiases[item] 
    return alpha + userBias + itemBias

def unpack(theta):
    global alpha
    global userBiases
    global itemBiases
    alpha = theta[0]
    userBiases = dict(zip(users, theta[1:nUsers+1]))
    itemBiases = dict(zip(items, theta[1+nUsers:]))

def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)

def cost(theta, labels, lamb):
    unpack(theta)
    predictions = [prediction(d['user_id'], d['item_id']) for d in dataset]
    cost = MSE(predictions, labels)

    predictions_valid = [prediction(d['user_id'], d['item_id']) for d in dataset_valid]
    cost_valid = MSE(predictions_valid, labels_valid)
    print("On train set, MSE = " + str(cost)+", On valid set, MSE = " + str(cost_valid))


    for u in userBiases:
        cost += lamb*userBiases[u]**2
    for i in itemBiases:
        cost += lamb*itemBiases[i]**2
    return cost

def derivative(theta, labels, lamb):
    unpack(theta)
    N = len(dataset)
    dalpha = 0
    dUserBiases = defaultdict(float)
    dItemBiases = defaultdict(float)
    for d in dataset:
        u,i = d['user_id'], d['item_id']
        pred = prediction(u, i)
        diff = pred - d['rating']
        dalpha += 2/N*diff
        dUserBiases[u] += 2/N*diff
        dItemBiases[i] += 2/N*diff
    for u in userBiases:
        dUserBiases[u] += 2*lamb*userBiases[u]
    for i in itemBiases:
        dItemBiases[i] += 2*lamb*itemBiases[i]
    dtheta = [dalpha] + [dUserBiases[u] for u in users] + [dItemBiases[i] for i in items]
    return numpy.array(dtheta)

# main program
# =============================================================================
# path = "train_interactions.csv.gz"
# f = gzip.open(path, 'rt', encoding="utf8")
# header = f.readline()
# header = header.strip().split(',')
# dataset = []
# validation=[]
# 
# count=0
# for line in f:
#     count += 1
#     if count <= 199900:
#         fields = line.strip().split(',')
#         d = dict(zip(header, fields))
#         d['rating'] = int(d['rating'])
#         dataset.append(d)
#     else:
#         fields = line.strip().split(',')
#         d = dict(zip(header, fields))
#         d['rating'] = int(d['rating'])
#         validation.append(d)
# =============================================================================
dataset=[]
null=None #fix small issue of json reading
with gzip.open("cloth_data.json.gz",'rt') as f:
    for line in f:
        data=json.loads(line)
        if data['rating']==None or len(data)!=15:  #If it doesn't have all attributes or rating is None, abort thus sample  
            continue
        if data['fit']=='fit':
            data['whetherFit']=1
        else:
            data['whetherFit']=0         
   
        data['age']=int(data['age'])
        data['rating']=int(data['rating'])
        data['weight']=int(data['weight'][:-3])
        height_tmp=data['height'].strip('\"').split('\'')
        data['height']=float(height_tmp[0])+0.1*float(height_tmp[1])
        dataset.append(data)

dataset,dataset_valid=train_test_split(dataset,test_size=1/5,random_state=1)
   
usersPerBook = defaultdict(set)
BooksPerUser = defaultdict(set)
for d in dataset:
    user,Book = d['user_id'], d['item_id']
    usersPerBook[Book].add(user)
    BooksPerUser[user].add(Book)

N = len(dataset)
nUsers = len(BooksPerUser)
nItems = len(usersPerBook)
users = list(BooksPerUser.keys())
items = list(usersPerBook.keys())

labels = [d['rating'] for d in dataset]
labels_valid = [d['rating'] for d in dataset_valid]
ratingMean = sum([d['rating'] for d in dataset]) / len(dataset)
alpha=ratingMean
userBiases = defaultdict(float)
itemBiases = defaultdict(float)

scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nItems),derivative, args = (labels, 0.00005))

#calculate the MSE of validation set
# =============================================================================
# predictions = open("predictions_Rating.txt", 'w')
# for l in open("pairs_Rating.txt"):
#     if l.startswith("userID"):
#         #header
#         predictions.write(l)
#         continue
# 
#     u,b = l.strip().split('-')
#     if u in userBiases:
#         user_bias = userBiases[u]
#     else:
#         user_bias = 0
#     if b in itemBiases:
#         book_bias = itemBiases[b]
#     else:
#         book_bias = 0
#     rate_prediction = alpha + user_bias + book_bias
# 
#     predictions.write(u + '-' + b + ',' + str(rate_prediction) + '\n')
# 
# 
# predictions.close()
# =============================================================================

# =============================================================================
# y=[int(d['rating']) for d in dataset]
# N3=len(dataset)
#         
# X_train=dataset[:(N3//11)*10]
# X_valid=dataset[(N3//11)*10:]
# y_train=y[:(N3//11)*10]
# y_valid=y[(N3//11)*10:]
# predict=[]
# rows=len(X_train)    
# cols=len(X_train[0])
# for l in X_train:
#     u=l['user_id']
#     i=l['item_id']
#     if u in userBiases:
#         user_bias = userBiases[u]
#     else:
#         user_bias = 0
#     if i in itemBiases:
#         item_bias = itemBiases[i]
#     else:
#         item_bias = 0
#     row_result=alpha + user_bias + item_bias
#     predict.append(row_result)
#             
# differences=[(x-y)**2 for (x,y) in zip(predict,y_train)]
# MSE=sum(differences)/len(differences)
# =============================================================================

