# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 19:14:32 2017

@author: KRapes
"""

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, fbeta_score
import pickle
import numpy as np
import os
from time import time
from  random import random, randint
import warnings
from dataset_lib import normalize, ascii_to_map
from Tensorflow_network import findings_search, elect_value
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def file_names(folder):
    files = []
    for file in os.listdir(folder):
        if file.startswith("labeled_images_"):
            files.append(os.path.join(folder, file.split('.')[0]))
    return files
        
def ravel(dataset):
    #x_train = [x[1].shape[0] * x[1].shape[1] for x in dataset]
    x_train = []
    start = time()
    #print("START RAVEL")
    for image in dataset:
        '''
        print(image)
        if len(image) > 1:
            print("Enter if")
            image = image[1]
        print(image)
        '''

        image = image.reshape(np.prod([shape for shape in image.shape]))

        '''
        append_length = max_img - image.shape[0]
        fill = np.full([1,append_length], -1, dtype=None)
        image = np.append(image,fill)

        rang = abs(max(image) - min(image))
        image = image/rang
        image = image - min(image)
        #print("max: {} min: {} average: {}".format(max(image), min(image), np.mean(image)))
        '''
        x_train.append(image)
    end = time()
    #print("Elapsed Time: {}".format(round(end-start, 2)))
    return x_train

def find_max(dataset, testset):
    maxes = []
    maxes.append( max([x[0].shape[0] * x[0].shape[1] for x in dataset]))
    maxes.append( max([x[0].shape[0] * x[0].shape[1] for x in testset]))
    return max(maxes)
def predict_score(x_test, y_test, clf):
    start = time()
    #print("START PREDICT")
    predictions_test = clf.predict(x_test)
    end = time()
    #print("Elapsed Time: {}".format(round(end-start, 2)))
    acc = accuracy_score(y_test,predictions_test)
    fbeta = fbeta_score(y_test,predictions_test,0.5)
    print("Accurcy: {}   fBeta: {} ".format(acc,fbeta))
    return acc, fbeta


def append_results(dataset, results):
    for index in range(len(dataset)):
        entry = dataset[index]
        output = results[index]
        entry.append(output)
    return dataset

def append_probabilities(dataset, ravel, clf):
    pre_prob = 1
    prob = clf.predict_proba(ravel[0])[0][0]
    for index in range(len(dataset)):
        entry =  dataset[index]
        if index < len(dataset) - 1:
            next_prob = clf.predict_proba(ravel[index + 1])[0][0]
        else:
            next_prob = 1
        entry.append(pre_prob)
        entry.append(prob)
        entry.append(next_prob)
        pre_prob = prob
        prob = next_prob
    
    return dataset

def append_probabilities_train(dataset, y):
    pre_prob = 1
    prob = y[0]
    for index in range(len(dataset)):
        entry =  dataset[index]
        if index < len(dataset) - 1:
            next_prob = y[index + 1]
        else:
            next_prob = 1
        entry.append(pre_prob)
        entry.append(prob)
        entry.append(next_prob)
        pre_prob = prob
        prob = next_prob
    
    return dataset

def seperate_data(dataset, percent, clf):
    def process(dataset, clf):
        x = ravel([n[0] for n in dataset])
        if clf != None:
            p_train = clf.predict(x)
            x = [[p, n[1], n[2], n[3], n[4]] for n, p in zip(dataset, p_train)]
        y = [0 if ascii_to_map(n[5]) >= 0 else 1 for n in dataset]
        return x, y

    
    data_length = len(dataset)
    testset = []
    while len(testset) < data_length*percent:
        testset.append(dataset.pop(int(random() * len(dataset))))
    if len(testset) > 0:
        x_test, y_test = process(testset, clf)
    else:
        x_test, y_test = [], []
    if len(dataset) > 0:
        x_train, y_train = process(dataset, clf)
    else:
        x_train, y_train = [], []
    

    
    return x_train, y_train, x_test, y_test

def load_data(folder_name, percent, clf=None):
    files = file_names(folder_name)
    for file in files:
        dataset = load_obj(file)
        yield seperate_data(dataset, percent, clf)

def fit_tree(clf1, clf2):
    x = []
    y = []
    for x_train, y_train, x_test, y_test in load_data('./labeled_images', 0.01):
        x += x_train
        y += y_train
    print("Training clf1")
    clf1.fit(x, y)
    #end = time()
    #print("Elapsed Time: {}".format(round(end-start, 2)))
    #print("File: {}".format(file))
    #predict_score(x_test, y_test, clf)
    x = []
    y = []
    for x_train, y_train, x_test, y_test in load_data('./labeled_images', 0.01, clf1): 
        x += x_train
        y += y_train
    #print(x[0])
    #print("Length x: {}    Length y: {}".format(len(x), len(y)))
    print("Training clf2")
    clf2.fit(x,y)
    print("Training complete")
    '''
    features_data = [[ x[2], x[3], x[4], x[5]] for x in dataset]
    x_train = append_probabilities_train(features_data, y_train)
    features_test = [[ x[2], x[3], x[4], x[5]] for x in testset]
    x_test = append_probabilities_train(features_test, y_test)
    #print(x_train[0])
    clf2 = RandomForestClassifier(warm_start=True) 
    start = time()
    #print("START FIT")
    clf2.fit(x_train, y_train)
    end = time()
    #print("Elapsed Time: {}".format(round(end-start, 2)))
    predict(x_test, y_test, clf2)
    '''
        
        
        
        
            
            
        
        
        
        
    save_obj(clf1, "AOI_1")
    save_obj(clf2, "AOI_2")
    return clf1, clf2
        #save_obj(clf2, "AOI_RF-2")

def select_classifier():
    prog_start = time()
    classifiers = [SGDClassifier(), RandomForestClassifier(),
                    DecisionTreeClassifier(),
                    AdaBoostClassifier()]
    
    max_acc = 0
    clf1 = DecisionTreeClassifier()
    for clf2 in classifiers:
        print("")
        print("Training with {} classifier".format(str(clf2)[0:6]))
        start = time()
        
    
        
        clf1, clf2 = fit_tree(clf1, clf2)
        train_time = time() - start
        start = time()
        
        
        count, acc, fb = 0, 0, 0
        
        for x_train, y_train, x_test, y_test in load_data('.', .99, clf1):
            a, f = predict_score(x_test, y_test, clf2)
            acc += a
            fb += f
            count += 1
        accuarcy = acc/count
        if accuarcy >= max_acc:
            best_case = [clf1, clf2]
        print("Accuarcy: {:.2}   fbeta: {:.2}    "
              "Time to train: {:.2}    Time to predict: {:.2}".format(accuarcy,
                                                                    fb/count,
                                                                    train_time/60,
                                                                    (time() - start)/60))
    clf1, clf2 = fit_tree(best_case[0], best_case[1])    
    end = time()
    print("Total training time {:.2}".format((end-prog_start)/60))
    return clf1, clf2

def hyperparam_selection():
    def selecting_values(name, findings, valuerange, keep_prob):
        if random() < keep_prob:
            return elect_value(name, findings, valuerange)
        else:
            return randint(valuerange[0],valuerange[1])
    
    start = time()
    clf = DecisionTreeClassifier()
    max_depth = [3, 100]  # 1- 50
    criterions = ['gini', 'entropy'] # gini or “entropy”
    splitters = ['best', 'random'] # best or random
    min_samples_split = [2, 20] # 2 - 10
    min_samples_leaf = [1, 20] # 1 - 5
    
    N = 100
    keep_prob = 0.8
    findings = {}
    for n in range(N):
        i = 0
        depth = selecting_values('depth', findings, max_depth, keep_prob)
        split = selecting_values('split', findings, min_samples_split, keep_prob)
        leaf = selecting_values('leaf', findings, min_samples_leaf, keep_prob)
        for criterion in criterions:
            for splitter in splitters:
                i += 1
                print("Fitting {} of {}:  {}%".format(n,N, (i/4)*100))
                
                clf = DecisionTreeClassifier(criterion=criterion,
                                             splitter=splitter,
                                             max_depth=depth,
                                             min_samples_split=split,
                                             min_samples_leaf=leaf)
                clf = fit_tree(clf)
                acc, fb, count = 0, 0, 0
                for x_train, y_train, x_test, y_test in load_data('.', .99):
                    a, f = predict_score(x_test, y_test, clf)
                    acc += a
                    fb += f
                    count += 1
                accuarcy = acc/count
                findings[round(accuarcy,2)] = {'depth': depth, 
                                      'split': split,
                                      'leaf': leaf,
                                      'criterion': criterion,
                                      'splitter': splitter}
                #print("Accuarcy: {:.2}   Time: {:.2}".format(accuarcy, (time()-start)/60))
    
    depth = findings[max(findings)]['depth']
    split = findings[max(findings)]['split']
    leaf = findings[max(findings)]['leaf']
    splitter = findings[max(findings)]['splitter']
    criterion = findings[max(findings)]['criterion']
    clf = DecisionTreeClassifier(criterion=criterion,
                                             splitter=splitter,
                                             max_depth=depth,
                                             min_samples_split=split,
                                             min_samples_leaf=leaf)
    clf = fit_tree(clf)
    acc, fb, count = 0, 0, 0
    for x_train, y_train, x_test, y_test in load_data('.', .99, clf1):
        a, f = predict_score(x_test, y_test, clf2)
        acc += a
        fb += f
        count += 1
    accuarcy = acc/count
    findings[round(accuarcy,2)] = {'depth': depth, 
                          'split': split,
                          'leaf': leaf,
                          'criterion': criterion,
                          'splitter': splitter}
    print("Accuarcy: {:.2}   Time: {:.2}".format(accuarcy, (time()-start)/60))


#select_classifier()