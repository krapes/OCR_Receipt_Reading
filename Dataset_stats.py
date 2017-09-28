# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 19:34:54 2017

@author: KRapes
Statiticts
"""

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def file_names(folder):
    files = []
    for file in os.listdir(folder):
        if file.startswith("labeled_images_"):
            files.append(os.path.join(folder, file.split('.')[0]))
    return files

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

def distrubution(labels):
    c = Counter(labels)
    c = c.items()
    a_values = [n[0] for n in c]
    freq = [n[1] for n in c]
    print(a_values)
    return a_values, freq
        
files = file_names('labeled_images')
labels = []
labels_nospace = []
bnw = []
for file in files:
    data = load_obj(file)
    for entry in data:
        if entry[5] == 32:
            bnw.append('Non-character')
        labels.append(entry[5])
        if entry[5] != 32:
            labels_nospace.append(entry[5])
            bnw.append('Character')
x, freq = distrubution(labels)
x_nospace, freq_nospace = distrubution(labels_nospace)
x_bnw, freq_bnw = distrubution(bnw)
print(x_bnw, freq_bnw)

plt.figure('Figure 0')
plt.bar(x, freq, align='center')
plt.title("Label Distribution Including ASCii 32 (Space)")
plt.xlabel("ASCII Value")
plt.ylabel("Frequency")

plt.figure('Figure 1')
plt.bar(x_nospace, freq_nospace, align='center')
plt.title("Label Distribution Excluding ASCii 32 (Space)")
plt.xlabel("ASCII Value")
plt.ylabel("Frequency")

plt.figure('Figure 2')
plt.bar([0,1], freq_bnw, align='center')
plt.xticks([0,1], x_bnw)
plt.title("Label Distribution ")
plt.xlabel("ASCII Value")
plt.ylabel("Frequency")

plt.show()

        


