# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 16:07:12 2017

@author: KRapes
"""

import os
import struct
import numpy as np
import cv2
from array import array
import gzip
import os.path
import pickle
from collections import Counter
from scipy import ndimage
from scipy.misc import imresize




USELOCALDATA = False

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def file_names(folder):
    files = []
    for file in os.listdir(folder):
        if file.startswith("labeled_images_"):
            files.append(os.path.join(folder, file.split('.')[0]))
    return files

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
def show_img(i):
    cv2.imshow('image',i)
    while(10):
        k = cv2.waitKey(33)
        if k==116:    #t key for positive
            output = True         
            break
        elif k==102:  # f for negative
            output = False
            break
        elif k==-1:  # normally -1 returned,so don't print it
            continue
        else:
            print("I'm sorry, press 't' for positive or 'f' for negative") # else print its value
    cv2.destroyAllWindows()
    return output


def read(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """
    print("In Read      Dataset: {}".format(dataset))
    
    
        
    
    if dataset != "training" and dataset != "testing":
        print("In image files")
        files = file_names('./labeled_images')
        for count, file in enumerate(files):
            data = load_obj(file)
            data = [(ascii_to_map(n[5]), n[0]) for n in data]
            yield data, count
    else:
        if dataset is "training":
            fname_img = os.path.join(path, 'emnist-bymerge-train-images-idx3-ubyte')
            fname_lbl = os.path.join(path, 'emnist-bymerge-train-labels-idx1-ubyte')
            divisor = 10
            print("Training")
        elif dataset is "testing":
            fname_img = os.path.join(path, 'emnist-bymerge-test-images-idx3-ubyte')
            fname_lbl = os.path.join(path, 'emnist-bymerge-test-labels-idx1-ubyte')
            divisor = 5
            print("Testing")
        
        print("divisor: {}".format(divisor))
         # Load everything in some numpy arrays
        with open(fname_lbl, 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            lbl = np.fromfile(flbl, dtype=np.int8)
        
        with open(fname_img, 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = np.fromfile(fimg, dtype=np.uint8)
            img = img.reshape(num, rows, cols)    
      
        count = 0
          
        while int(count*len(lbl)/divisor) < len(lbl):
            count += 1
            print("Count: {}".format(count))
            get_img = lambda idx: (lbl[idx], format_letter(flip(rotate(img[idx]))))
        
            # Create an iterator which returns each image in turn
            data = []
            lower_bound = int((count - 1)*len(lbl)/divisor)
            upper_bound = min(int(count*len(lbl)/divisor), len(lbl))
            for i in range(lower_bound, upper_bound):
                data.append(get_img(i))
            yield data, count

        
        '''
        for file in [fname_lbl, fname_img]:
            outfilename = file[:len(file) - 3]
            inF = gzip.open(file, 'rb')
            outF = open(outfilename, 'wb')
            outF.write( inF.read() )
            inF.close()
            outF.close()
            print("File {} unzipped".format(outfilename))
        '''
 
    
def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    image = image.reshape(28,28)
    #image = list(image)
    from matplotlib import pyplot
    #import matplotlib as mpl
    pyplot.close('all')
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap='gray')
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()
    show_img(image)

def rotate(img):
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
    return cv2.warpAffine(img,M,(cols,rows))

def flip(img):
    return np.fliplr(img)
    

def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    # TODO: Implement Function
    import sklearn.preprocessing

    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(range(46))
    b = label_binarizer.transform(x)
    
    #print('{}'.format(b))
    return b

def create_map(direction):
    mapping = {}
    tf = open("./mnist/emnist-bymerge-mapping.txt", 'r')
    for line in tf.readlines():
        translation_table = dict.fromkeys(map(ord, '\n'), None)
        line = line.translate(translation_table)
        tup = line.split(' ')
        if direction == 'map_to_ascii':
            mapping[int(tup[0])] = int(tup[1])
        elif direction == 'ascii_to_map':
            mapping[int(tup[1])] = int(tup[0])
        else:
            print("understandable map direction")
    return mapping
    
def map_to_ascii(index_list):
    mapping = create_map('map_to_ascii')
    values = []
    for index in index_list:
        try: index = index[0]
        except TypeError: index = index
        finally: values.append(mapping[index])
    return values

def ascii_to_map(asc):
    lower_to_upper = {99: 67, 105: 73, 106: 74, 107: 75, 108: 76, 111: 79, 112: 80, 115: 83, 117: 85, 
                      118: 86, 119: 87, 120: 88, 121: 89, 122: 90}
    mapping = create_map('ascii_to_map')

    if asc in lower_to_upper:
        asc = lower_to_upper[asc]    
    
    if asc in mapping:
        return mapping[asc]
    else:
        return -1

    

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (28,28,1)
    : return: Numpy array of normalize data
    """
    x = np.array(x)
    x=x/255
    x = x.reshape(-1,28,28,1)

    return x
    
def mash_data_w_lables(features, lables):
    return [(lable, feature) for lable, feature in zip(lables, features)]
    


def get_data(batch_id, dataset='training', path='./mnist'):
           
    fname = dataset + '_' + str(batch_id) + '.pkl'
    if os.path.isfile(fname):
        with open( fname, 'rb') as f:
            data = pickle.load(f)
            return data
    else:
        print("Processing dataset {}".format(dataset))
        for  data, count in read(dataset, path):
            data = mash_data_w_lables(normalize([x[1] for x in data]), one_hot_encode([x[0] for x in data]))
            with open( dataset + '_' + str(count) + '.pkl', 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        if os.path.isfile(fname):
            return get_data(batch_id, dataset=dataset, path=path)
        else:
            return None
            
def blur(img):
    blurred = cv2.GaussianBlur(img, (3,3), 0)
    return blurred

def format_letter(clipL):
    #clipL = clipL.reshape(28,28)
    #clipL = (clipL*255).astype('uint8')
    ret3,clipL = cv2.threshold(blur(clipL),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    clipL = imresize(clipL, (24,24), interp='bilinear', mode=None)
    clipL = cv2.copyMakeBorder(clipL,2,2,2,2,cv2.BORDER_CONSTANT,value=[0,0,0])
    ret3,clipL = cv2.threshold(blur(clipL),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    y_COM, x_COM = ndimage.measurements.center_of_mass(clipL)
    height, width = clipL.shape
    tx = int(width/2 - x_COM)
    ty = int(height/2 - y_COM)
    M = np.float32([[1,0,tx],[0,1,ty]])
    clipL = cv2.warpAffine(clipL,M,(width,height), cv2.BORDER_CONSTANT, 0)
    
    return clipL
# TODO test accuracy if the EMNIST images are reformatted  with the same function as the real images            
 
'''
training_data = list(get_data(1))#print("Finished")
#img = np.multiply(format_letter(training_data[0][1]), 1)
#img = normalize([img])
#img = format_letter(training_data[0][1])
img = training_data[0][1]
print(img)
print("Shape: {}".format(img.shape))
print("lable: {}".format(training_data[0][0]))

show_img(img)
'''
#show_img(format_letter(img))
'''
dir = os.getcwd()
filename = dir + '\example_real.jpg'
print(filename)
cv2.imwrite(filename, img)
print(img)

print("Done")
'''
#get_data()
'''
for  i in range(1,2):
    data = get_data(i, dataset='testing')
    if data:
        print("length data: {}  count: ".format(len(data)))
    else:
        print(data)
print("Done")
'''
'''
print(len(training_data))
for i in range(len(training_data)):
    label, pixels = training_data[i]
    print(chr(map_to_ascii(list(label).index(1))))
    #print(pixels.shape)
    show(pixels)
'''
'''
with open( 'labeled_receipt_IMG_20170524_154436' + '.pkl', 'rb') as f:
        dataset = pickle.load(f)
for i, [image, label] in enumerate(dataset):
    dataset[i][1] = ascii_to_map(label)
for image, label in dataset:
    print(label)
'''