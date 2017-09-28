# -*- coding: utf-8 -*-
"""
Created on Fri May 19 13:06:36 2017

@author: KRapes
"""

import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from skimage.filters import threshold_local
from skimage import img_as_ubyte
import pickle
from collections import Counter
from scipy import ndimage
from scipy.misc import imresize
from AreaOfInterest import ravel, predict_score, seperate_data
from Tensorflow_network import TF_model, test_model
from dataset_lib import normalize, ascii_to_map, one_hot_encode
from time import time
#import matplotlib.mlab as mlab

plt.ion()
#matplotlib.rcParams['backend'] = "GTKAgg"

# Load an color image in grayscale
def image_names(folder):
    files = []
    for file in os.listdir(folder):
        if file.endswith('.jpg') or file.endswith('.png'):
            files.append(os.path.join(folder, file))
    return files

def Canny_Edge_Detection(img, sigma=0.33):
    #Threshold1 = 25;
    #Threshold2 = 100;
    Filtersize = 10
    #sigma = 0.33
    med = np.median(img)
    Threshold1 = int(max(0, (1.0 - sigma) * med))
    Threshold2 = int(min(255, (1.0 + sigma) * med))
 
        
    E = cv2.Canny(img, Threshold1, Threshold2, Filtersize)
    return E
    
def rotate_img(img, degrees):
    height, width = img.shape
    M = cv2.getRotationMatrix2D((width/2, height/2),degrees,1)
    dst = cv2.warpAffine(img,M,(width,height))
    return dst
    
def straighten_image(img):
    def scale_up(value,step):
        return int(value/step)
    def scale_down(value,step):
        return value*step
     
    height, width = img.shape
    devs = {}
    step = 0.3
    min_deg = scale_up(-20, step)
    max_deg = scale_up(20, step)
    for n in range(min_deg,max_deg):
        n = scale_down(n,step)
        dst = rotate_img(img, n)
        dev = np.std(horz_proj(dst))
        devs[n] = dev
    return max(devs, key=devs.get)

def bound_image(dst):
    def cut_edge(proj, start, stop, step):
        mean = np.mean(proj)
        for i in range(start, stop, step):
            value = proj[i]
            if value > mean:
                break
        return i
    hproj = horz_proj(dst)
    l_hproj = len(hproj)
    top = cut_edge(hproj, 0, l_hproj, 1)
    bottom = cut_edge(hproj, l_hproj-1, -1, -1)
    vproj = vert_proj(dst)
    l_vproj = len(vproj)
    left = cut_edge(vproj, 0, l_vproj, 1)
    right = cut_edge(vproj, l_vproj-1, -1, -1)
    return top, bottom, left, right

def crop_image(img, top, bottom, left, right):
    return img[top:bottom, left:right]
    
def horz_proj(img):
    height, width = img.shape
    h_proj = []
    for i in range(height):
        h_proj.append(cv2.sumElems(img[i])[0])    
    return h_proj
    
def vert_proj(img):
    height, width = img.shape
    v_proj = []
    for i in range(width):
        v_proj.append(cv2.sumElems(img[:, i])[0])
    return v_proj
    
def show_img(i):
    cv2.imshow('image',i)
    k = -1
    while(k == -1):
        k = cv2.waitKey(33)
    cv2.destroyAllWindows()
    return k
    
def adaptive_thresholding(h_img, block_size=105):
    #block_size = 105
    
    adaptive_thresh = threshold_local(h_img, block_size, offset=10)
    binary_adaptive = h_img > adaptive_thresh
    h_img = img_as_ubyte(binary_adaptive)
    
    #ret3,h_img = cv2.threshold(blur(h_img),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return h_img

def blur(img):
    blurred = cv2.GaussianBlur(img, (3,3), 0)
    return blurred

def text_lines(boxes, img, whole_image):
    def combine(b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        x = x1 if x1 < x2 else x2
        y = y1 if y1 < y2 else y2
        w = (x1 + w1) - x if (x1 +w1) > (x2 + w2) else (x2 + w2) - x
        h = (y1 + h1) - y if (y1 +h1) > (y2 + h2) else (y2 + h2) - y
        #print("New Box: {}".format((x,y,w,h)))
        return (x, y, w, h)
    
    def remove_boxes(boxes, function, *args):
        index = 0
        while index < len(boxes):
            b = boxes[index]
            if function(b, *args):
                boxes.pop(index)
                index -= 1
            index += 1
        return boxes
    
    def long(b, thresholdA, thresholdB):
        box_width = b[2]
        if box_width >= thresholdA or box_width <= thresholdB:
            return True
        return False
        
    def tall(b, thresholdA, thresholdB):
        box_height = b[3]
        if box_height >= thresholdA or box_height <= thresholdB:
            return True
        return False
          
    def horz(b1,b2,threshold):
        if ((b1[1] <= b2[1] + threshold and b1[1] >= b2[1] - threshold) and 
                    (b1[3] <= b2[3] + threshold and b1[3] >= b2[3] - threshold)):
            return True
        return False
    
    def vert(b1,b2,threshold):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        if ((x1 <= x2 + threshold and x1 >= x2 - threshold) and
            (w1 <= w2 + threshold and w1 >= w2 - threshold)):
            return True
        return False
            
            
    def interior_boxes(b1,b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        if y1 <= y2 and (y1 + h1) >= (y2 + h2):
            return True
        elif y2 <= y1 and (y2 + h2) >= (y1 + h1):
            return True
        return False 
    
    def interior_boxes_vert(b1,b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        if x1 <= x2 and (x1 + w1) >= (x2 + w2):
            return True
        elif x2 <= x1 and (x2 + w2) >= (x1 + w1):
            return True
        return False 
        
    def combine_overlapping_areas(b1, b2, threshold):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        ymin_interior = y1 if y1 > y2 else y2
        ymax_interior = (y1 + h1) if (y1 + h1) < (y2 + h2) else (y2 + h2)
        interior_length = ymax_interior - ymin_interior
        if interior_length >= h1*threshold or interior_length >= h2*threshold:
            return True
        return False
    
    def combine_overlapping_areas_vert(b1, b2, threshold):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        ymin_interior = x1 if x1 > x2 else x2
        ymax_interior = (x1 + w1) if (x1 + w1) < (x2 + w2) else (x2 + w2)
        interior_length = ymax_interior - ymin_interior
        if interior_length >= w1*threshold or interior_length >= w2*threshold:
            return True
        return False
         
    def combine_boxes(boxes, function, *args):
        cBoxes = []
        index = len(boxes) - 1
        while index >= 0:
            b1 = boxes[index]
            if index == 0:
                cBoxes.append(boxes.pop(0))
                index = len(boxes) - 1
            else:
                b2 = boxes[index - 1]
                if function(b1,b2,*args):
                    new_box = combine(b1, b2)
                    boxes.pop(index)
                    boxes.pop(index - 1)
                    boxes.append(new_box)
                    index = len(boxes) - 1
                else:
                    cBoxes.append(boxes.pop(index))
                    index = len(boxes) - 1
        return cBoxes

    
    
    
    if whole_image == True:
        boxes.sort(key=lambda x: x[1])
        boxes = remove_boxes(boxes, tall, img.shape[0]*0.1, 1)
        boxes = combine_boxes(boxes,horz, 0)
        length_boxes = len(boxes)
        if length_boxes == 0: return boxes
        hsum = sum([x[3] for x in boxes])
        boxes = remove_boxes(boxes, tall, 2*hsum/length_boxes, .5*hsum/length_boxes)
        boxes = combine_boxes(boxes, interior_boxes)
        boxes = combine_boxes(boxes,horz, 3)
        boxes = combine_boxes(boxes, combine_overlapping_areas, 0.50)
    else:
        boxes.sort(key=lambda x: x[0])
        boxes = remove_boxes(boxes, long, img.shape[1]*.1, 3)
        boxes = combine_boxes(boxes, vert, 1)
        boxes = combine_boxes(boxes, interior_boxes_vert)
        boxes = combine_boxes(boxes, combine_overlapping_areas_vert, .50)
        boxes = remove_boxes(boxes, tall, img.shape[0]*1.1, img.shape[0]*.25)
    return boxes
    

def find_lines(img, whole_image=True):
    height, width = img.shape
    _, contours0, hierarchy = cv2.findContours( img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]
    boxes = []
    for c in contours:
        boxes.append(cv2.boundingRect(c))
    if whole_image != 'letter':
        boxes = text_lines(boxes, img, whole_image)
    return boxes

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)
        
def groom_image(image):
    img = cv2.imread(image,0)
    h_angle = straighten_image(Canny_Edge_Detection(blur(img)))
    img = rotate_img(img, h_angle)
    dst = Canny_Edge_Detection(img)
    top, bottom, left, right = bound_image(dst)
    img = crop_image(img, top, bottom, left, right)
    return img


    
def locate_receipt(img):
    for _ in range(2):
        h_angle = straighten_image(Canny_Edge_Detection(blur(img)))
        img = rotate_img(img, h_angle)
        dst = Canny_Edge_Detection(img)
        top, bottom, left, right = bound_image(dst)
        img = crop_image(img, top, bottom, left, right)
        dst = crop_image(dst, top, bottom, left, right)
    return img
    
def clip_lines(img):
    lines = []            
    adp = Canny_Edge_Detection(blur(img))
    boxes = find_lines(adp)
    height, width = img.shape
    for index,box in enumerate(boxes):
        x, y, w, h = box
        y_start = int(y - 0.25*h) if int(y - 0.25*h) > 0 else 0
        y_end = int(y + 1.25*h) if int(y + 1.25*h) < height else height
        clip = img[y_start:y_end, 0:width]
        lines.append([image, clip, 0, y_start, width, int(1.5*h)])
    return lines
    


class Pointer:
    
    def __init__(self, index, heat_map):
      self.index = index
      self.heat_map = heat_map
      if index < len(heat_map):
          self.value = heat_map[index] 
      else: 
          self.value = 0
          
    def step(self):
        self.index += 1
        if self.index < len(self.heat_map):
            self.value = self.heat_map[self.index]
        else:
            self.value = 0
    
    def value(self):
        return self.value
    
    def index(self):
        return self.index


def create_heat_map(boxes, img):
    height, width = img.shape
    boxes.sort(key=lambda x: x[0])
    heat_map = [0 for x in range(width)]
    for box in boxes:
        x, y, w, h = box
        reward = 0
        for index in range(x, x+w+1):
            if index <= (2*x+w)/2:                
                reward += -1
            else: 
                reward += 1
            heat_map[index] = -3 if reward <= -3 else reward
    return heat_map
    
def text_size_range(t_width_count, t_height_count):
    text_size = []
    for h in range(2):
        for w in range(5):
            ratio = t_height_count.most_common(h+1)[h][0] / t_width_count.most_common(w+1)[w][0]
            if ratio < 2 and ratio > 1:
                text_size.append(t_width_count.most_common(w+1)[w][0])
    if len(text_size) == 0:
        return [1, max(t_width_count)]
    min_w = min(text_size)
    max_w = max(text_size)
    std = np.std([min_w, max_w])
    text_size = [1 , min(max(t_width_count), int(max_w + 2*std))]
    return text_size

def best_fit_window(line, text_size):
    orgional, clip, x, y, w, h, heat_map = line
    height, width = clip.shape
    max_sum = None
    width_options = [x for x in range(text_size[0],text_size[1])]
    for t_width in width_options:
        max_t_width_sum = None
        index = 0
        pointers =[]
        while index < width:
            pointers.append(Pointer(index, heat_map))
            index += t_width
        for step in range(t_width):
            pointers_sum = 0
            for pointer in pointers:
                pointer.step()
                pointers_sum += pointer.value
            pointers_sum = pointers_sum / len(pointers)
            if max_t_width_sum == None or pointers_sum > max_t_width_sum:
                max_t_width_sum = pointers_sum
                sum_step = step
        if max_sum == None or max_t_width_sum > max_sum:
            max_sum = max_t_width_sum
            max_t_width = t_width
            max_step = sum_step
    return max_t_width, max_step
    
def create_breaks(window_width, index, heat_map):
    breaks = []
    breaks.append(index)
    width = len(heat_map)
    while index + window_width < width:
        pointer1 = Pointer(index,heat_map)
        pointer2 = Pointer(index + window_width, heat_map)
        if pointer1.value + pointer2.value >= -1:
            if index not in breaks:
                breaks.append(index)
            breaks.append(index + window_width)
            index = index + window_width
        else:
            index += 1
    return breaks

def breaks_to_clips(breaks, clip, relative_x, relative_y):
    prev_split = breaks[0]
    height, width = clip.shape
    clipsL = []
    for split in breaks[1:]:
        clipL = clip[0:height, prev_split:split]
        global_x = prev_split + relative_x
        global_y = relative_y
        w = split - prev_split
        h = height
        clipsL.append( [clipL, global_x, global_y, w, h])
        prev_split = split
    return clipsL



def replicate_receipt(img, clipsL):
    def find_font_scale(prediction, h):
        fontscale = 1
        while(1):
                bounding_box, _ = cv2.getTextSize(prediction, font, fontscale, thickness)
                if bounding_box[1] <= h*.8:
                    fontscale += 0.1
                elif bounding_box[1] >= h*1.2:
                    fontscale -= 0.1
                else:
                    return fontscale
                    
    height, width = img.shape
    computed_image = np.zeros((height, width), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    thickness = 5
    color = (255,255,255)
    img2 = img.copy()
    predictions = TF_model(normalize([clip[1] for clip in clipsL]))
    
    i = 0
    for clip, prediction in zip(clipsL, predictions):
        i += 1
        orgional, clipL, global_x, global_y, w, h = clip
        cv2.rectangle(img2, (global_x,global_y),(global_x+w,global_y+h), (0,255,0),2)
        fontscale = find_font_scale(chr(prediction), h)
        cv2.putText(
                    computed_image,
                    chr(prediction),
                    (global_x,global_y + h), 
                    font,
                    fontscale,
                    color,
                    thickness,
                    cv2.LINE_AA)
    
    #plt.close('all')
    plt.figure(orgional)
    plt.imshow(img2, cmap='gray')
    
    plt.figure('Computed')
    plt.imshow(computed_image, cmap='gray')
    plt.show
    #key = show_img(clipL)
    
    
def clip_letter(lines):
    t_width_count = Counter()
    t_height_count = Counter()
    for line in lines:
        orgional, clip, x, y, w, h = line
        height, width = clip.shape     
        dst = Canny_Edge_Detection(blur(adaptive_thresholding(clip)))
        boxes = find_lines(dst, whole_image=False)        
        heat_map = create_heat_map(boxes, clip)
        line.append(heat_map)
        t_width_count += Counter([x[2] for x in boxes])
        t_height_count += Counter([x[3] for x in boxes])
    text_size = text_size_range(t_width_count, t_height_count)

    table_width = Counter()
    for line in lines:
        max_t_width, max_step = best_fit_window(line, text_size)
        table_width += Counter([max_t_width])
    
    clipsL = []
    for line in lines:
        orgional, clip, x, y, w, h, heat_map = line
        height, width = clip.shape
        window_width = table_width.most_common(1)[0][0]
        window_width, step = best_fit_window(line, [window_width, window_width + 1])
        breaks = create_breaks(window_width, step, heat_map)
        for entry in breaks_to_clips(breaks, clip, x, y):
            clipsL.append([orgional] + entry)
    
    index = 0
    image_window_width = window_width
    new_clipsL = []
    for line in clipsL:
        height, width = img.shape
        orgional, clipL, global_x, global_y, global_w, global_h = line
        if global_w < image_window_width*.33:
            clipsL.pop(index)
        elif global_w > image_window_width*1.5:
            dst = Canny_Edge_Detection(blur(adaptive_thresholding(clipL)))
            boxes = find_lines(dst, whole_image='letter')        
            heat_map = create_heat_map(boxes, clipL)
            line2 = line.copy()
            line2.append(heat_map)
            window_width, step = best_fit_window(line2, [1, max(2, int(global_w*.75))])
            breaks = create_breaks(window_width, step, heat_map)
            for entry in breaks_to_clips(breaks, clipL, global_x, global_y):
                new_clipsL.append([orgional] + entry)
    for clip in new_clipsL:
        orgional, clipL, global_x, global_y, global_w, global_h = clip
        if global_w > image_window_width*.33:
            clipsL.append(clip)


       
    
        
    for i in range(len(clipsL)):
        orgional, clipL, global_x, global_y, w, h = clipsL[i]
        padding = 2
        clipL = cv2.bitwise_not(clipL)
        y_COM, x_COM = ndimage.measurements.center_of_mass(clipL)
        common_height = t_height_count.most_common(1)[0][0]
        y_globalCOM = global_y + y_COM
        new_global_y = int(max(y_globalCOM - common_height/2 - padding, global_y))
        h = int(min(common_height + padding * 2, h))
        top = new_global_y - global_y
        bottom = min(top + h, clipL.shape[0])
        clipL = clipL[top:bottom]
        clipsL[i] = [orgional, clipL, global_x, new_global_y, w, h]
        
    return clipsL

def format_letter(clipL):
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

def label_characters(clipsL, fname):
    start = time()
    data = []
    average_width = np.mean([x[4] for x in clipsL])
    average_height = np.mean([x[5] for x in clipsL])
    for clip in clipsL:
        orgional, clipL, global_x, global_y, w, h = clip
        height, width = img.shape
        xnorm = global_x/width
        ynorm = global_y/height
        wnorm = w/average_width
        hnorm = h/average_height
        img2 = img.copy()[global_y - 3:global_y + h + 3, 0:width]
        cv2.rectangle(img2, (global_x,0),(global_x+w,h + 3), (255,255,0),2)
            
        plt.close('all')
        plt.imshow(img2, cmap='gray')
        plt.show
        key = show_img(clipL)            
        data.append([clipL, xnorm, ynorm, wnorm, hnorm, key])
    save_obj(data, fname.split('.')[0])
    print("Label Photo: {}".format((time() - start)/60))
    return data

def remove_deadspace(clipsL, clf1, clf2):
    i = 0
    average_width = np.mean([x[4] for x in clipsL])
    average_height = np.mean([x[5] for x in clipsL])
    while i < len(clipsL): 
        #print(i)
        orgional, clipL, global_x, global_y, w, h = clipsL[i]
        height, width = img.shape
        prediction = clf1.predict(ravel([clipL]))
        xnorm = global_x/width
        ynorm = global_y/height
        wnorm = w/average_width
        hnorm = h/average_height
        if clf2.predict([prediction, xnorm, ynorm, wnorm, hnorm]) == 1:
            clipsL.pop(i)
        else: 
            i += 1
    return clipsL

def load_files(fname, skipdata=False):
    if skipdata == False:
        if os.path.isfile(fname):
            with open( fname, 'rb') as f:
                data = pickle.load(f)    
        else:
            data = label_characters(clipsL, fname)
    else:
        data = None
        
    if os.path.isfile("AOI_1.pkl") and os.path.isfile("AOI_2.pkl"):
        with open( "AOI_1.pkl", 'rb') as f:
            clf1 = pickle.load(f)
        with open( "AOI_2.pkl", 'rb') as f:
            clf2 = pickle.load(f)
    else:
        print("No AOI file")
        exit()
    return data, clf1, clf2

def score_image(data, clf1, clf2):
    x_train, y_train, x_test, y_test = seperate_data(data, 0, clf1)
    accuracy_DT = predict_score(x_train, y_train, clf2)
    valid_features = []
    valid_labels = []
    for i in range(len(data)):
        mapkey = ascii_to_map(data[i][5])
        if mapkey >= 0:
            valid_features.append(data[i][0].reshape(28, 28, 1))
            valid_labels.append(mapkey)
    accuracy_TF = test_model(valid_features, one_hot_encode(valid_labels))
    return accuracy_DT, accuracy_TF
'''
images = image_names("images")
print("START")
#count = 1
DT_acc = 0
TF_acc = 0
total_clips = 0
for image_count, image in enumerate(images):

    img = cv2.imread(image,0)
    img = locate_receipt(img)
    lines = clip_lines(img)
    # TODO break this into a yeild counter loop and a cutting algorthim
    clipsL = clip_letter(lines)

    for i in range(len(clipsL)):
        orgional, clipL, global_x, global_y, w, h = clipsL[i]
        clipL = format_letter(clipL)
        clipsL[i] = [orgional, clipL, global_x, global_y, w, h]
    
    fname = ".\labeled_images\labeled_images_" + image.split('.')[0].split('images\\')[1] + '.pkl'
    data, clf1, clf2 = load_files(fname, skipdata=True)
        
    total_clips += len(clipsL)
    #print("Length clipsL: {}".format(len(clipsL)))
    
    clipsL = remove_deadspace(clipsL, clf1, clf2)
    
    accuracy_DT, accuracy_TF = score_image(data, clf1, clf2)
    DT_acc += accuracy_DT[0]
    TF_acc += acc
    print("The Area of Interest accuarcy was {}".format(DT_acc/(image_count + 1)))
    print("The TF accuarcy was {}".format(TF_acc/(image_count + 1)))
    #print("Total Clips: {}".format(total_clips))
    print("")
    
    replicate_receipt(img, clipsL)
'''    
   
   
    
    

    
        
        
        
