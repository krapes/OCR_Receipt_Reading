README

Welcome to the Optical Character Recognition - Receipt Reading program 
written to fulfill the Udacity Machine Learning Nanodegree Capstone 
project.

Below are breif explaniations of the most important libraries used in the
project.

Opencv (cv2)
This library is aimed at real-time computer vision. It is a 
cross-platform library that supports both python and TensorFlow which were
used in the creation of this project. The goals of the library are to 
advance vision research by providing an optimized code with a common 
infrastructure that developers could build upon. In this project we 
relied on the library heavely to locate characters within an image of 
a receipt. 

Scikit-Learn (sklearn)
This is a software machine learning library for the Python programming 
language. The libraries goals are to provide various ML support including
classification, regression, clustering, support vector machines, random
forests, gradient boosting, and much more. In this program we used the 
the library for selection of block of images that contained
recognizable characters, also known as the "Decision Tree" or
"Area of Interest" section of the program.

TensorFlow (tensorflow)
This library is a system used to build and train neural networks with the 
objective of detecting and deciphering patterns and correlations in a 
form that resembles human learning. Within this project we used TensorFlow
to create a neural network that classifies characters from a 28x28 pixel 
image. 

Other Libraries Utilized:
Matplotlib
For displaying images to the user.
Numpy
For matrix manipulation and calculations.
Pickle
For saving and retrieving files. 


Using the program:
This program should not require any setup. It was origionally made to be
run from the Spyder IDE from the "Main.py". However, to make visulazation
a little easier, the high level code has been commented out in "Main.py"
and repeated in a Juypter Notebook named "High_Level_Flow". If you
encounter issues running the Notebook, I suggest uncommenting the code in
"Main.py" and running from there. 

The program pulls images from the top level of the "Image" folder.
Currently there are three images that were used for testing. To see the 
results of more images, the items within "Report_Images" can be moved to 
the top level of the folder. 

If the program encounters an image that it does not have a labeled pickle
file for it will begin the labeling process. Or in other words it will 
it will begin displaying sections of the text to the user with the 
expectation that the user will label the character (or lack of character)
using the keyboard. 

 
