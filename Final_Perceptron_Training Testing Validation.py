# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 13:36:10 2018

@author: AbdElRahman Salah AbdelKarim

This Code contain all the phases Training, Testing and validation

"""
# ########################### #
# *  import needed package  * #
# ########################### #

from collections import Counter
import matplotlib.pyplot as plt 
from scipy import misc
import numpy as np
import datetime
import random
import time
import os                   

# ########################### #
# *        Functions        * #
# ########################### #

# This function calculate the error for each point in points and return the error summation and points errors info 
def CalculatePoints_Error(Points,WeightVector,TargetValues):
    
    # declare dictionary hold the misscalssified points in their respective errors
    missclassifiedPoints_Errors = []
    
    # declare a variable for errors summation
    Errors_Summation = float(0.0)
    
    # iterate on each point to check if it's misscalssified or not
    for index in range(len(Points)):
        
        # Calculate Error based on equation Wt * Xn * Tn
        Error = np.dot(WeightVector,Points[index].transpose())*TargetValues[index]
        
        #check the error
        if Error <= 0 : 
            missclassifiedPoints_Errors.append([index,Error,TargetValues[index]])
            Errors_Summation += (Error*-1)
            
    return missclassifiedPoints_Errors,Errors_Summation;

# initiate the Target values vector for the classified points based on
def TargetsVector_Initiation(Points_Classes,TargtedClassify_Class):
    
    # declare an empty array for the target values
    TargetVector = []
    
    # iterate on each point to map each point to its target value
    for Point in Points_Classes:
    
        if Point == TargtedClassify_Class :
            TargetVector.append(1)
        else :
            TargetVector.append(-1)
    
    return np.array(TargetVector);

# Perceptron function that return all the weight vectors to all the classes
def perceptron(Points,Points_Classes,InitialWeight_Vector,Learning_Rate):
    
    # dictionary that contains the classes and their final respective weight vectors
    FinalWeightVectors = {}
    
    # count the number of classes the algorithm will classify points to
    Classes = list(Counter(Points_Classes).keys())
        
    # Looping the classes to calculate the weight vector for each class
    for Class in Classes:
        
        Weight_Vector = InitialWeight_Vector
       
        # Target values vector realte to this class 
        TargetValues = TargetsVector_Initiation(Points_Classes,Class)
        
        # 1. Calulate error for each point and the summation of all the errors with respect to the initial vector
        Points_Errors_Info = CalculatePoints_Error(Points,Weight_Vector,TargetValues)
        Errors_RespectTo_Vector = Points_Errors_Info[0]
        Errors_Summation = (Points_Errors_Info[1])
        
        #2. keep iterating to decrease the error summation till it becomes 0 
        while len(Points_Errors_Info[0]) != 0 :
            
            # 3. Calculate new weight vector
            # 3.1 pick one of the misclassified points
            PickRandomPoint = random.choice(Errors_RespectTo_Vector)
            PointIndex = PickRandomPoint[0]
          
            # 3.2 Calculate new weight vector
            Weight_Vector = np.add(Weight_Vector.transpose(),(Learning_Rate*Points[PointIndex].transpose()*TargetValues[PointIndex]))
            
            # 4. Calculate error using the new weight vector.
            Points_Errors_Info = CalculatePoints_Error(Points,Weight_Vector,TargetValues)
            Errors_RespectTo_Vector = Points_Errors_Info[0]
            Errors_Summation = Points_Errors_Info[1]
        
        FinalWeightVectors[Class] = Weight_Vector
        
    return FinalWeightVectors;

# function used for creating the confusion matrix and plot the image for it
def CreatConfusionMatrix(TestingPoints,TestingPointsClasses,TrainedWeightVectors):
    
    # array that contains the respective classes for each point
    ConfusionMatrix_Classification = np.zeros((10,10))
    
    # index for iterating points
    index = 0
    
    # count the number of classes the algorithm will classify points to
    Classes = list(Counter(TestingPointsClasses).keys())
    
    # Looping the classes
    for Class in Classes:
        
        while index < len(TestingPoints) and TestingPointsClasses[index] == Class :
            
            # Array contians the result of f(x) for each weight vector
            PointsClassification = []
            
            for key,vector in TrainedWeightVectors.items():
                
                Weightvector = np.array(vector)
                
                PointValue = np.dot(Weightvector,TestingPoints[index])
                
                PointsClassification.append(PointValue)
            
            MaxPointValue = max(PointsClassification)
            
            ClassOf_MaxPoint = PointsClassification.index(MaxPointValue)
            
            ConfusionMatrix_Classification[int(TestingPointsClasses[index]),ClassOf_MaxPoint] += 1
            index += 1;
    
    # Ploting image
    print("Image for Confusion Matrix")
    plt.matshow(ConfusionMatrix_Classification, cmap=plt.get_cmap('Blues'))
    plt.show()
    
    # Calculate accuracy
    Accuracy = 0
    for i in range(10):
        Accuracy += ConfusionMatrix_Classification[i][i]
    Accuracy = Accuracy/(len(TestingPoints)) * 100
    print("*** Accuracy ***",Accuracy)
    print("Confusion Matrix",ConfusionMatrix_Classification)
    return ConfusionMatrix_Classification;


# function read images in certain directory and reshape them into two dimensions array
def ReshapeImages_To_Points(FolderPath):
    
    os.chdir(FolderPath)
    files = os.listdir(FolderPath)
    files.pop()
    files = sorted(files,key=lambda x: int(os.path.splitext(x)[0]))
    
    Images_Points = []
    
    for i in files:    
        img=misc.imread(i)
        type(img)
        img.shape
        #change dimension to 1 dimensional array instead of (28x28)
        img=img.reshape(784,)
        img=np.append(img,1)
        Images_Points.append(img)
    
    return np.array(Images_Points);

# ########################### #
# ****  Global Variables **** #
# ########################### #

# **** Training data info ****

# training images path
Images_FolderPath = "C:/Users/abdelkarima/data science  diploma/Labs/Lab 2/Assignment Dataset/Train/"

# Reshape images to points
Images_Points = ReshapeImages_To_Points(Images_FolderPath)

# map points with their respective classes
Images_Classes = [num for num in range(10) for i in range(240)]

# intialize the the initail weight vector
Initail_WeightVector = np.array([0] * 785)
Initail_WeightVector[0] = 1

# learing  rates
Learning_Reates = [1,10**-1,10**-2,10**-3,10**-4,10**-5,10**-6,10**-7,10**-8,10**-9]

# Dictionary contains all the etas and their trained weight vectors
EtasWeightVextors = {}

# **** Testing data info ****
Images_TestingFolderPath = "C:/Users/abdelkarima/data science  diploma/Labs/Lab 2/Assignment Dataset/Test/"
Images_TestingPoints = ReshapeImages_To_Points(Images_TestingFolderPath)
Images_TesstingClasses = [num for num in range(10) for i in range(20)]
TestingConfusionMatrices = {}
# **** Validation data info ****
Images_ValidationFolderPath = "C:/Users/abdelkarima/data science  diploma/Labs/Lab 2/Assignment Dataset/Validation/"
Images_ValidationPoints = ReshapeImages_To_Points(Images_TestingFolderPath)
Images_ValidationClasses = Images_TesstingClasses 

# Dictionary contains the classe and their best weight vector
BestAccuracyClassifier = {}

# Dictionary contains all the etas and their confusion matrices
EtasConfusionMatrices = {}

# ########################### #
# ********** Logic ********** #
# ########################### #

for Learning_Reate in Learning_Reates:
    
    # start execution time
    start_time = time.time()
    
    # *** Training Phase
    print("Calculate Eta",Learning_Reate)

    FinalWeightVectors = perceptron(Images_Points,Images_Classes,Initail_WeightVector,Learning_Reate)
    EtasWeightVextors[Learning_Reate] = FinalWeightVectors
    
    # End Execution Time
    TimeInSeconds = (time.time() - start_time)
    TimeFormat = str(datetime.timedelta(seconds=TimeInSeconds))
    print("** Elapsed Time",TimeFormat)
    
    # *** Testing Phase
    TestConfusionMatrix = CreatConfusionMatrix(Images_TestingPoints,Images_TesstingClasses,FinalWeightVectors)
    TestingConfusionMatrices[Learning_Reate] = TestConfusionMatrix

# *** Validation Phase
for Eta,EtaWeightVectors in EtasWeightVextors.items():
    
    ValidateConfusionMatrix = CreatConfusionMatrix(Images_ValidationPoints,Images_ValidationClasses,EtaWeightVectors)
    EtasConfusionMatrices[Learning_Reate] = ValidateConfusionMatrix

BestAccuracy = [0] * 10

for EtaM,Matrix in EtasConfusionMatrices.items():
    
    for index in range(10):
        if Matrix[index][index] >  BestAccuracy[index]:
            BestAccuracy[index] = Matrix[index][index]
            Eta_WeightVectors = EtasWeightVextors[EtaM]
            BestAccuracyClassifier[index] = Eta_WeightVectors[index]

# Test again best accuracy classifiers on testing images
FinalConfusionMatrix = CreatConfusionMatrix(Images_TestingPoints,Images_TesstingClasses,BestAccuracyClassifier)
