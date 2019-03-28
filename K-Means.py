# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 20:10:10 2019

@author: AbdelKarimA
"""

# ########################### #
# *  import needed package  * #
# ########################### #

from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
import operator
import random

# ########################### #
# *        Functions        * #
# ########################### #

# Read Points from Data.txt file
def Read_Data(filePath):
    
    file = open(filePath,"r")
    Data = file.read().splitlines()
    List_Points = []
    
    for point in Data:
        pair_Point = point.split('\t')
        List_Points.append(list(map(float, pair_Point)))
    
    return List_Points

# function return an array of distances from one point to ohter set of points
def Calculate_Distances(Center,Points):
    
    # array contains all the distances from center to all points
    dst_Center_Points = {}
    
    # calculate distance from center to each point in points
    for point in Points:
        dst_Center_Points[Points.index(point)] = distance.euclidean(Center, point)
    
    return dst_Center_Points

# function calculate the initial centers
def Get_InitialCenters(DataPoints,NumberOfCenters):
    
    # array Of Centers
    Centers = []
    Points = DataPoints[:]
    # 1. Calculate center 1
    Center1 = random.choice(Points)
    Centers.append(Center1)
    
    # 2. Calculate rest of the centers
    if NumberOfCenters != 1:
        
        Center = Center1
        
        for NumCenter in range(NumberOfCenters-1):    
            
            dist_CenterTo_Points = Calculate_Distances(Center,Points)
            Max_Dist = (max(dist_CenterTo_Points.items(), key=operator.itemgetter(1)))
            New_Center = Points[Max_Dist[0]]
            Centers.append(New_Center)
            Point_Index = Points.index(Center)
            del Points[Point_Index]
            Center = New_Center
    
    return Centers

# function calculate new centers
def Calculate_NewCenters(ClassifiedPoints):
    
    # array Of Centers
    Centers = []
    
    # loop each on each class in the dictionary
    for Class, Points in ClassifiedPoints.items():
        
        # get all x values for class points
        X_Values = [value[0] for value in Points]
        # get all y values for class points
        Y_Values = [value[1] for value in Points]
        # calculate the new center values and append it to Centers array
        Centers.append([sum(X_Values)/len(X_Values),sum(Y_Values)/len(Y_Values)])
    return Centers

# calcaulte average for centers distances
def calculate_avg(clustersDistances):
    
    sumof_avg = 0
    
    for key,value in clustersDistances.items():
        sumof_avg += value[0]/value[1]
    return sumof_avg

# function apply k-means algorithm on the provided points 
def K_Means(Points,K,iterations):
    
    # this list contains all the centers calcualted till convergence, the number of centers is based on the number of iterations 
    list_OfCenters = []
    # this list contains the average distance calculated for each iteration.
    list_OfDistances = []
    # this list contains the points clustered based on the K value for each iteration
    list_OfClassifiedPoints = []
    
    # 1. looping till the number of itrations.
    for iteration in range(iterations):
        
        # 2. calculate initial centers
        Initial_Centers = Get_InitialCenters(Points,K)
        
        Previous_Centers = [0,0]
        New_Centers = Initial_Centers
        
        # 3. looping till the previous and new center are equal, that indicate convergence
        while Previous_Centers != New_Centers:
           
            Previous_Centers = New_Centers
            
            # Dictionary contains calss the calss and their calssification points
            Calssified_Points = dict.fromkeys(range(K),[])
            # a variable hold the summation of Distances
            Sum_OfDistances = dict.fromkeys(range(K),[])
            # 3.1 looping each point in points 
            for Point in Points:
                # 3.2 calculate distances between the point and the centers
                Distances = Calculate_Distances(Point,New_Centers)
                # 3.3 pick the min distance from Distances
                MinDistance = (min(Distances.items(), key=operator.itemgetter(1)))
                
                if len(Sum_OfDistances[MinDistance[0]]) == 0:
                    
                    Sum_OfDistances[MinDistance[0]] = [MinDistance[1],1]
                else:
                    Sum_OfDistances[MinDistance[0]][0] += MinDistance[1]
                    Sum_OfDistances[MinDistance[0]][1] += 1
                    
                # 3.4 set the point to their respective cluster in Calssified_Points dictionary
                if len(Calssified_Points[MinDistance[0]]) == 0:
                    
                    Calssified_Points[MinDistance[0]] = [Point]
                else:
                    Calssified_Points[MinDistance[0]].append(Point)
            # 3.4 calculate the new centers
            New_Centers = Calculate_NewCenters(Calssified_Points)
        
        list_OfDistances.append(calculate_avg(Sum_OfDistances))
        list_OfClassifiedPoints.append(Calssified_Points)
        list_OfCenters.append(New_Centers)
    return list_OfCenters,list_OfDistances,list_OfClassifiedPoints

# function pick the best centers from all the the centers obtained from the iterations and plot the data 
def Pick_BestCenter(List_OfCenters,Distances,ClassifiedPoints):
    
    # to pick the best centers values, get min average centers
    minDistance = min(Distances)
    bestIndex = Distances.index(minDistance)
    # get points clustered for this best centers
    points = ClassifiedPoints[bestIndex]
    print(" *** Best Center Values *** ",List_OfCenters[bestIndex])
    print(" *** Average distances for best centers *** ",minDistance)
    # plot clustered points relate to best centers values
    for key,points_value in points.items():
        
        X_Values = [value[0] for value in points_value]
        Y_Values = [value[1] for value in points_value]
        plt.scatter(X_Values,Y_Values,marker=11)
    
    bestCenters = List_OfCenters[bestIndex]
    plt.scatter([value[0] for value in bestCenters],[value[1] for value in bestCenters],marker="D")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('K-Means Clustring')
    plt.show()

# ########################### #
# ****  Global Variables **** #
# ########################### #

# **** Training data info ****
# Data File Path
Data_FolderPath = "C:/Users/abdelkarima/data science  diploma/Introduction to machine learning/Labs/Assignment 4/Data.txt"
# Number of K
K=3
# Number of iterations
No_Iterations = 100

# ########################### #
# ********** Logic ********** #
# ########################### #

# 1. Read Data
Points = Read_Data(Data_FolderPath)

# 2. appply k-means on the points data
Calculated_Centers,Distances,ClassifiedPoints = K_Means(Points,K,No_Iterations)

# 3. pick best center from all centers and plot points clustered based on that best centers
Pick_BestCenter(Calculated_Centers,Distances,ClassifiedPoints)