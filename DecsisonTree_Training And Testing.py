# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 21:42:49 2019

@author: AbdelKarimA
"""
# ########################### #
# *  import needed package  * #
# ########################### #
from sklearn.metrics import accuracy_score
from collections import Counter
import numpy as np
import statistics
import operator
import pandas
import math

# ########################### #
# *        Functions        * #
# ########################### #

# Read Csv data
def Read_CSVData(FilePath):
    
    df = pandas.read_csv(FilePath) 
    return df

# Calculate Entropy
def Calculate_Entropy(Data,Decision_Attribute,Postive_Value,Negative_Value):
    
    Pos_Values = Data[(Data[Decision_Attribute] == Postive_Value)]
    Count_Pos = Pos_Values[Decision_Attribute].count()

    Neg_Values = Data[(Data[Decision_Attribute] == Negative_Value)]
    Count_Neg = Neg_Values[Decision_Attribute].count()

    Total = Count_Pos + Count_Neg
    Entropy = float(0.0)
    
    if Total == 0:
        Entropy = 0.0
        
    elif Count_Pos == 0:
        Entropy = - ((Count_Neg/Total) * math.log(Count_Neg/Total,2))
    
    elif Count_Neg == 0: 
        Entropy = -((Count_Pos/Total) * math.log(Count_Pos/Total,2))
    
    else:
        Entropy = -((Count_Pos/Total) * math.log(Count_Pos/Total,2)) - ((Count_Neg/Total) * math.log(Count_Neg/Total,2))
    
    return round(Entropy,4)

# Split List to number of chuncks
def Chunck_List(ChuncksNumbers,List):
    
    ListMax = max(List)
    ListMin = min(List)

    ChunckingStep = (ListMax-ListMin) / float(ChuncksNumbers)

    return [ListMin+ChunckingStep,ListMin+(ChunckingStep*2)];

# Calculate Gain based on the mean of attribute values.
def Calculate_Gain_Mean(AttributeName,Data,Decision_Attribute,Postive_Decision,Negative_Decision):
    
    # 1. Calculate total number of rows for the Data
    TotalNumbers = int(Data.count()[0])
    
    # 2. Calculate the smaple Entropy for the Data
    SampleEntropy = Calculate_Entropy(Data,Decision_Attribute,Postive_Decision,Negative_Decision)
    
    # 3. Calculate the mean of the attribute values
    Column_Values = Data[AttributeName]
    Data_Mean = round(statistics.mean(Column_Values.tolist()),1)
    
    # 4. Calculate the entropy for the data greater than mean
    Attribute_Values1 = Data[(Data[AttributeName] > Data_Mean)]
    Attribute_Count1 = Attribute_Values1[AttributeName].count()
    Value1_Entropy = Calculate_Entropy(Attribute_Values1,Decision_Attribute,Postive_Decision,Negative_Decision)
    
    # 5. Calculate the entropy for the data less than and equal mean
    Attribute_Values2 = Data[(Data[AttributeName] <= Data_Mean)]
    Attribute_Count2 = Attribute_Values2[AttributeName].count()
    Value2_Entropy = Calculate_Entropy(Attribute_Values2,Decision_Attribute,Postive_Decision,Negative_Decision)
    
    # 6. Calculate the summation of entropies
    Summation_Entropies = (Attribute_Count1/TotalNumbers)*(Value1_Entropy) + (Attribute_Count2/TotalNumbers)*(Value2_Entropy)
    
    # 7. Calculate the Gain
    Gain = SampleEntropy - Summation_Entropies
    
    return [round(Gain,4),int(Data_Mean)];

# Calculate Gain based on Chuncking Data
def Calculate_Gain_Chuncking(AttributeName,Data,NumberOfChuncks,Decision_Attribute,Postive_Decision,Negative_Decision):
    
    # 1. Calculate total number of rows for the Data
    TotalNumbers = int(Data.count()[0])
    
    # 2. Calculate the smaple Entropy for the Data
    SampleEntropy = Calculate_Entropy(Data,Decision_Attribute,Postive_Decision,Negative_Decision)
    
    # 3. Calculate the Chuncking values
    Attribute_Values = (Data[AttributeName]).tolist()
    Chucnking_Values = Chunck_List(3,Attribute_Values)
    
    # 4. Calculate the entropy for each chunck
    # 4.1 First Chunck all the data greater than and equal Chucnking_Values[1]
    Attribute_Values1 = Data[(Data[AttributeName] >= Chucnking_Values[1])]
    Attribute_Count1 = Attribute_Values1[AttributeName].count()
    Value1_Entropy = Calculate_Entropy(Attribute_Values1,Decision_Attribute,Postive_Decision,Negative_Decision)
    
    # 4.2 Second Chunck all the data between Chucnking_Values[0] and Chucnking_Values[1]
    Attribute_Values2 = Data[(Data[AttributeName] >= Chucnking_Values[0]) & (Data[AttributeName] < Chucnking_Values[1])]
    Attribute_Count2 = Attribute_Values2[AttributeName].count()
    Value2_Entropy = Calculate_Entropy(Attribute_Values2,Decision_Attribute,Postive_Decision,Negative_Decision)
    
    # 4.3 Third Chunck all the data less than Chucnking_Values[0]
    Attribute_Values3 = Data[(Data[AttributeName] < Chucnking_Values[0])]
    Attribute_Count3 = Attribute_Values3[AttributeName].count()
    Value3_Entropy = Calculate_Entropy(Attribute_Values3,Decision_Attribute,Postive_Decision,Negative_Decision)

    # 5. Calculate the summation of entropies
    Summation_Entropies = (Attribute_Count1/TotalNumbers)*(Value1_Entropy) + (Attribute_Count2/TotalNumbers)*(Value2_Entropy) + (Attribute_Count3/TotalNumbers)*(Value3_Entropy)
    
    # 6. Calculate the Gain
    Gain = SampleEntropy - Summation_Entropies
    return [round(Gain,4),int(Chucnking_Values[0]),int(Chucnking_Values[1])];

# Calculate Gain on data that's already discritized, with two or three values.
def Calculate_GainChuncks(AttributeName,Data,Decision_Attribute,Postive_Decision,Negative_Decision):
    
    # 1. Calculate total number of rows for the Data
    TotalNumbers = int(Data.count()[0])
    
    # 2. Calculate the smaple Entropy for the Data
    SampleEntropy = Calculate_Entropy(Data,Decision_Attribute,Postive_Decision,Negative_Decision)
    
    # 3. Calculate entropies for an atrribute
    Summation_Entropies = 0.0
    
    # 3.1 loop on the number of values which is 3 to calculate the entropy for each value
    
    for Index in range(3):
        
        Attribute_Values = Data[(Data[AttributeName] == Index)]
        Attribute_Count = Attribute_Values[AttributeName].count()
        Entropy_Value = Calculate_Entropy(Attribute_Values,Decision_Attribute,Postive_Decision,Negative_Decision)
        Summation_Entropies += (Attribute_Count/TotalNumbers)*Entropy_Value

    # 4. Calculate the Gain
    Gain = SampleEntropy - Summation_Entropies
    return [round(Gain,4),2,1,0];

# Function collect the Gains for list attributes and return the max Gain attribute
def Attributes_Gains(Data,Attributes_Columns,Decision_Attribute,Postive_Decision,Negative_Decision):
 
    GainsInfo = {}
    
    # looping in each attribute an calculate gain for it
    for attribute in Attributes_Columns:
        
        if attribute == 'AR' or attribute == 'HR':
            GainsInfo[attribute] = Calculate_GainChuncks(attribute,Data,Decision_Attribute,Postive_Decision,Negative_Decision)
        else:
            GainsInfo[attribute] = Calculate_Gain_Chuncking(attribute,Data,3,Decision_Attribute,Postive_Decision,Negative_Decision)
    
    return (max(GainsInfo.items(), key=operator.itemgetter(1)))

# This function return the majority of of H or A values in a list
def Get_Majority(Data,Decision_Attribute):
    
    Decision_Values = (Data[Decision_Attribute].tolist())
    Counters = Counter(Decision_Values)
    return Counters.most_common()[0][0]

# Creat three levels decision tree based on the provide dataset 
def Decision_Tree(Data,Decision_Attribute,Postive_Decision,Negative_Decision):
    
    Decison_Tree = {}
    # 1. Get Root Node
    # 1.1 Calcaulate Gains for each attribute
    # 1.2 Dictionrary contains { attribute : gain }
    Root_Gains = {}
    
    # 1.3 Get all the arritbutes and exclude the attributes won't locate in decision tree.
    Columns_Names = Data.columns
    Attributes_Columns = Columns_Names.drop(['HomeTeam','AwayTeam','FTR'],1)
    
    # 1.4 Loop the attributes to calculate gain for each one of them
    for attribute in Attributes_Columns:
        Root_Gains[attribute] = Calculate_Gain_Mean(attribute,Data,Decision_Attribute,Postive_Decision,Negative_Decision)
    
    # 1.5 Get max attribute with maximum gain to be the root node
    Root = max(Root_Gains.items(), key=operator.itemgetter(1))
    Decison_Tree['Root'] = Root
    
    # 2. Get Branches for the Root Node
    # 2.1 Exclude root node from the data
    Attributes_Columns = Attributes_Columns.drop(Root[0],1)
    
    # 2.2 get gain for Branch one Data
    Data_Branch1 = Data[(Data[(Root[0])] > Root[1][1])]
    Branch1_Max = Attributes_Gains(Data_Branch1,Attributes_Columns,Decision_Attribute,Postive_Decision,Negative_Decision)
    
    # 2.3 get gain for Branch two Data
    Data_Branch2 = Data[(Data[(Root[0])] <= Root[1][1])]
    Branch2_Max = Attributes_Gains(Data_Branch2,Attributes_Columns,Decision_Attribute,Postive_Decision,Negative_Decision)
    
    Decison_Tree['LeftBranch'] = Branch1_Max
    Decison_Tree['RightBranch'] = Branch2_Max
    
    # 3. calculate the third level branches for the tree
    Attributes_Columns = Attributes_Columns.drop([Branch1_Max[0],Branch2_Max[0]],1)
    Level3_LeftData = []
    Level3_RightData = []
    Level3_LeftNodes = []
    Level3_RightNodes = []
    
    Level3_LeftData.append(Data_Branch1[(Data_Branch1[(Branch1_Max[0])] >= Branch1_Max[1][2])])
    Level3_LeftData.append(Data_Branch1[(Data_Branch1[(Branch1_Max[0])] >= Branch1_Max[1][1]) & (Data_Branch1[Branch1_Max[0]] < Branch1_Max[1][2])])
    Level3_LeftData.append(Data_Branch1[(Data_Branch1[(Branch1_Max[0])] < Branch1_Max[1][1])])
    
    # 3.1 Calculate the left branches
    for Dataframe in Level3_LeftData:
        Level3_LeftNodes.append(Attributes_Gains(Dataframe,Attributes_Columns,Decision_Attribute,Postive_Decision,Negative_Decision))
        
    Level3_RightData.append(Data_Branch2[(Data_Branch2[(Branch2_Max[0])] >= Branch2_Max[1][2])])
    Level3_RightData.append(Data_Branch2[(Data_Branch2[(Branch2_Max[0])] >= Branch2_Max[1][1]) & (Data_Branch2[Branch2_Max[0]] < Branch2_Max[1][2])])
    Level3_RightData.append(Data_Branch2[(Data_Branch2[(Branch2_Max[0])] < Branch2_Max[1][1])])
    
    # 3.2 Calculate the Right branches
    for Dataframe in Level3_RightData:
        Level3_RightNodes.append(Attributes_Gains(Dataframe,Attributes_Columns,Decision_Attribute,Postive_Decision,Negative_Decision))
    
    Decison_Tree['LeftBranchNodes'] = Level3_LeftNodes
    Decison_Tree['RightBranchNodes'] = Level3_RightNodes
    
    # 4. Check the majority of H and A to create the leafes nodes
    Leafs_LeftData = []
    Leafs_RightData = []
    index = 0
    for Dataframe in Level3_LeftData:
        
        NodeInfo = Level3_LeftNodes[index]
        
        DataLeft = Dataframe[(Dataframe[(NodeInfo[0])] >= NodeInfo[1][2])]
        Leafs_LeftData.append(Get_Majority(DataLeft,Decision_Attribute))
        
        DataMid = Dataframe[(Dataframe[(NodeInfo[0])] >= NodeInfo[1][1]) & (Dataframe[NodeInfo[0]] < NodeInfo[1][2])]
        Leafs_LeftData.append(Get_Majority(DataMid,Decision_Attribute))
        
        DataRight = Dataframe[(Dataframe[(NodeInfo[0])] < NodeInfo[1][1])]
        Leafs_LeftData.append(Get_Majority(DataRight,Decision_Attribute))
        index +=1;
    
    index = 0    
    for Dataframe in Level3_RightData:
        
        NodeInfo = Level3_RightNodes[index]
        
        DataLeft = Dataframe[(Dataframe[(NodeInfo[0])] >= NodeInfo[1][2])]
        Leafs_RightData.append(Get_Majority(DataLeft,Decision_Attribute))
        
        DataMid = Dataframe[(Dataframe[(NodeInfo[0])] >= NodeInfo[1][1]) & (Dataframe[NodeInfo[0]] < NodeInfo[1][2])]
        Leafs_RightData.append(Get_Majority(DataMid,Decision_Attribute))
        
        DataRight = Dataframe[(Dataframe[(NodeInfo[0])] < NodeInfo[1][1])]
        Leafs_RightData.append(Get_Majority(DataRight,Decision_Attribute))
        index +=1;
    
    Decison_Tree['LeftLeafs'] = Leafs_LeftData
    Decison_Tree['RightLeafs'] = Leafs_RightData
    
    return Decison_Tree;

# Plot Decision Tree
def Plot_Tree(Tree):
    
    print('\t'*8+Tree['Root'][0])
    print('\t'*4+str(Tree['LeftBranch'][0])+'\t'*3+'\t'*5+str(Tree['RightBranch'][0])+'\t'*3+'\n')
    print('\t'*2+str(Tree['LeftBranchNodes'][0][0])+'\t'*2+str(Tree['LeftBranchNodes'][1][0])+'\t'*2+str(Tree['LeftBranchNodes'][2][0])+'\t'*4+str(Tree['RightBranchNodes'][0][0])+'\t'*2+str(Tree['RightBranchNodes'][1][0])+'\t'*2+str(Tree['RightBranchNodes'][2][0])+'\n')
    print('\t'*1+str(Tree['LeftLeafs'][0:3])+'\t'*1+str(Tree['LeftLeafs'][3:6])+'\t'*1+str(Tree['LeftLeafs'][6:9])+'\t'*3+str(Tree['RightLeafs'][0:3])+'\t'*1+str(Tree['RightLeafs'][3:6])+'\t'*1+str(Tree['RightLeafs'][6:9])+'\n')
    
# function to test decision tree on the test data    
def Test_Decision_Tree(Tree,TestData):
    
    Decision_Values = []
    ConfusionMatrix = np.zeros((2,2))
    
    for Index,Record in TestData.iterrows():
        
        Real_Decision = Record['FTR'] 
        Branch = ''
        SecondBranch = ''
        LeafsBranch = ''
        Leafs_Index = 0
        if Record[Tree['Root'][0]] > Tree['Root'][1][1]:
            Branch = Tree['LeftBranch']
            SecondBranch = Tree['LeftBranchNodes']
            LeafsBranch = Tree['LeftLeafs']
        else:
            Branch = Tree['RightBranch']
            SecondBranch = Tree['RightBranchNodes']
            LeafsBranch = Tree['RightLeafs']
            
        # Level 2
        if Record[Branch[0]] >= Branch[1][2]:
            SecondBranch = SecondBranch[0]
            Leafs_Index = 0
        
        elif Record[Branch[0]] >= Branch[1][1] and Record[Branch[0]] < Branch[1][2]:
            SecondBranch = SecondBranch[1]
            Leafs_Index = 3
        else:
            SecondBranch = SecondBranch[2]
            Leafs_Index = 6
            
        # Level 3
        if Record[SecondBranch[0]] >= SecondBranch[1][2]:
            Decision_Value = LeafsBranch[Leafs_Index:(Leafs_Index+3)][0]
            Decision_Values.append(Decision_Value)
            
        elif Record[SecondBranch[0]] >= SecondBranch[1][1] and Record[SecondBranch[0]] < SecondBranch[1][2]:
            Decision_Value = LeafsBranch[Leafs_Index:(Leafs_Index+3)][1]
            Decision_Values.append(Decision_Value)
            
        else:
            Decision_Value = LeafsBranch[Leafs_Index:(Leafs_Index+3)][2]
            Decision_Values.append(Decision_Value)
        
        # Confusion Matrix
        if Real_Decision == 'H' and Decision_Value == 'H':
            ConfusionMatrix[0][0] += 1
        elif Real_Decision == 'H' and Decision_Value == 'A':
            ConfusionMatrix[0][1] += 1
        elif Real_Decision == 'A' and Decision_Value == 'H':
            ConfusionMatrix[1][0] += 1
        else:
            ConfusionMatrix[1][1] += 1
            
    print("*** Confusion Matrix *** \n",ConfusionMatrix)
    return Decision_Values

# ########################### #
# ****  Global Variables **** #
# ########################### #

# **** Training data info ****
# Training CSV File Path
Training_FilePath = "C:/Users/abdelkarima/data science  diploma/Introduction to machine learning/Labs/Assignment 3/Training_Data.csv"
# Descision attribute name
Decision_Attribute = 'FTR'
# First Decision value
Postive_Decision = 'H'
# Second Decision value
Negative_Decision = 'A'

# **** Testing data info ****
# Testing CSV File Path
Testing_FilePath = "C:/Users/abdelkarima/data science  diploma/Introduction to machine learning/Labs/Assignment 3/Liverpool.csv"

# ########################### #
# ********** Logic ********** #
# ########################### #

# 1. Read Csv Data
CSV_Data = Read_CSVData(Training_FilePath)
CSV_Test = Read_CSVData(Testing_FilePath)

# 2. Creat Decision Tree
Tree = Decision_Tree(CSV_Data,Decision_Attribute,Postive_Decision,Negative_Decision)
Plot_Tree(Tree)

# 3. Test Decision Tree
Decision_Val = Test_Decision_Tree(Tree,CSV_Test)

# 4. Get the real values for FTR column to comapre it with the reultant value.
Real_DecisionVal = CSV_Test[Decision_Attribute]

#5. Calculate Accuracy
Accuracy_Score = round((accuracy_score(Real_DecisionVal.tolist(),Decision_Val)*100),1)
print("*** Accuracy = {0} %".format(Accuracy_Score))

print("Decision Tree Detials Values: \n",Tree)