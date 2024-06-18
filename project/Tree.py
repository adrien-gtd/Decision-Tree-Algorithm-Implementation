from typing import List
import numpy as np
from PointSet import PointSet, FeaturesTypes

class Tree:
    """A decision Tree

    Attributes
    ----------
        points : PointSet
            The training points of the tree
    """
    def __init__(self,
                 features: List[List[float]],
                 labels: List[bool],
                 types: List[FeaturesTypes],
                 h: int = 1,
                 min_split_points: int = 1, 
                 beta: float = 0):
        """
        Parameters
        ----------
            labels : List[bool]
                The labels of the training points.
            features : List[List[float]]
                The features of the training points. Each sublist
                represents a single point. The sublists should have
                the same length as the `types` parameter, while the
                list itself should have the same length as the
                `labels` parameter.
            types : List[FeaturesTypes]
                The types of the features.
        """
        ## This tree is built recursively by splitting the data on the value of a feature providing
        ## the best split gini gains. 
        self.beta = beta
        self.min_split_points = min_split_points
        self.h = h
        self.points = PointSet(features, labels, types, min_split_points)

        self.buildTree()
        

    def decide(self, features: List[float]) -> bool:
        """Give the guessed label of the tree to an unlabeled point

        Parameters
        ----------
            features : List[float]
                The features of the unlabeled point.

        Returns
        -------
            bool
                The label of the unlabeled point,
                guessed by the Tree
        """
        ## We traverse the tree built previously choosing the next subtree with the same condition as when we built it
        currTree = self
        while (currTree.isleaf == False) :
            if (self.points.types[currTree.splitAttribute] == FeaturesTypes.REAL and 
                features[currTree.splitAttribute] < currTree.splitCondition) :
                currTree = currTree.child1
            elif (features[currTree.splitAttribute] == currTree.splitCondition) :
                currTree = currTree.child1
            else:
                currTree = currTree.child2

        ## Returning the tag which is the max between class1 values and class0 values of the node
        return currTree.tag
    
    def add_training_point(self, features: List[float], label: bool): 
        currTree = self
        currTree.points.add_training_point(features, label)
        currTree.updateCount += 1 
        isChanged = currTree.check_update()
        while (currTree.isleaf == False and not isChanged):
            if (self.points.types[currTree.splitAttribute] == FeaturesTypes.REAL and 
                features[currTree.splitAttribute] < currTree.splitCondition) :
                currTree = currTree.child1
            elif (features[currTree.splitAttribute] == currTree.splitCondition) :
                currTree = currTree.child1
            else:
                currTree = currTree.child2
            currTree.points.add_training_point(features, label)
            currTree.updateCount += 1 
            isChanged = currTree.check_update()
            


    def del_training_point(self, features: List[float], label: bool):
        currTree = self
        currTree.points.del_training_point(features, label)
        currTree.updateCount += 1 
        isChanged = currTree.check_update()
        while (currTree.isleaf == False and not isChanged):
            if (self.points.types[currTree.splitAttribute] == FeaturesTypes.REAL and 
                features[currTree.splitAttribute] < currTree.splitCondition) :
                currTree = currTree.child1
            elif (features[currTree.splitAttribute] == currTree.splitCondition) :
                currTree = currTree.child1
            else:
                currTree = currTree.child2
            currTree.points.del_training_point(features, label)
            currTree.updateCount += 1 
            isChanged = currTree.check_update()
        

    def check_update(self):
        if self.points.numberOfPoints * self.beta <= self.updateCount:
            self.buildTree()
            return True
        return False

    def buildTree (self):
        self.updateCount = 0
        self.tag = False            ## tag is used for the prediction, it will be the label predicted if we arrive to this leaf
        self.isleaf = True          ## Usefull to check if the current node is a leaf or not
        class1 = sum([1 for bool in self.points.labels if bool])    ## Number of class1 labels 
        class0 = sum([1 for bool in self.points.labels if not bool]) ## Number of class0 labels 
        if (class1 > class0):     
            self.tag = True
        if (self.h > 0 and class1 < self.points.labels.size and class0 < self.points.labels.size): 
            ## The stopping condition of the construction are:
            ## - The lables are only Trues or Falses
            ## - The max height of the tree is reached (h==0)
            gini_split_info = self.points.get_best_gain()
            if (gini_split_info == None):
                return
            
            self.isleaf = False
            
            self.splitAttribute = gini_split_info[0]
            
            ## Getting the value of the spliiting condition
            if (self.points.types[self.splitAttribute] == FeaturesTypes.CLASSES or  self.points.types[self.splitAttribute] == FeaturesTypes.REAL):
                self.splitCondition = self.points.get_best_threshold()
            if (self.points.types[self.splitAttribute] == FeaturesTypes.BOOLEAN):
                self.splitCondition = 0

            if (self.points.types[self.splitAttribute] == FeaturesTypes.REAL):
                feature1 = np.array([np.array(row) for row in self.points.features if row[self.splitAttribute] < self.splitCondition])
                feature2 = np.array([np.array(row) for row in self.points.features if row[self.splitAttribute] >= self.splitCondition])
                label1 = np.array([self.points.labels[rowIndex] for rowIndex in range(self.points.features.shape[0]) if self.points.features[rowIndex][self.splitAttribute] < self.splitCondition])
                label2 = np.array([self.points.labels[rowIndex] for rowIndex in range(self.points.features.shape[0]) if self.points.features[rowIndex][self.splitAttribute] >= self.splitCondition])
            else :    
                ## Splitting the points in 2 points sets on the splitting condition found above 
                feature1 = np.array([np.array(row) for row in self.points.features if row[self.splitAttribute] == self.splitCondition])
                feature2 = np.array([np.array(row) for row in self.points.features if row[self.splitAttribute] != self.splitCondition])
                label1 = np.array([self.points.labels[rowIndex] for rowIndex in range(self.points.features.shape[0]) if self.points.features[rowIndex][self.splitAttribute] == self.splitCondition])
                label2 = np.array([self.points.labels[rowIndex] for rowIndex in range(self.points.features.shape[0]) if self.points.features[rowIndex][self.splitAttribute] != self.splitCondition])

            ## Recursively building the sub trees of heigth -1
            self.child1 = Tree(feature1, label1, self.points.types, self.h-1, self.min_split_points)
            self.child2 = Tree(feature2, label2, self.points.types, self.h-1, self.min_split_points)
        
