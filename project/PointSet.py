from typing import List, Tuple

from enum import Enum
import numpy as np

class FeaturesTypes(Enum):
    """Enumerate possible features types"""
    BOOLEAN=0
    CLASSES=1
    REAL=2

class PointSet:
    """A class representing set of training points.

    Attributes
    ----------
        types : List[FeaturesTypes]
            Each element of this list is the type of one of the
            features of each point
        features : np.array[float]
            2D array containing the features of the points. Each line
            corresponds to a point, each column to a feature.
        labels : np.array[bool]
            1D array containing the labels of the points.
    """
    def __init__(self, features: List[List[float]], labels: List[bool], types: List[FeaturesTypes], min_split_points: int = 1):
        """
        Parameters
        ----------
        features : List[List[float]]
            The features of the points. Each sublist contained in the
            list represents a point, each of its elements is a feature
            of the point. All the sublists should have the same size as
            the `types` parameter, and the list itself should have the
            same size as the `labels` parameter.
        labels : List[bool]
            The labels of the points.
        types : List[FeaturesTypes]
            The types of the features of the points.
        """
        self.numberOfPoints = len(labels)
        self.types = types
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.min_split_points = min_split_points
    
    def get_gini(self) -> float:
        """Computes the Gini score of the set of points

        Returns
        -------
        float
            The Gini score of the set of points
        """

        if (self.labels.size == 0):
            return 0
        inverse_gini = (sum([1 for i in self.labels if i == 1])/self.labels.size)**2 + (sum([1 for i in self.labels if i == 0])/self.labels.size)**2
        return 1-inverse_gini 
    

    def get_gini_split(self, featureSplit) :
        """Computes the Gini split score of the set of points in case the attribute is a boolean

        Returns
        -------
        float
            The Gini split score of the set of points
        """

        ## We split the features and lables acording to the feature index passed as a parameter
        feature1 = np.array([np.array(row) for row in self.features if row[featureSplit] == 0])
        feature2 = np.array([np.array(row) for row in self.features if row[featureSplit] == 1])
        label1 = np.array([self.labels[rowIndex] for rowIndex in range(self.features.shape[0]) if self.features[rowIndex][featureSplit] == 0])
        label2 = np.array([self.labels[rowIndex] for rowIndex in range(self.features.shape[0]) if self.features[rowIndex][featureSplit] == 1])
        child1 = PointSet(feature1, label1, self.types)
        child2 = PointSet(feature2, label2, self.types)
        nbr_el = self.features.size
        ratio_child1 = feature1.size / nbr_el
        ratio_child2 = feature2.size / nbr_el
        if (label1.size >= self.min_split_points and label2.size >= self.min_split_points):
            return child1.get_gini()*ratio_child1 + child2.get_gini()*ratio_child2
        else: ## -1 is an impossible value, no split found on this attribute. In the for loop of the get_best_gain function, this attribute will be skipped
            return -1

    def get_gini_split_class(self, featureSplit):
        """ To keep the code clear, I made a second function. We could use the function above
        but I find it easier to understand this way. Computes the Gini split score of the set of points
        whene the feature we wish to perform a split on are class values

        Deprecated, see get_gini_split_efficient() bellow. Performing the same task in O(nlog(n)) complexity

        Returns
        -------
        float
            The Gini split score of the set of points
        float
            The class value of the feature providing the best split gain
        """
        class_values = np.unique(self.features[:, featureSplit])
        return_array = []
        for class_value in class_values: ## We do the same as above but on every possibility of splitting the features
            feature1 = np.array([np.array(row) for row in self.features if row[featureSplit] == class_value])
            feature2 = np.array([np.array(row) for row in self.features if row[featureSplit] != class_value])
            label1 = np.array([self.labels[rowIndex] for rowIndex in range(self.features.shape[0]) if self.features[rowIndex][featureSplit] == class_value])
            label2 = np.array([self.labels[rowIndex] for rowIndex in range(self.features.shape[0]) if self.features[rowIndex][featureSplit] != class_value])
            child1 = PointSet(feature1, label1, self.types)
            child2 = PointSet(feature2, label2, self.types)
            nbr_el = self.features.size
            ratio_child1 = feature1.size / nbr_el
            ratio_child2 = feature2.size / nbr_el
            if (label1.size >= self.min_split_points and label2.size >= self.min_split_points):
                return_array.append((child1.get_gini()*ratio_child1 + child2.get_gini()*ratio_child2, class_value))
        if (not return_array): ## -1 is an impossible value, no split found on this attribute. In the for loop of the get_best_gain function, this attribute will be skipped
            return -1, None
        return min(return_array, key = lambda item: item[0])
    

    def get_gini_split_real(self, featureSplit):
        """ Same here, to keep the code clear I made a third function. Computes the Gini split score of the set of points
        whene the feature we wish to perform a split on are real values

        Deprecated, see get_gini_split_efficient() bellow. Performing the same task in O(nlog(n)) complexity

        Returns
        -------
        float
            The Gini split score of the set of points
        float
            The class value of the feature providing the best split gain
        """
        real_values = np.unique(self.features[:, featureSplit])
        return_array = []
        for real_value_index in range(real_values.size): ## We do the same as above but on every possibility of splitting the features
            feature1 = np.array([np.array(row) for row in self.features if row[featureSplit] < real_values[real_value_index]])
            feature2 = np.array([np.array(row) for row in self.features if row[featureSplit] >= real_values[real_value_index]])
            label1 = np.array([self.labels[rowIndex] for rowIndex in range(self.features.shape[0]) if self.features[rowIndex][featureSplit] < real_values[real_value_index]])
            label2 = np.array([self.labels[rowIndex] for rowIndex in range(self.features.shape[0]) if self.features[rowIndex][featureSplit] >= real_values[real_value_index]])
            child1 = PointSet(feature1, label1, self.types)
            child2 = PointSet(feature2, label2, self.types)
            nbr_el = self.features.size
            ratio_child1 = feature1.size / nbr_el
            ratio_child2 = feature2.size / nbr_el
            potential_threshold = (real_values[real_value_index] + real_values[real_value_index-1]) / 2 
            if (label1.size >= self.min_split_points and label2.size >= self.min_split_points):
                return_array.append((child1.get_gini()*ratio_child1 + child2.get_gini()*ratio_child2, potential_threshold))
        if (not return_array): ## -1 is an impossible value, no split found on this attribute. In the for loop of the get_best_gain function, this attribute will be skipped
            return -1, None
        return min(return_array, key = lambda item: item[0])
    

    def get_gini_split_efficient (self, featureSplit, type = 'r'):
        """ This function is getting the best gini split for class and real types.
        Parameters
        -------
        featureSplit: int
            The attribute to compute the gini splits on
        type: char
            The type of the considered attribute, 'r' for real and 'c' for class

        Returns
        -------
        tuple containing 
            The Gini split score of the set of points
            The value of the threshold / the class value providing the best split for this attribute
        """

        ## Instead of checking for each possible value of the attribute if each value of the column is respecting a condition (== for class and < for real)
        ## the idea is to only update informations necessary to compute the gini splits while traversing the array only once. 
        ## The sorting of the column (np.argsort, and np.unique) are in O(nlog(n)). The traverse is in O(n), only checking a condition and making computaion for each step.
        ## This function is performing a lot better than the 2 previous functions and compute the difficults exmepls of the project in under a second.
        split_column = self.features[:, featureSplit]       # getting the features corresponding to the column specidied in parameter
        sorted_indices = np.argsort(split_column)           # sorting the labels and column with respect to the sort order of the splitting column
        sorted_column = split_column[sorted_indices] 
        sorted_labels = self.labels[sorted_indices]
        unique_elements, counts = np.unique(sorted_column, return_counts = True)    # getting the possible values of the column as well as the number of time they appear

        ## Variable initialization 
        label_index = 0                 # index used to travers the label array
        ## let us note that we do not need all of those attribute because we can easly find most of them with a simple substraction
        ## for instance, nbr_true_left == nbr_el_left - nbr_false_left. For readability I prefer to keep them and it is not affecting the performance a lot.
        nbr_true_left, nbr_false_left = 0, 0        # counter of the number of true and false in the left subtree
        nbr_true_right = np.sum(sorted_labels)      # counter of the number of true and false in the right subtree
        nbr_false_right = np.sum(~sorted_labels)    
        nbr_els = sorted_labels.size                #total number of elements in the arrrays split column and labels
        nbr_el_left = 0                             #number of elements in the left subtree (nbr_true_left + nbr_false_left)
        nbr_el_right = nbr_els                      #number of elements in the right subtree 
        result_array = []                           #used to store the result while the function is running. Here again we could just keep the best value


        for index in range(unique_elements.size):
            gap = counts[index]         # number of elements of with the same value in the splitting column (in a row because the list is sorted)
            if (type == 'c'):           # for class we want to test each value against the rest of the possible values. We then need to reset the counting varaibles
                nbr_el_left = 0         
                nbr_el_right = nbr_els
                nbr_true_left, nbr_false_left = 0, 0
                nbr_true_right = np.sum(sorted_labels)
                nbr_false_right = np.sum(~sorted_labels)

            nbr_el_left += gap  
            nbr_el_right -= gap

            for gap_index in range (gap):       # we check the value of the labels corresponding to the features having the value unique_elements[index]
                if (sorted_labels[label_index + gap_index]):
                    nbr_true_left += 1
                    nbr_true_right -= 1
                else:
                    nbr_false_left += 1
                    nbr_false_right -= 1

            label_index += gap                 

            if (nbr_el_left >= self.min_split_points and nbr_el_right >= self.min_split_points):        #checking if we keep a minimal number of elements in each subtrees
                
                left_gini = 1 - ((nbr_true_left / nbr_el_left) **2 + (nbr_false_left / nbr_el_left) **2)    #computing gini splits
                right_gini = 1 - ((nbr_true_right / nbr_el_right) **2 + (nbr_false_right / nbr_el_right) **2)
                gini_split = ((nbr_el_left/nbr_els) * left_gini) + ((nbr_el_right/nbr_els) * right_gini)


                if (type == 'r'):        # we get the gini_split and the value of the associated threshold
                    result_array.append((gini_split, (unique_elements[index] + unique_elements[index + 1]) / 2))


                elif (type == 'c'):     #we get the gini_split as well as the value associated
                    result_array.append((gini_split, unique_elements[index]))

        if (not result_array): # -1 is an impossible value, no split found on this attribute. In the for loop of the get_best_gain function, this attribute will be skipped
            return -1, None
        
        return min(result_array, key = lambda item: item[0])        # returning the best split found as well as the best threshold / cass value associated


    def get_best_threshold(self):
        return self.threshold       # this value is defined in the get_best_gain bellow



    def get_best_gain(self) -> Tuple[int, float]:
        """Compute the feature along which splitting provides the best gain

        Returns
        -------
        int
            The ID of the feature along which splitting the set provides the
            best Gini gain.
        float
            The best Gini gain achievable by splitting this set along one of
            its features.
        float
            Only if the feature is a class, the class value providing the best split
        """

        ## Note the the default values are set to -1 which are not possible values for those variables. 
        ## This allows us to quickly identify if a split was found or not.
        min = -1
        minIndex = -1
        value = -1
        ## We compute the gini gain of all the features and return the feature to split on to get the best gini after the split
        for featureIndex in range(len(self.types)):
            if (self.types[featureIndex] == FeaturesTypes.BOOLEAN):
                gini_split = self.get_gini_split(featureIndex)
            elif (self.types[featureIndex] == FeaturesTypes.CLASSES):
                gini_split, value = self.get_gini_split_efficient(featureIndex, 'c')
            elif (self.types[featureIndex] == FeaturesTypes.REAL):
                gini_split, value = self.get_gini_split_efficient(featureIndex) 
            else:
                raise Exception ("Type non reconnu")
            if (gini_split != -1 and (gini_split < min or min == -1) ) :  #if gini_split == -1 then the condition on the number of points per nodes is violated for each split on this attribute, this attribute is skipped
                min = gini_split
                minIndex = featureIndex
                min_value = value

        if (minIndex == -1): ## No split found on any of the attributes
            return None
        if (self.types[minIndex] == FeaturesTypes.CLASSES or self.types[minIndex] == FeaturesTypes.REAL):
            self.threshold = min_value
        else:
            self.threshold = None
        return minIndex, self.get_gini() - min

    def del_training_point(self, features: List[float], label: bool):
        del_index = np.where((self.features==features).all(axis=1))
        if del_index[0].size > 0:
            for index in del_index[0]:
                if (self.labels[index] == label):
                    self.features = np.delete(self.features, index, axis = 0)
                    self.labels = np.delete(self.labels,index)
                    self.numberOfPoints -= 1
                    return


    def add_training_point(self, features: List[float], label: bool):
        self.features = np.append(self.features, [features], axis = 0 )
        self.labels = np.append(self.labels, label)
        self.numberOfPoints += 1 