import numpy as np
import pandas as pd

EMPTY = ''
NOTHING = -1
class DTreeNode:
    def __init__(self):
        self.left = None
        self.right = None
        self.feature = EMPTY
        self.threshold = NOTHING
        self.prediction = NOTHING
        
class DecisionTreeClassifier:
    
    def __init__(self, min_data_leaf=10):
        self.root = DTreeNode()
        self.min_data_leaf = min_data_leaf
    
    def fit(self, X: pd.DataFrame, Y: pd.Series):
        """
        Fits the random forest model to the provided data.

        Parameters:
        X (pd.DataFrame): The input features for the model.
        Y (pd.Series): The target variable for the model.
        """
        target_col = Y.name
        features = list(X.columns)
        self.generateTree(X, features, target_col, self.root)
    
    def predict(self, X: pd.DataFrame):
        """
        Predict the class labels for the given input data.

        Parameters:
        X (pd.DataFrame): The input data for which predictions are to be made. Each row represents a sample.

        Returns:
        list: A list of predicted class labels for each sample in the input data.
        """
        pred_y = []
        for row in X:
            pred_y.append(self.traverseTree(row, self.root))
        return pred_y
    
    def computeEntropy(self, X: pd.DataFrame, target_col: str):
        """
        Compute the entropy of a given target column in a DataFrame.
        It is calculated using the formula:
            entropy = -sum(p * log2(p))
        where p is the proportion of each unique value in the target column.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the data.
        target_col (str): The name of the target column for which to compute the entropy.
        
        Returns:
        float: The computed entropy value.
        """
        proportions = X[target_col].value_counts() / X[target_col].count()
        weighted_proportions = proportions * np.log2(proportions)
        return -1 * np.sum(weighted_proportions)
        
    def computeGini(self, X: pd.DataFrame, target_col: str):
        """
        Compute the Gini index for a given dataset.

        Parameters:
        X (pd.DataFrame): The dataset containing the features and target column.
        target_col (str): The name of the target column in the dataset.

        Returns:
        float: The Gini impurity of the target column in the dataset.
        """
        proportions = X[target_col].value_counts() / X[target_col].count()
        return 1 - np.sum(proportions * proportions)
    
    def getOptimalSplit(self, X: pd.DataFrame, feature: list, target_col: str):
        """
        Finds the optimal split for a given feature in the dataset to maximize information gain.

        Parameters:
        X (pd.DataFrame): The input dataframe containing the features and target column.
        feature (list): The feature for which the optimal split is to be found.
        target_col (str): The name of the target column in the dataframe.

        Returns:
        tuple: A tuple containing the best threshold value, the best information gain,
               the left split dataframe, and the right split dataframe.
        """
        best_threshold = NOTHING
        best_information_gain = NOTHING
        best_left_split = NOTHING
        best_right_split = NOTHING
        num_root = X[feature].count()
        if hasattr(X[feature], "cat"):
            for t in X[feature].unique():
                left_split = X[X[feature] == feature]
                right_split = X[X[feature] != feature]
                num_left = left_split[feature].count()
                num_right = left_split[feature].count()
                information_gain = self.computeEntropy(X, target_col) - (num_left/num_root) * self.computeEntropy(left_split, target_col) - (num_right/num_root) * self.computeEntropy(right_split, target_col)
                if information_gain > best_information_gain:
                    best_threshold = t
                    best_information_gain = information_gain
                    best_left_split = left_split
                    best_right_split = right_split
        else:
            for t in X[feature].unique():
                left_split = X[X[feature] <= feature]
                right_split = X[X[feature] > feature]
                num_left = left_split[feature].count()
                num_right = left_split[feature].count()
                information_gain = self.computeEntropy(X, target_col) - (num_left/num_root) * self.computeEntropy(left_split, target_col) - (num_right/num_root) * self.computeEntropy(right_split, target_col)
                if information_gain > best_information_gain:
                    best_threshold = t
                    best_information_gain = information_gain
                    best_left_split = left_split
                    best_right_split = right_split
        return (best_threshold, best_information_gain, best_left_split, best_right_split)
    
    # Potential TODO: could improve generation by splitting
    # multiple times in same feature, like for
    # categorical values with more than 2 categories
    # or continuous values
    def generateTree(self, X: pd.DataFrame, features: list, target_col: str, node: DTreeNode):
        """
        Generates a decision tree by finding the optimal feature and threshold to split the data.
        
        Parameters:
        X (pd.DataFrame): The input data.
        features (list): List of features to consider for splitting.
        target_col (str): The name of the target column.
        node (DTreeNode): The current node in the decision tree.
        """
        if len(features) == 0 or len(X[target_col].unique()) == 1 or len(X[target_col]) <= self.min_data_leaf:
            node.prediction = X[target_col].mode()[0]
            return
        
        best_gain = NOTHING
        best_threshold = NOTHING
        best_feature = EMPTY
        best_left_split = NOTHING
        best_right_split = NOTHING
        
        for feature in features:
            (threshold, gain, left_split, right_split) = self.getOptimalSplit(X, feature, target_col)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
                best_left_split = left_split
                best_right_split = right_split
        
        node.left = DTreeNode()
        node.right = DTreeNode()
        node.feature = best_feature
        node.threshold = best_threshold
        features.remove(best_feature)
        self.generateTree(best_left_split, features, target_col, node.left)
        self.generateTree(best_right_split, features, target_col, node.right)
    
    def traverseTree(self, X: pd.Series, node: DTreeNode):
        """
        Traverse the decision tree to make a prediction for a given sample.

        Parameters:
        X (pd.Series): A single sample with features as a pandas Series.
        node (DTreeNode): The current node in the decision tree.

        Returns:
        The prediction value for the sample or a placeholder if the node is None.
        """
        if node == None:
            return NOTHING # TODO: how to handle this case?
        if node.left == None and node.right == None:
            return node.prediction
        if hasattr(X[node.feature], "cat"):
            if(X[node.feature] == node.threshold):
                return self.traverseTree(X, node.left)
            else:
                return self.traverseTree(X, node.right)
        else:
            if(X[node.feature] <= node.threshold):
                return self.traverseTree(X, node.left)
            else:
                return self.traverseTree(X, node.right)


# TODO: finish implementing random forest classifier    
class RandomForestClassifier:
    def __init__(self, num_trees=10):
        self.models = []
        self.num_trees = num_trees
    
    def fit(self, X, Y):
        # create num_trees models, by sampling rows/columns randomly
        pass
    
    def predict(self, X):
        # predict output on each of the models, then take the majority vote
        pass
    
    def bootstrap(self, X):
        # return a random sample of the data, after removing some of the columns
        pass
    
    def aggregate(self, predictions):
        # take the predictions, then take the majority vote and output that
        pass