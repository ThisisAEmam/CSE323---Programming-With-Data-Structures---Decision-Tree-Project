import numpy as np  #used in numerical operations
from sklearn.metrics import classification_report   #ONLY used in generating some evaluation metrics report

# The entropy calculation function
def entropy(y):
    N = len(y)
    s1 = (y == 1).sum()
    if s1 == 0 or N == s1:
        return 0
    p1 = float(s1) / N
    p0 = 1 - p1
    return -p0 * np.log2(p0) - p1 * np.log2(p1)

class Node:
    def __init__(self, depth= 0, max_depth= None):
        self.depth = depth
        self.max_depth = max_depth
    
    def fit(self, X, Y):
            if len(Y) == 1 or len(set(Y)) == 1:
                self.col = None
                self.split = None
                self.left = None
                self.right = None
                self.prediction = Y[0]
    
            else:
                D = X.shape[1]
                cols = range(D)
    
                max_ig = 0
                best_col = None
                best_split = None
                for col in cols:
                    ig, split = self.find_split(X, Y, col)
                    if ig > max_ig:
                        max_ig = ig
                        best_col = col
                        best_split = split
    
                if max_ig == 0:
                    self.col = None
                    self.split = None
                    self.left = None
                    self.right = None
                    self.prediction = np.round(Y.mean())
                else:
                    self.col = best_col
                    self.split = best_split
    
                    if self.depth == self.max_depth:
                        self.left = None
                        self.right = None
                        self.prediction = [
                            np.round(Y[X[:,best_col] < self.split].mean()),
                            np.round(Y[X[:,best_col] >= self.split].mean()),
                        ]
                    else:
                        left_idx = (X[:,best_col] < best_split)
                        Xleft = X[left_idx]
                        Yleft = Y[left_idx]
                        self.left = Node(self.depth + 1, self.max_depth)
                        self.left.fit(Xleft, Yleft)
    
                        right_idx = (X[:,best_col] >= best_split)
                        Xright = X[right_idx]
                        Yright = Y[right_idx]
                        self.right = Node(self.depth + 1, self.max_depth)
                        self.right.fit(Xright, Yright)
    
    def find_split(self, X, Y, col):
        x_values = X[:, col]
        sort_idx = np.argsort(x_values)
        x_values = x_values[sort_idx]
        y_values = Y[sort_idx]

        boundaries = np.nonzero(y_values[:-1] != y_values[1:])[0]
        best_split = None
        max_ig = 0
        for b in boundaries:
            split = (x_values[b] + x_values[b+1]) / 2
            ig = self.information_gain(x_values, y_values, split)
            if ig > max_ig:
                max_ig = ig
                best_split = split
        return max_ig, best_split

    def information_gain(self, x, y, split):
        y0 = y[x < split]
        y1 = y[x >= split]
        N = len(y)
        y0len = len(y0)
        if y0len == 0 or y0len == N:
            return 0
        p0 = float(len(y0)) / N
        p1 = 1 - p0
        return entropy(y) - p0 * entropy(y0) - p1 * entropy(y1)

    def predict_one(self, x):
        if self.col is not None and self.split is not None:
            feature = x[self.col]
            if feature < self.split:
                if self.left:
                    p = self.left.predict_one(x)
                else:
                    p = self.prediction[0]
            else:
                if self.right:
                    p = self.right.predict_one(x)
                else:
                    p = self.prediction[1]
        else:
            p = self.prediction
        return p

    def predict(self, X):
        N = len(X)
        P = np.zeros(N)
        for i in range(N):
            P[i] = self.predict_one(X[i])
        return P


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth   # specifing max depth of the tree IF NEEDED 
    
    # fits the model to the training data (training the model)
    def fit(self, X, Y):
        self.root = Node(max_depth=self.max_depth)
        self.root.fit(X, Y)
    
    # predicting the label (target) class
    def predict(self, X):
        return self.root.predict(X)
    
    # calculating the accuracy of the model
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)
    
    # showing the evaluation metics report
    def getCR(self, X, Y):
        P = self.predict(X)
        return classification_report(Y, P)