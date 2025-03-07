import csv
import numpy as np
import random
from collections import Counter
import time
import matplotlib.pyplot as plt
from itertools import combinations

def all_columns(X, rand):
    return range(X.shape[1])    


def random_sqrt_columns(X, rand):
    n_cols = X.shape[1]
    sqrt_cols = int(np.sqrt(n_cols))
    indices = rand.sample(range(0, n_cols), k=sqrt_cols)
    return np.array(indices)



class Node:

    def __init__(self, left=None, right=None, column_index=None, threshold=None, prediction=None):
        self.left = left
        self.right = right
        self.column_index = column_index
        self.threshold = threshold
        self.prediction = prediction

class Tree:

    def __init__(self, rand=None,
                 get_candidate_columns=all_columns,
                 min_samples=2):
        self.rand = rand  # for replicability
        self.get_candidate_columns = get_candidate_columns  # needed for random forests
        self.min_samples = min_samples
        self.columns = []

    def build(self, X, y):
        self.columns = []
        return TreeModel(self.build_tree_recursively(X, y), sorted(self.columns))  #Sorted to have consistent indexing for importances3

    def build_tree_recursively(self, X, y):
        if len(np.unique(y)) == 1:
            return Node(prediction=y[0])
        
        if len(y) < self.min_samples:
            return Node(prediction=Counter(y).most_common(1)[0][0])
        column, threshold = self.best_gini(X, y, self.get_candidate_columns(X, self.rand))
        self.columns.append(column)

        if len(np.unique(X[:, column])) == 1:
            #print("holy shit it happened")
            count = Counter(y)
            val, count = count.most_common()[0]
            return Node(prediction=val)

        data_left, data_right = self.partition(X, y, column, threshold)

        left_child = self.build_tree_recursively(*data_left)
        right_child = self.build_tree_recursively(*data_right)

        return Node(left_child, right_child, column, threshold=threshold) 

    def best_gini_in_column(self, X, y, column):

        sorted_X_indices = np.argsort(X[:, column])

        sorted_X = X[sorted_X_indices, column]
        sorted_y = y[sorted_X_indices]
        best_gini = 2
        best_threshold = None
        total_samples = len(sorted_X)

        left_counter = Counter()
        right_counter = Counter(sorted_y)

        if len(np.unique(sorted_X)) == 1:
            best_threshold = sorted_X[0]
            best_gini = 1 - sum((c/len(sorted_y))**2 for c in right_counter)
            return best_gini, best_threshold

        for i in range(1, total_samples):
            label = sorted_y[i - 1]
            left_counter[label] += 1
            right_counter[label] -= 1

            if sorted_X[i] == sorted_X[i - 1]:
                continue

            left_size = i
            right_size = total_samples - i
            assert left_size + right_size == len(y)

            left_gini = 1 - sum((left_counter[c] / left_size) ** 2 for c in left_counter)
            right_gini = 1 - sum((right_counter[c] / right_size) ** 2 for c in right_counter)

            gini_value = (left_gini * left_size + right_gini * right_size) / total_samples

            if gini_value < best_gini:
                best_gini = gini_value
                best_threshold = (sorted_X[i] + sorted_X[i - 1]) / 2#threshold in the middle of the unique values instead of directly on them!!! (it generalizes better)
                                                                    #Solution advised at labs :)
        return best_gini, best_threshold

    def best_gini(self, X, y, columns):
        best_gini_value = 2
        best_column = None
        best_threshold = None
        for column in columns:
            gini_value, threshold = self.best_gini_in_column(X, y, column)
            if gini_value < best_gini_value:
                best_gini_value = gini_value
                best_column = column
                best_threshold = threshold
        return best_column, best_threshold

    def partition(self, X, y, column, threshold):
        left_indices = np.where(X[:, column] <= threshold)[0]  # Get indices only
        right_indices = np.where(X[:, column] > threshold)[0]
        return ((X[left_indices], y[left_indices]), (X[right_indices], y[right_indices]))  # Return indices instead of full arrays

class TreeModel:

    def __init__(self, root_node = None, columns= None):
        self.root = root_node
        self.columns = columns

    def predict(self, X):
        predictions = [self.predict_one(self.root, x) for x in X]
        return np.array(predictions)

    def predict_one(self, node, x):
        while node.prediction is None:
            column_index = node.column_index
            threshold = node.threshold
            node = node.left if x[column_index] <= threshold else node.right
        return node.prediction
    """
        if node.prediction != None:
            return node.prediction
        else:
            if x[node.column_index] <= node.threshold:
                return self.predict_one(node.left, x)
            else:
                return self.predict_one(node.right, x)
    """


class RandomForest:

    def __init__(self, rand=None, n=50):
        self.n = n
        self.rand = rand
        self.rftree = Tree(rand=self.rand,
                           get_candidate_columns=random_sqrt_columns,
                           min_samples=2)  # initialize the tree properly

    def build(self, X, y):
        root_nodes = [0] * self.n
        OOB_indices_collection = [0] * self.n
        for i in range(self.n):
            Xb, yb, OOB_indices= self.bootstrap_dataset(X, y)
            OOB_indices_collection[i] = OOB_indices
            root_nodes[i] = self.rftree.build(Xb, yb)
        return RFModel(root_nodes, OOB_indices_collection, X, y)  # return an object that can do prediction
    
    def bootstrap_dataset(self, X, y):
        possible_indices = range(X.shape[0])
        bootstrap_indices = self.rand.choices(possible_indices, k=X.shape[0])
        possible_indices_set = set(possible_indices)
        bootstrap_indices_set = set(bootstrap_indices)
        out_of_bag_indices = list(possible_indices_set - bootstrap_indices_set)
        return X[bootstrap_indices], y[bootstrap_indices], out_of_bag_indices


class RFModel:

    def __init__(self, trees, OOB_indices_collection, X, y):
        self.trees = trees
        self.OOB_indices_collection = OOB_indices_collection
        self.X = X
        self.y = y

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        all_results = np.zeros((len(self.trees), X.shape[0]))
        for i, tree in enumerate(self.trees):
            all_results[i] = tree.predict(X)
        for u in range(0, X.shape[0]):
            count = Counter(all_results[:, u])
            predictions[u] = count.most_common()[0][0]
        return predictions

    def importance(self):

        np.random.seed(42)
        imps_matrix = np.zeros((len(self.trees), self.X.shape[1]))
        imps = np.zeros(self.X.shape[1])

        for i in range(len(self.trees)):

            if len(self.OOB_indices_collection[i]) == 0:
                continue

            X_cur = self.X[self.OOB_indices_collection[i]]
            y_cur = self.y[self.OOB_indices_collection[i]]
            predictions = self.trees[i].predict(X_cur)
            accuracy = self.prediction_accuracy(predictions, y_cur)

            #permuted_accuracy_loss_i = np.zeros(self.X.shape[1])
            X_copy = X_cur.copy()

            for p in self.trees[i].columns:
                X_copy[:, p] = np.random.permutation(X_copy[:, p])
                permuted_predictions = self.trees[i].predict(X_copy)
                permuted_accuracy = self.prediction_accuracy(permuted_predictions, y_cur)
                imps[p] += (accuracy - permuted_accuracy)/len(self.trees)
                X_copy[:, p] = X_cur[:, p]

            #imps_matrix[i] = permuted_accuracy_loss_i

        #imps = np.mean(imps_matrix, axis=0)
        return imps
    
    def importances3(self):
        
        np.random.seed(42)
        imps_matrix = np.zeros((self.X.shape[1], self.X.shape[1], self.X.shape[1]))
        #imps = np.zeros(self.X.shape[1])
            
        for i in range(len(self.trees)):
            X_cur = self.X[self.OOB_indices_collection[i]]
            y_cur = self.y[self.OOB_indices_collection[i]]
            predictions = self.trees[i].predict(X_cur)
            accuracy = self.prediction_accuracy(predictions, y_cur)

            #permuted_accuracy_loss_i = np.zeros((self.X.shape[1], self.X.shape[1], self.X.shape[1]))
            X_copy = X_cur.copy()

            for p in combinations(self.trees[i].columns, 3):
                X_copy[:, p] = np.random.permutation(X_copy[:, p])
                permuted_predictions = self.trees[i].predict(X_copy)
                permuted_accuracy = self.prediction_accuracy(permuted_predictions, y_cur)
                imps_matrix[p[0]][p[1]][p[2]] = (accuracy - permuted_accuracy)/len(self.trees)
                X_copy[:, p] = X_cur[:, p]

            #imps_matrix[i] = permuted_accuracy_loss_i

        #imps = np.mean(imps_matrix, axis=0)
        return imps_matrix

    def prediction_accuracy(self, y_pred, y_label):
        return 1 - misclassification_rate(y_pred, y_label)

def read_tab(fn, adict):
    content = list(csv.reader(open(fn, "rt"), delimiter="\t"))

    legend = content[0][1:]
    data = content[1:]

    X = np.array([d[1:] for d in data], dtype=float)
    y = np.array([adict[d[0]] for d in data])

    return legend, X, y

def report_metrics(y_pred, y_true):
    misclassification = misclassification_rate(y_pred, y_true)
    SD = np.std(y_pred)
    SE = SD/np.sqrt(len(y_pred))
    #print(f"Misclassification rate: {round(misclassification, 4)} +/- {SE}")
    return misclassification, SE
    #print(f"Accuracy: {round(misclassification, 4)}")

def hw_tree_full(learn, test):
    print("Classification Tree")
    print("====================")
    start = time.time()
    t = Tree(None, all_columns, 2)
    p = t.build(*learn)
    end = time.time()
    print(f"Build time: {end - start}")
    predictions_learn = p.predict(learn[0])
    predictions_test = p.predict(test[0])
    m_train = report_metrics(predictions_learn, learn[1])
    m_test = report_metrics(predictions_test, test[1])
    return m_train, m_test

def hw_randomforests(learn, test):
    print("Random Forest")
    print("====================")
    start = time.time()
    rand = random.Random()
    rand.seed(42)
    f = RandomForest(rand=rand, n=100)
    predictor = f.build(*learn)
    end = time.time()
    print(f"Build time: {end - start}")
    predictions_learn = predictor.predict(learn[0])
    predictions_test = predictor.predict(test[0])
    m_train = report_metrics(predictions_learn, learn[1])
    m_test = report_metrics(predictions_test, test[1])
    start2 = time.time()
    imps = predictor.importance()
    end2 = time.time()
    print(f"Importance took: {end2 - start2}")
    np.save("importances", imps)
    
    #start3 = time.time()
    #imps3 = predictor.importances3()
    #end3 = time.time()
    #print(f"Importances3 took: {end3 - start3}")
    #print(np.max(imps3))
    #ind = np.unravel_index(np.argmax(imps3, axis=None), imps3.shape)
    #print(ind)
    #plt.bar(range(len(imps)), imps)
    #plt.show()
    return m_train, m_test

def misclassification_rate(predictions, labels):
    incorrect = sum(pred != lab for pred, lab in zip(predictions, labels))
    return incorrect/len(labels)

def root_features(X, y):
    rand = random.Random()
    rand.seed(42)
    tree = Tree()
    feats = []
    for i in range(100):
        possible_indices = range(X.shape[0])
        bootstrap_indices = rand.choices(possible_indices, k=X.shape[0])
        column, threshold = tree.best_gini(X[bootstrap_indices], y[bootstrap_indices], range(X.shape[1]))
        feats.append(column)
    return np.array(feats)


def tki():
    legend, Xt, yt = read_tab("tki-train.tab", {"Bcr-abl": 1, "Wild type": 0})
    _, Xv, yv = read_tab("tki-test.tab", {"Bcr-abl": 1, "Wild type": 0})
    return (Xt, yt), (Xv, yv), legend


if __name__ == "__main__":
    learn, test, legend = tki()

    #print("full", hw_tree_full(learn, test))
    #print("random forests", hw_randomforests(learn, test))
    feats = root_features(*learn)
    np.save("root_features", feats)
    plt.bar(range(len(feats)), feats)
    plt.show()