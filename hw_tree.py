import csv
import numpy as np
import random
from collections import Counter
import time
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.ticker import FuncFormatter

def all_columns(X, rand):
    return range(X.shape[1])    


def random_sqrt_columns(X, rand):
    n_cols = X.shape[1]
    sqrt_cols = int(np.sqrt(n_cols))
    indices = rand.sample(range(0, n_cols), k=sqrt_cols)
    return np.array(indices)

def get_top_3_features(X, rand):
    imps = np.load("importances_1000.npy")
    return np.argsort(imps)[-3:][::-1]

def get_top_triplet(X, rand):
    imps3 = np.load("importances3_1000.npy")
    return np.unravel_index(np.argmax(imps3, axis=None), imps3.shape)

def get_custom_threes(X, rand):
    return [275, 273, 278]


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
        return TreeModel(self.build_tree_recursively(X, y), sorted(set(self.columns)))  #Sorted to have consistent indexing for importances3

    def build_tree_recursively(self, X, y):
        if len(np.unique(y)) == 1:
            return Node(prediction=y[0])
        
        if len(y) < self.min_samples:
            return Node(prediction=Counter(y).most_common(1)[0][0])
        column, threshold = self.best_gini(X, y, self.get_candidate_columns(X, self.rand))
        self.columns.append(int(column))

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
        left_indices = np.where(X[:, column] <= threshold)[0] #Whole splitting could be improved for better efficiency by only returning indices and never passing data.
        right_indices = np.where(X[:, column] > threshold)[0]
        return ((X[left_indices], y[left_indices]), (X[right_indices], y[right_indices]))

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

            X_copy = X_cur.copy()

            for p in self.trees[i].columns:
                X_copy[:, p] = np.random.permutation(X_copy[:, p])
                permuted_predictions = self.trees[i].predict(X_copy)
                permuted_accuracy = self.prediction_accuracy(permuted_predictions, y_cur)
                imps[p] += (accuracy - permuted_accuracy)/len(self.trees)
                X_copy[:, p] = X_cur[:, p]


        return imps
    
    def importances3(self):
        
        np.random.seed(42)
        imps_matrix = np.zeros((self.X.shape[1], self.X.shape[1], self.X.shape[1]))
            
        for i in range(len(self.trees)):
            X_cur = self.X[self.OOB_indices_collection[i]]
            y_cur = self.y[self.OOB_indices_collection[i]]
            predictions = self.trees[i].predict(X_cur)
            accuracy = self.prediction_accuracy(predictions, y_cur)

            X_copy = X_cur.copy()

            for p in combinations(self.trees[i].columns, 3):
                X_copy[:, p[0]] = np.random.permutation(X_copy[:, p[0]])
                X_copy[:, p[1]] = np.random.permutation(X_copy[:, p[1]])
                X_copy[:, p[2]] = np.random.permutation(X_copy[:, p[2]])
                permuted_predictions = self.trees[i].predict(X_copy)
                permuted_accuracy = self.prediction_accuracy(permuted_predictions, y_cur)
                imps_matrix[p[0]][p[1]][p[2]] = (accuracy - permuted_accuracy)/len(self.trees)
                X_copy[:, p] = X_cur[:, p]

        return imps_matrix
    
    def importance3_structure(self):
        imps = dict()
        for t in self.trees:
            tree_imps = dict.fromkeys(set(t.columns), 0)
            self.traverse(tree_imps, t.root)
            for key, value in tree_imps.items():
                if key not in imps.keys():
                    imps[key] = value
                else:
                    imps[key] += value
        return sorted(imps.items(), key=lambda x: x[1], reverse=True)[:3]
    
    def traverse(self, dic, node, d=1):
        
        if node.prediction != None: #leaf
            return
        dic[node.column_index] += 1/d
        self.traverse(dic, node.left, d=d+1)
        self.traverse(dic, node.right, d=d+1)

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
    rand = random.Random()
    rand.seed(42)

    miscs = []
    
    for i in range(150):
        possible_indices = range(y_pred.shape[0])
        bootstrap_indices = rand.choices(possible_indices, k=y_pred.shape[0])
        miscs.append(misclassification_rate(y_pred[bootstrap_indices], y_true[bootstrap_indices]))

    print(f"Misclassification rate: {round(misclassification, 4)} | standard error: {SE} | bootstrap SD: {np.std(miscs)}")
    return misclassification, np.std(miscs)


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

    #np.save("importances", imps)
    start3 = time.time()
    imps3 = predictor.importances3()
    end3 = time.time()
    print(f"Importances3 took: {end3 - start3}")

    print(np.max(imps3))
    struct_imps = (predictor.importance3_structure())
    print(struct_imps) #returns 275, 273, 278
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

def importances_1000(X, y):
    rand = random.Random()
    rand.seed(42)
    start = time.time()
    forest = RandomForest(rand=rand, n=1000)
    predictor = forest.build(X, y)
    stop = time.time()
    print(f"Forest took: {stop - start}")
    start = time.time()
    imps = predictor.importance()
    stop = time.time()
    print(f"Importances took: {stop - start}")
    np.save("importances_1000", imps)
    start = time.time()
    imps3 = predictor.importances3()
    stop = time.time()
    print(f"Importances3 took: {stop - start}")
    np.save("importances3_1000", imps3)
    
def important_trees(learn, test):
    rand = random.Random()
    rand.seed(42)
    top_3_tree = Tree(rand=rand, get_candidate_columns=get_top_3_features)
    top_3_predictor = top_3_tree.build(*learn)
    top_3_learn = report_metrics(top_3_predictor.predict(learn[0]), learn[1])
    top_3_test = report_metrics(top_3_predictor.predict(test[0]), test[1])
    top_triplet_tree = Tree(rand=rand, get_candidate_columns=get_top_triplet)
    top_triplet_predictor = top_triplet_tree.build(*learn)
    top_triplet_test = report_metrics(top_triplet_predictor.predict(learn[0]), learn[1])
    top_triplet_test = report_metrics(top_triplet_predictor.predict(test[0]), test[1])

def misclassification_vs_size(learn, test):
    misclass = np.zeros(100)
    uncertainties = np.zeros(100)
    rand = random.Random()
    rand.seed(42)
    f = RandomForest(rand=rand, n=100)
    p = f.build(*learn)
    all_trees = p.trees

    for i in range(1, 101):
        p.trees = all_trees[:i]
        m, u = report_metrics(p.predict(test[0]), test[1])
        misclass[i-1] = m
        uncertainties[i-1] = u
    return misclass, uncertainties

def custom_important_tree(learn, test):
    rand = random.Random()
    rand.seed(42)
    t = Tree(rand=rand, get_candidate_columns=get_custom_threes)
    p = t.build(*learn)
    report_metrics(p.predict(test[0]), test[1])

def tki():
    legend, Xt, yt = read_tab("tki-train.tab", {"Bcr-abl": 1, "Wild type": 0})
    _, Xv, yv = read_tab("tki-test.tab", {"Bcr-abl": 1, "Wild type": 0})
    return (Xt, yt), (Xv, yv), legend

#==============================================================
#HELPER METHODS FOR VISUALISATION
#These were originally in their own files but i included them just in case

def vis2():
    misclassifications = np.load("misclassifications.npy")
    uncertainties = np.load("uncertainties.npy")

    plt.figure(figsize=(8,4))
    plt.plot(range(1, len(misclassifications)+1), misclassifications[:], label="Misclassification rate")
    plt.fill_between(range(1, len(misclassifications)+1), 
                    misclassifications[:] - uncertainties[:], 
                    misclassifications[:] + uncertainties[:], 
                    color="blue", alpha=0.2, label="Uncertainty")
    plt.ylim((0.0))
    plt.legend()
    plt.xlim((1,100))
    plt.grid("y")
    plt.ylabel("Misclassification rate")
    plt.xlabel("Number of trees")
    plt.savefig("misclassifications_with_uncertainties.pdf", bbox_inches="tight")
    plt.show()

def custom_tick_format(x, pos):
    return f"{x * 2 + 1000:.0f}"

def vis():
    importances = np.load("importances.npy")
    features = np.load("root_features.npy")
    counter = dict.fromkeys(range(len(importances)), 0)
    for i in features:
        counter[i] += 1

    vals = [v for v in counter.values()]
        
    figure, axis = plt.subplots()
    axis.bar(range(len(importances)), importances, width=1.3, label="Feature importances")
    axis.set_ylabel("Feature importance")
    axis.set_xlim(0, 400)
    axis.set_ylim((-0.002, 0.016))
    axis.xaxis.set_major_formatter(FuncFormatter(custom_tick_format))

    axis2 = axis.twinx()
    axis2.plot(range(len(importances)), vals, color='#d62728', alpha=1, label="Root feature count")
    axis2.set_ylabel("Root feature count")

    axis2.set_xlim(0, 396)
    axis2.set_ylim((-10, 80))

    axis.set_xlabel("Feature")
    handles1, labels1 = axis.get_legend_handles_labels()
    handles2, labels2 = axis2.get_legend_handles_labels()

    axis.grid(axis="y")
    axis.set_axisbelow(True)
    axis.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    plt.savefig("importances.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    learn, test, legend = tki()

    #hw_tree_full(learn, test)
    #hw_randomforests(learn, test)
    #importances_1000(*learn)
    #important_trees(learn, test)
    #misclass, uncertainties = misclassification_vs_size(learn, test)
    #np.save("misclassifications", misclass)
    #np.save("uncertainties", uncertainties)
    #custom_important_tree(learn, test)