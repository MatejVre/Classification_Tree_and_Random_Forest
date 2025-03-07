import unittest
import numpy as np
from hw_tree import Tree

class MiscTests(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[1, 3],
                           [6, 3],
                           [3, 4],
                           [5, 1],
                           [5, 2],
                           [3, 1]])
        self.y = np.array([0, 0, 1, 0, 1, 1])
        self.t = Tree()
        self.pred = self.t.build(self.X, self.y)

    def test_best_gini_in_column(self):  
        self.assertEqual(self.t.best_gini_in_column(self.X, self.y, 0), (0.40, 2))
        self.X = np.array([[6, 3],
                           [3, 4],
                           [5, 1],
                           [5, 2],
                           [3, 1]])
        self.y = np.array([0, 1, 0, 1, 1])
        self.assertEqual(self.t.best_gini_in_column(self.X, self.y, 0), (0.27, 4))
        self.X = np.array([[6, 3],
                           [5, 1],
                           [5, 2],])
        self.y = np.array([0, 0, 1])
        self.assertEqual(self.t.best_gini_in_column(self.X, self.y, 0), (0.33, 5.5))
    
    def test_best_gini(self):
        self.assertEqual(self.t.best_gini(self.X, self.y, [0, 1]), (0, 2))
        self.X = np.array([[6, 3],
                           [3, 4],
                           [5, 1],
                           [5, 2],
                           [3, 1]])
        self.y = np.array([0, 1, 0, 1, 1])
        self.assertEqual(self.t.best_gini(self.X, self.y, [0, 1]), (0, 4))
        self.X = np.array([[6, 3],
                           [5, 1],
                           [5, 2],])
        self.y = np.array([0, 0, 1])
        self.assertEqual(self.t.best_gini(self.X, self.y, [0, 1]), (0, 5.5))
    
    def test_predictions(self):
        self.assertEqual(self.pred.predict(np.array([[1,5],
                                                     [4, 12],
                                                     [8, 0]])).tolist(), [0, 1, 0])
    
    #create dataset to manually test the outputs of the model!
    #efficient check for the best split:
    #threshold in the middle of the unique values instead of directly on them!!! (it generalizes better)
    #Try the "rolling window" approach (look at the differences!)
    #Why select a set of features for each split, not tree? IF you select a particulary bad combination, the tree would probably not perform good. We can get additional depth.

    #mean and sd of different test sets is a good way of quantifying the uncertainty. But we have only one test set (BOOOTSRAP BABYYYYY!)
    #Bootstrap the errors! 100+?
    #for SD(e)/sqrt(n)

    #who the report is for? make grading easier.
    
 
if __name__ == "__main__":
    unittest.main()