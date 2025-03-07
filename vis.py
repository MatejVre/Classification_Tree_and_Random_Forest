import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

importances = np.load("importances.npy")
features = np.load("root_features.npy")
print(len(np.unique(features)))

figure, axis = plt.subplots()
# First plot: Bar chart
axis.bar(range(len(importances)), importances, width=1.3, color='#37afa9')
axis.set_ylabel("Importances")
#axis.grid(axis="y")
#axis.set_ylim(0, 0.0175)
axis.set_xlim(0, 400)
axis2 = axis.twinx()
axis2.hist(features, bins=20, color='red', alpha=0.3)
axis2.set_ylabel("Root feature count")
#axis2.grid(axis="y")
axis2.set_xlim(0, 400)
axis2.set_ylim(-14.193)
print(max(importances))
#152329
plt.show()