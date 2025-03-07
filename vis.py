import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

importances = np.load("importances.npy")
features = np.load("root_features.npy")

figure, axis = plt.subplots(2, 1)

# First plot: Bar chart
axis[0].bar(range(len(importances)), importances, width=1.3)
axis[0].set_ylabel("Importances")
axis[0].set_xlim(0, 400)

axis[1].hist(features, bins=20)
axis[1].set_ylabel("Root feature count")
axis[1].grid(axis="y")
axis[1].set_xlim(0, 400)

plt.show()