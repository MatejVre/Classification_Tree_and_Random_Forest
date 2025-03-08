import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter

def custom_tick_format(x, pos):
    return f"{x * 2 + 1000:.0f}"


importances = np.load("importances.npy")
features = np.load("root_features.npy")
print(len(np.unique(features)))

figure, axis = plt.subplots()
axis.bar(range(len(importances)), importances, width=1.3, color='#ff7f0e', label="Feature importances")
axis.set_ylabel("Feature importance")
axis.set_xlim(0, 400)
axis.xaxis.set_major_formatter(FuncFormatter(custom_tick_format))

axis2 = axis.twinx()
axis2.hist(features, bins=30, color='#1f77b4', alpha=0.6, label="Root feature count")
axis2.set_ylabel("Root feature count")

axis2.set_xlim(0, 400)
axis2.set_ylim(-14.193)

axis.set_xlabel("Feature")
handles1, labels1 = axis.get_legend_handles_labels()
handles2, labels2 = axis2.get_legend_handles_labels()

axis.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
plt.show()