import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter

def custom_tick_format(x, pos):
    return f"{x * 2 + 1000:.0f}"


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