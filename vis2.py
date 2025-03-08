import numpy as np
import matplotlib.pyplot as plt

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