import numpy as np

imps = np.load("importances_1000.npy")
imps3 = np.load("importances3_1000.npy")

print(np.max(imps))
print(np.argsort(imps)[-3:][::-1])
print(np.max(imps3))
ind = np.unravel_index(np.argmax(imps3, axis=None), imps3.shape)
print(ind)
