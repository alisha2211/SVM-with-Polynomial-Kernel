# SVM-with-Polynomial-Kernel

from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

# Create a circular dataset for non-linear classification
np.random.seed(0)
X = np.random.randn(100, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
y = y.astype(int)

# Initialize SVM classifier with a polynomial kernel
svm = SVC(kernel='poly', degree=3, C=1)
svm.fit(X, y)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='autumn', edgecolor='k', s=50)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Plot decision boundary
xx = np.linspace(xlim[0], xlim[1], 100)
yy = np.linspace(ylim[0], ylim[1], 100)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm.predict(xy).reshape(XX.shape)

plt.contourf(XX, YY, Z, alpha=0.3, cmap='autumn')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("SVM with Polynomial Kernel")
plt.show()
