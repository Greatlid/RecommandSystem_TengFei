from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

import numpy as np
np.savetxt('X_train.txt', X_train)
np.savetxt('X_test.txt', X_test)
np.savetxt('y_train.txt', y_train)
np.savetxt('y_test.txt', y_test)