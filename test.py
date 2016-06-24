from datasets import twenty_newsgroups
from linear_models import softmax_regression
from sklearn import linear_model
import numpy as np

data = twenty_newsgroups.TwentyNewsgroups()
ind_train = np.random.permutation(data.y.shape[0])[:2000]
ind_test = np.random.permutation(data.y.shape[0])[:500]
X_train = data.X[ind_train,:]
y_train = data.y[ind_train]
X_test = data.X[ind_test,:]
y_test = data.y[ind_test]

clf = softmax_regression.LogisticRegression()
clf.fit(X_train,y_train,max_iterations=250,step_size=.50,opt='NAG')
scoreclf = clf.score(X_test,y_test)

#clf2 = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs')
#clf2.fit(X_train,y_train)
#scoreclf2 = clf2.score(X_test,y_test)

print(scoreclf)
#print(scoreclf2)
