from datasets import twenty_newsgroups
from linear_models import tf_softmax_regression
# from sklearn import linear_model
import numpy as np

data = twenty_newsgroups.TwentyNewsgroups(hot_one=True,min_feat=25)
# ind_train = np.random.permutation(data.y.shape[0])[:2000]
# ind_test = np.random.permutation(data.y.shape[0])[2000:2500]
ind_train = np.random.permutation(data.y.shape[0])[:1]
ind_test = np.random.permutation(data.y.shape[0])[1:2]
X_train = data.X[ind_train,:]
y_train = data.y[ind_train]
X_test = data.X[ind_test,:]
y_test = data.y[ind_test]
# print(len(X_train))
# X_train is sparse matrix

print("Starting to train LR model")
clf = tf_softmax_regression.LogisticRegression()
# clf.fit(X_train,y_train,learning_rate=0.01,training_epochs=25,batch_size=100,loss='l2')
clf.fit(X_train,y_train,learning_rate=0.01,training_epochs=1,batch_size=100,loss='l2')
print(clf.score(X_test,y_test))
print(clf.coef_)
print(clf.intercept_)

#clf2 = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs')
#clf2.fit(X_train,y_train)
#scoreclf2 = clf2.score(X_test,y_test)

#print(scoreclf2)
