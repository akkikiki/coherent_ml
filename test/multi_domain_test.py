from datasets import multi_domain_sentiment
from sklearn.multiclass import OneVsRestClassifier
from sklearn import linear_model
from evaluation_metrics.coherence.coherence import Coherence
import numpy as np
from six.moves import xrange


def make_classifiers(clfs, penalties, Cs):
    clfDict = {
        "LogisticRegression": linear_model.LogisticRegression,
        "SGDClassifier": linear_model.SGDClassifier,
        "RidgeClassifier": linear_model.RidgeClassifier,
        "PassiveAggressiveClassifier": linear_model.PassiveAggressiveClassifier
    }
    classifiers = []
    for clf in clfs:
        if clf in clfDict:
            for penalty in penalties:
                for C in Cs:
                    classifiers.append(OneVsRestClassifier(clfDict[clf](penalty=penalty, C=C)))
    return classifiers


# domains = ['books','dvd','electronics','kitchen']
domains = ['electronics', 'kitchen']
data = []
for domain in domains:
    data.append(multi_domain_sentiment.load(domain))

N = len(domains)
outData = []
# penalties = ['l1', 'l2']
penalties = ['l1']
# Cs = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
Cs = [1.0]
clfs = make_classifiers(["LogisticRegression"], penalties, Cs)
print("Num. of different classifiers that we will test: %s" % len(clfs))
cohere = Coherence(type='OC-Auto-NPMI')
"""
OC-Auto-NPMI is pairwise PMI of top-N topic words.
See http://www.aclweb.org/anthology/E14-1056 for more details.
"""
for i, clf in enumerate(clfs):
    print("Classifier %s" % i)
    for i in xrange(N):
        clf.fit(data[i].X, data[i].y)
        print(data[i].X, data[i].y)
        """
        Calculate the coherence score for each class
        """
        cScore = cohere.score(clf, data[i].X, data[i].y, fit=False)
    print("coherence score: %s" % cScore)
    for j in xrange(N):
        if j != i:
            aScore = clf.score(data[j].X, data[j].y)
            clf.fit(data[j].X, data[j].y)
            cScores = np.append(cScore, cohere.score(clf, data[j].X, data[j].y, fit=False))
            outData.append(np.append(aScore, cScores))

np.savetxt("./multi_domain.csv", outData, delimiter=",")
