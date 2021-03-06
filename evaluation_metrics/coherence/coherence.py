import numpy as np
import scipy.misc
from six.moves import xrange


def clip(x):
    if x < 1e-10:
        x = 1e-10
    return x


class Coherence():
    """
    @param: Top M tokens to calculate the coherence.
    @param: Coherence type:
        'OC-Auto-PMI': oc_auto_pmi
        'OC-Auto-NPMI': oc_auto_npmi
        'OC-Auto-LCP': oc_auto_lcp
        'OC-Auto-DS': oc_auto_ds
    @param: Distance function

    """
    def __init__(self, M=20, type='default', dist='cosine'):
        self.M = M
        self.type = type
        self.dist = dist

    def score(self, classifier, X, y, fit=True):
        if fit == True:
            classifier.fit(X, y)
        topicSz = classifier.classes_.shape[0]
        print("topicSz: %s" % topicSz)
        coherences = np.zeros(topicSz)
        for topic in xrange(topicSz):
            coherences[topic] = coherence(classifier.estimators_[topic].coef_,
                                          X,
                                          self.M,
                                          type=self.type,
                                          dist=self.dist)
        return coherences


def topic_coherence(vmvl, vl):
    return np.log((vmvl + 1.0) / vl)


def oc_auto_pmi(Pwjwi, Pwi, Pwj):
    return np.log(Pwjwi / (Pwi * Pwj))


def oc_auto_npmi(Pwjwi, Pwi, Pwj):
    return (np.log(Pwjwi / (Pwi * Pwj))) / (-np.log(Pwjwi))


def oc_auto_lcp(Pwjwi, Pwi):
    return np.log(Pwjwi / Pwi)


def sim_cos(wi, wj):
    return (np.dot(wi, wj) + 1) / (np.linalg.norm(wi) * np.linalg.norm(wj) + 1)


def sim_dice(wi, wj):
    N = wi.shape[0]
    return (2 * np.sum([min(wi[n], wj[n]) for n in xrange(N)]) + 1) / (np.sum(wi + wj) + 1)


def sim_jaccard(wi, wj):
    N = wi.shape[0]
    return (np.sum([min(wi[n], wj[n]) for n in xrange(N)]) + 1) / (np.sum([max(wi[n], wj[n]) for n in xrange(N)]) + 1)


def oc_auto_ds(wi, wj, dist=sim_cos):
    return dist(wi, wj)


def coherence(coefs, X, M, type='default', dist='cosine'):
    numDocs = X.shape[0]
    coherenceType = {
        'default': topic_coherence,
        'OC-Auto-PMI': oc_auto_pmi,
        'OC-Auto-NPMI': oc_auto_npmi,
        'OC-Auto-LCP': oc_auto_lcp,
        'OC-Auto-DS': oc_auto_ds
    }

    if type == 'OC-Auto-DS':
        distributionType = {
            'cosine': sim_cos,
            'dice': sim_dice,
            'jaccard': sim_jaccard
        }

    coherenceMetric = coherenceType[type]
    topTokens = (-coefs).argsort()[0][:M]
    print("topTokens %s" % topTokens)
    cohere = 0

    # Calculating coherence for all pairs of top tokens (or weights of top tokens).
    for m in xrange(1, M):
        for l in xrange(0, m - 1):
            if type == 'default':
                vl, vm, vmvl = calculate_doc_occurence(X, l, m, topTokens)
                cohere += coherenceMetric(vmvl.shape[0], vm.shape[0], vl.shape[0])

            elif type == 'OC-Auto-DS':
                wi = (X[:, topTokens[m]] > 0).astype(np.int).toarray().T[0]
                wj = (X[:, topTokens[l]] > 0).astype(np.int).toarray().T[0]
                distribution = distributionType[dist]
                cohere += coherenceMetric(wi, wj, distribution)

            elif type == 'OC-Auto-LCP':
                vl, vm, vmvl = calculate_doc_occurence(X, l, m, topTokens)
                Pwjwi = clip(vmvl.shape[0] + 1 / float(numDocs))
                Pwi = clip(vl.shape[0] / float(numDocs))
                cohere += coherenceMetric(Pwjwi, Pwi)

            else:
                # PMI or NPMI
                vl, vm, vmvl = calculate_doc_occurence(X, l, m, topTokens)
                Pwjwi = clip(vmvl.shape[0] / float(numDocs))
                Pwi = clip(vl.shape[0] / float(numDocs))
                Pwj = clip(vm.shape[0] / float(numDocs))
                cohere += coherenceMetric(Pwjwi, Pwi, Pwj)

    if type == 'OC-Auto-DS':
        cohere = cohere / scipy.misc.comb(M, 2)

    return cohere

def calculate_doc_occurence(X, l, m, topTokens):
    vm = np.nonzero(X[:, topTokens[m]])[0] # indices that are non-zero
    vl = np.nonzero(X[:, topTokens[l]])[0] # which docs that this particular token appeared
    vmvl = np.intersect1d(vm, vl)
    return vl, vm, vmvl
