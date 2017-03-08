import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier as Rfc
from sklearn.svm import SVC


class BinaryMultiLabelClassifier:
    def __init__(self, type="RandomForest", random_state=None, thresholded=False, class_weight=None, n_jobs=1,
                 max_depth=None, n_estimators=10):
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._n_jobs = n_jobs
        self._class_weight = class_weight
        self._classifiers = dict()
        self._type = type
        self._random_state = random_state
        self._thresholded = thresholded
        self._thresholds = dict()

    def train_one_classifier(self, X, y, a):
        temp_labels = np.array([True if l == a else False for l in y])

        if self._type == "SVM":
            new_classifier = SVC(kernel="poly", degree=2, C=1, probability=True, random_state=self._random_state,
                                 class_weight=self._class_weight)
        else:
            new_classifier = Rfc(random_state=self._random_state, n_estimators=self._n_estimators,
                                 max_depth=self._max_depth, n_jobs=self._n_jobs, class_weight=self._class_weight)

        new_classifier.fit(X, temp_labels)
        self.set_activity_classifier(a, new_classifier)

        if self._thresholded:
            prob_pairs = list(zip(self._classifiers[a].predict_proba(X)[:, 1], temp_labels))
            threshold = find_optimal_threshold(prob_pairs, beta=1)
            self.set_threshold(a, threshold)

    def set_threshold(self, a, value):
        self._thresholds[a] = value

    def get_recognized_activities(self):
        return sorted(list(self._classifiers.keys()))

    def fit(self, X, y):
        activities_in_training_labels = set(y)

        for a in activities_in_training_labels:
            self.train_one_classifier(X, y, a)

        return self

    def predict_proba(self, X):
        results = [self.get_activity_classifier(a).predict_proba(X)[:, 1] for a in self.get_recognized_activities()]

        return np.vstack(results).transpose()

    def predict(self, X):
        probs = self.predict_proba(X)
        max_indices = np.argmax(probs, axis=1)

        activity_lookup = self.get_recognized_activities()
        return np.array([activity_lookup[i] for i in max_indices])

    def score(self, X, y):
        y_pred = self.predict(X)

        return sklearn.metrics.accuracy_score(y, y_pred)

    def predict_activity(self, X, a):
        return self.get_activity_classifier(a).predict(X)

    def score_each_activity(self, X, y):
        accuracies = dict()

        for a in self.get_recognized_activities():
            y_pred = self.predict_activity(X, a)
            labels = [True if l == a else False for l in y]
            accuracies[a] = sklearn.metrics.accuracy_score(labels, y_pred)

        return accuracies

    def set_activity_classifier(self, a, classifier):
        self._classifiers[a] = classifier

    def get_activity_classifier(self, a):
        return self._classifiers[a]


def f_score(tp, fp, fn, beta):
    beta_2 = beta * beta
    score = (1 + beta_2) * tp / ((1 + beta_2) * tp + beta_2 * fn + fp)
    return score


def find_optimal_threshold(pairs, beta):
    pairs = sorted(pairs, reverse=True)

    true_negatives = 0
    false_negatives = 0
    true_positives = sum(p[1] for p in pairs)
    false_positives = len(pairs) - true_positives

    best_threshold = 0.0
    max_f_score = float("-inf")

    while pairs:
        prob, classification = pairs.pop()

        if classification:
            false_negatives += 1
            true_positives -= 1
        else:
            true_negatives += 1
            false_positives -= 1

        if pairs and prob == pairs[-1][1]:
            continue

        score = f_score(tp=true_positives, fp=false_positives, fn=false_negatives, beta=beta)
        if score > max_f_score:
            max_f_score = score
            best_threshold = prob

    return best_threshold
