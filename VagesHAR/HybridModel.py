import numpy as np
import sklearn

from VagesHAR.BinaryMultiLabelClassifier import BinaryMultiLabelClassifier


class HybridModel:
    def __init__(self, random_state=None, class_weight=None):
        self.forests = BinaryMultiLabelClassifier(type="RandomForest", random_state=random_state,
                                                  class_weight=class_weight)
        self.svms = BinaryMultiLabelClassifier(type="SVM", random_state=random_state, thresholded=True,
                                               class_weight=class_weight)
        self.activities = []
        self.thresholds = dict()

    def fit(self, X, y):
        self.forests.fit(X, y)
        self.svms.fit(X, y)
        self.activities = self.forests.get_recognized_activities()
        self.thresholds = self.svms._thresholds

        return self

    def predict(self, X):
        forest_proba = self.forests.predict_proba(X)
        svms_proba = self.svms.predict_proba(X)
        argmaxes = self.forests.predict(X)
        argmax_count = 0

        predicted = []
        no_acts = len(self.activities)

        for i in range(len(argmaxes)):
            f = forest_proba[i]
            s_p = svms_proba[i]
            tups = zip(range(no_acts), f, s_p)
            tups = sorted(tups, key=lambda x: x[1])

            for t in tups:
                j, _, s_p = t
                act_label = self.activities[j]
                act_threshold = self.thresholds[act_label]
                if s_p >= act_threshold:
                    predicted.append(act_label)
                    break
            else:
                predicted.append(argmaxes[i])
                argmax_count += 1

        print("Used", argmax_count, "argmaxes, out of", len(X))
        return np.array(predicted)

    def score(self, X, y):
        y_pred = self.predict(X)

        return sklearn.metrics.accuracy_score(y, y_pred)