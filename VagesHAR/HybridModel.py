import numpy as np
import sklearn

from VagesHAR.SeparatedActivityClassifier import SeparatedActivityClassifier
from VagesHAR.StatisticHelpers import f_score


class HybridModel:
    def __init__(self, random_state=None, class_weight=None):
        self.ordering_classifier = SeparatedActivityClassifier(classifier_type="RandomForest", random_state=random_state,
                                                               class_weight=class_weight)
        self.min_activation_classifier = SeparatedActivityClassifier(classifier_type="SVM", random_state=random_state,
                                                                     class_weight=class_weight)
        self.thresholds = dict()

    def fit(self, X, y):
        self.ordering_classifier.fit(X, y)
        self.min_activation_classifier.fit(X, y)

        for activity in self.get_activity_list():
            activity_booleans = np.array([True if l == activity else False for l in y])
            probability_and_label = list(
                zip(self.min_activation_classifier.predict_proba_activity(X, activity)[:, 1], activity_booleans))
            optimal_threshold = self.find_optimal_threshold(probability_and_label, beta=1)
            self.thresholds[activity] = optimal_threshold

        return self

    def get_activity_list(self):
        return self.ordering_classifier.get_activity_list()

    def predict(self, X):
        forest_probabilities = self.ordering_classifier.predict_proba(X)
        svms_probabilities = self.min_activation_classifier.predict_proba(X)
        arg_maxes = self.ordering_classifier.predict(X)
        arg_max_count = 0

        predicted = []
        activities = self.get_activity_list()
        number_of_activities = len(activities)

        for i in range(len(arg_maxes)):
            forest_probability = forest_probabilities[i]
            svm_probability = svms_probabilities[i]
            tups = zip(range(number_of_activities), forest_probability, svm_probability)
            tups = sorted(tups, key=lambda x: x[1])

            for t in tups:
                j, _, svm_probability = t
                act_label = activities[j]
                act_threshold = self.thresholds[act_label]
                if svm_probability >= act_threshold:
                    predicted.append(act_label)
                    break
            else:
                predicted.append(arg_maxes[i])
                arg_max_count += 1

        print("Used", arg_max_count, "argmaxes, out of", len(X))
        return np.array(predicted)

    def score(self, X, y):
        y_pred = self.predict(X)

        return sklearn.metrics.accuracy_score(y, y_pred)

    @staticmethod
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
