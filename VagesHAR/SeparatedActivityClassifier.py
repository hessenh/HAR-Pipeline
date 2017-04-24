import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier as Rfc
from sklearn.svm import SVC


class SeparatedActivityClassifier:
    def __init__(self, classifier_type="RandomForest", random_state=None, class_weight=None, n_jobs=1, max_depth=None,
                 n_estimators=10):
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._n_jobs = n_jobs
        self._class_weight = class_weight
        self._activity_classifiers = dict()
        self._type = classifier_type
        self._random_state = random_state

    def _train_one_activity_classifier(self, X, y, activity):
        activity_booleans = np.array([True if label == activity else False for label in y])

        if self._type == "SVM":
            activity_classifier = SVC(kernel="poly", degree=2, C=1, probability=True, random_state=self._random_state,
                                      class_weight=self._class_weight)
        else:
            activity_classifier = Rfc(random_state=self._random_state, n_estimators=self._n_estimators,
                                      max_depth=self._max_depth, n_jobs=self._n_jobs, class_weight=self._class_weight)

        activity_classifier.fit(X, activity_booleans)
        self.set_activity_classifier(activity, activity_classifier)

    def get_activity_list(self):
        return sorted(list(self._activity_classifiers.keys()))

    def fit(self, X, y):
        activities_in_training_labels = set(y)

        for activity in activities_in_training_labels:
            self._train_one_activity_classifier(X, y, activity)

        return self

    def predict_proba(self, X):
        results = [self.predict_proba_activity(X, activity)[:, 1] for activity in self.get_activity_list()]

        return np.vstack(results).transpose()

    def predict(self, X):
        probabilities = self.predict_proba(X)
        indices_of_most_probable_activities = np.argmax(probabilities, axis=1)

        activity_labels = self.get_activity_list()
        return np.array([activity_labels[i] for i in indices_of_most_probable_activities])

    def score(self, X, y):
        predictions = self.predict(X)

        return sklearn.metrics.accuracy_score(y, predictions)

    def predict_activity(self, X, activity):
        return self.get_activity_classifier(activity).predict(X)

    def predict_proba_activity(self, X, activity):
        return self.get_activity_classifier(activity).predict_proba(X)

    def score_activities_separately(self, X, y):
        activity_accuracies = dict()

        for activity in self.get_activity_list():
            activity_booleans = [True if label == activity else False for label in y]
            boolean_predictions = self.predict_activity(X, activity)
            activity_accuracies[activity] = sklearn.metrics.accuracy_score(activity_booleans, boolean_predictions)

        return activity_accuracies

    def set_activity_classifier(self, activity, classifier):
        self._activity_classifiers[activity] = classifier

    def get_activity_classifier(self, activity):
        return self._activity_classifiers[activity]
