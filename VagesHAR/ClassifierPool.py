from collections import defaultdict

from VagesHAR.BinaryMultiLabelClassifier import BinaryMultiLabelClassifier


class ClassifierPool:
    def __init__(self, type="RandomForest", random_state=None, thresholded=False, class_weight=None, n_jobs=1,
                 max_depth=None, n_estimators=10):
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._n_jobs = n_jobs
        self._class_weight = class_weight
        self._thresholded = thresholded
        self._random_state = random_state
        self._type = type
        self._pool = dict()

    def get_classifier(self, subject_id):
        return self._pool[subject_id]

    def put_classifier(self, subject_id, classifier):
        self._pool[subject_id] = classifier

    def train_one(self, subject_id, X, y):
        new_classifier = BinaryMultiLabelClassifier(type=self._type, random_state=self._random_state,
                                                    thresholded=self._thresholded,
                                                    class_weight=self._class_weight, n_jobs=self._n_jobs,
                                                    n_estimators=self._n_estimators,
                                                    max_depth=self._max_depth).fit(X, y)
        self.put_classifier(subject_id, new_classifier)

    def find_best_individual(self, X, y, ignored_ids=None):
        scores = []

        evaluated_ids = set(self._pool.keys())

        if ignored_ids:
            evaluated_ids -= set(ignored_ids)

        for subject_id in evaluated_ids:
            scores.append((self.get_classifier(subject_id).score(X, y), subject_id))

        scores.sort()
        best_id = scores.pop()[1]
        return self.get_classifier(best_id)

    def adapt(self, X, y, ignored_ids=None):
        activity_accuracies = defaultdict(list)

        evaluated_ids = set(self._pool.keys())

        if ignored_ids:
            evaluated_ids -= set(ignored_ids)

        for subject_id in evaluated_ids:
            subject_accuracies = self.get_classifier(subject_id).score_each_activity(X, y)
            for activity in subject_accuracies:
                score = subject_accuracies[activity]
                activity_accuracies[activity].append((score, subject_id))

        new_individual = BinaryMultiLabelClassifier(type=self._type, random_state=self._random_state,
                                                    thresholded=self._thresholded,
                                                    class_weight=self._class_weight)

        for activity in activity_accuracies:
            activity_accuracies[activity].sort()
            _, subject_id = activity_accuracies[activity].pop()
            classifier = self.get_classifier(subject_id).get_activity_classifier(activity)
            new_individual.set_activity_classifier(activity, classifier)

        return new_individual
