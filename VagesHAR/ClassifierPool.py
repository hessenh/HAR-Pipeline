from collections import defaultdict

from VagesHAR.SeparatedActivityClassifier import SeparatedActivityClassifier


class ClassifierPool:
    def __init__(self, classifier_type="RandomForest", random_state=None, class_weight=None, n_jobs=1, max_depth=None,
                 n_estimators=50):
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._n_jobs = n_jobs
        self._class_weight = class_weight
        self._random_state = random_state
        self._type = classifier_type
        self._pool = dict()

    def get_set_of_subjects_in_pool(self):
        return set(self._pool.keys())

    def get_subject_classifier(self, subject_id):
        return self._pool[subject_id]

    def put_subject_classifier(self, subject_id, classifier):
        self._pool[subject_id] = classifier

    def train_subject_classifier(self, subject_id, X, y):
        new_classifier = SeparatedActivityClassifier(classifier_type=self._type, random_state=self._random_state,
                                                     class_weight=self._class_weight, n_jobs=self._n_jobs,
                                                     n_estimators=self._n_estimators,
                                                     max_depth=self._max_depth).fit(X, y)
        self.put_subject_classifier(subject_id, new_classifier)

    def find_best_existing_classifier(self, X, y, ignored_ids=None):
        scores = []

        ids_to_be_evaluated = self.get_set_of_subjects_in_pool()

        if ignored_ids:
            ids_to_be_evaluated -= set(ignored_ids)

        for subject_id in ids_to_be_evaluated:
            scores.append((self.get_subject_classifier(subject_id).score(X, y), subject_id))

        scores.sort()
        _, best_subject_id = scores.pop()
        return self.get_subject_classifier(best_subject_id)

    def mix_new_classifier_from_pool(self, X, y, ignored_ids=None):
        activity_accuracies = defaultdict(list)

        ids_to_be_evaluated = self.get_set_of_subjects_in_pool()

        if ignored_ids:
            ids_to_be_evaluated -= set(ignored_ids)

        for subject_id in ids_to_be_evaluated:
            subject_activity_accuracies = self.get_subject_classifier(subject_id).score_activities_separately(X, y)
            for activity in subject_activity_accuracies:
                this_activity_accuracy = subject_activity_accuracies[activity]
                activity_accuracies[activity].append((this_activity_accuracy, subject_id))

        new_individual = SeparatedActivityClassifier(classifier_type=self._type, random_state=self._random_state,
                                                     class_weight=self._class_weight)

        for activity in activity_accuracies:
            activity_accuracies[activity].sort()
            _, best_subject_id = activity_accuracies[activity].pop()
            best_activity_classifier = self.get_subject_classifier(best_subject_id).get_activity_classifier(activity)
            new_individual.set_activity_classifier(activity, best_activity_classifier)

        return new_individual
