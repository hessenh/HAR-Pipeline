from __future__ import print_function, division

from collections import Counter
from datetime import datetime

import numpy as np
import os
import configparser

from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier as Rfc
from sklearn.model_selection import train_test_split

from VagesHAR import VAGESHAR_ROOT
from VagesHAR.DataLoader import DataLoader as Dl
from VagesHAR.ClassifierPool import ClassifierPool
from VagesHAR.StatisticHelpers import generate_and_save_confusion_matrix, generate_and_save_statistics_json
from definitions import PROJECT_ROOT

label_to_number_dict = {
    "none": 0,
    "walking": 1,
    "running": 2,
    "shuffling": 3,
    "stairs (ascending)": 4,
    "stairs (descending)": 5,
    "standing": 6,
    "sitting": 7,
    "lying": 8,
    "transition": 9,
    "bending": 10,
    "picking": 11,
    "undefined": 12,
    "cycling (sit)": 13,
    "cycling (stand)": 14,
    "heel drop": 15,
    "vigorous activity": 16,
    "non-vigorous activity": 17,
    "Transport(sitting)": 18,
    "Commute(standing)": 19,
    "lying (prone)": 20,
    "lying (supine)": 21,
    "lying (left)": 22,
    "lying (right)": 23
}

number_to_label_dict = dict([(label_to_number_dict[l], l) for l in label_to_number_dict])


def get_eligible_subjects(data_set_root, codewords, ignored=None):
    if ignored is None:
        ignored = []
    actual_subjects = []

    for r, _, files in os.walk(data_set_root):
        if os.path.split(r)[1] in ignored:
            continue
        codeword_list = []
        for f in files:
            codeword_list.append(f[4:-4])

        if set(codewords) <= set(codeword_list):
            actual_subjects.append(r)

    subject_sensor_paths = []
    subject_label_paths = []
    for s_path in sorted(actual_subjects):
        s_id = s_path[-3:]
        subject_sensor_paths.append([os.path.join(s_path, s_id + "_" + k + ".csv") for k in codewords])
        subject_label_paths.append(os.path.join(s_path, s_id + "_labels.csv"))

    return subject_sensor_paths, subject_label_paths


def load_sensors(csv_files, data_loader):
    read_data = [data_loader.read_sensor_data(f, abs_vals=ABSOLUTE_VALUES) for f in csv_files]
    if len(read_data) == 1:
        return read_data[0]
    return np.hstack(read_data)


def load_labels(csv_file, data_loader):
    return data_loader.read_label_data(csv_file, RELABEL_DICT)


def remove_unwanted_activities(sensor_data, label_data, keep_set):
    indices_to_keep = [i for i, a in enumerate(label_data) if a in keep_set]

    sensor_data = sensor_data[indices_to_keep]
    label_data = label_data[indices_to_keep]
    return sensor_data, label_data


def remove_singletons(sensor_data, label_data):
    indices_to_keep = list(range(len(label_data)))

    if remove_singletons:
        c = Counter(label_data)
        data_ = list(label_data)
        for k in c:
            if c[k] == 1:
                indices_to_keep.remove(data_.index(k))

    sensor_data = sensor_data[indices_to_keep]
    label_data = label_data[indices_to_keep]
    return sensor_data, label_data


def get_and_make_subdirectories(sub_name, *dirs):
    new_dirs = [os.path.join(d, sub_name) for d in dirs]
    map(lambda x: os.makedirs(x) if not os.path.exists(x) else None, new_dirs)
    return new_dirs


if __name__ == "__main__":
    for i in range(5, 6):
        config_name = str(i) + "_sensors_nostairs"
        print(config_name)
        experiments = None
        config = configparser.ConfigParser()
        config.read_file(open(os.path.join(VAGESHAR_ROOT, "configs", "no_stairs", "%s.cfg" % config_name)))

        now = datetime.now()
        datetime_prefix = now.strftime("%Y%m%d_%H_%M")

        all_experiments_id = datetime_prefix + "_" + config_name

        top_statistics_folder = os.path.join(VAGESHAR_ROOT, "statistics", all_experiments_id)
        top_images_folder = os.path.join(VAGESHAR_ROOT, "images", all_experiments_id)

        if experiments is None:
            experiments = set(config.keys())
            experiments.discard("DEFAULT")

        for experiment_name in experiments:
            print(experiment_name)
            experiment_config = config[experiment_name]

            SEED = experiment_config.getint("seed")

            ACTIVITIES_TO_KEEP = {int(s) for s in experiment_config.get("activities_to_keep").split(", ")}
            RELABEL_DICT = dict({(int(k), int(v)) for k, v in
                                 [s.split(", ") for s in experiment_config.get("relabel_dict").split("; ")]})

            N_TREES = experiment_config.getint("n_trees", 50)
            MAX_DEPTH = experiment_config.getint("max_depth", None)
            N_JOBS = -1
            ABSOLUTE_VALUES = experiment_config.getboolean("absolute_values")

            ADAPTATION_TEST = experiment_config.getboolean("adaptation_test")
            BEST_INDIVIDUAL_TEST = experiment_config.getboolean("best_individual_test")
            ORDINARY_RFC_TEST = experiment_config.getboolean("global_model_test")

            config_train_data_sets = experiment_config.get("train_data_sets").split(", ")
            config_train_sensor_codewords = experiment_config.get("train_sensor_codewords").split("; ")
            config_train_sensor_codewords = [s.split(", ") for s in config_train_sensor_codewords]
            config_test_data_sets = experiment_config.get("test_data_sets").split(", ")
            config_test_sensor_codewords = experiment_config.get("test_sensor_codewords").split("; ")
            config_test_sensor_codewords = [s.split(", ") for s in config_test_sensor_codewords]

            test_statistics_folder, test_images_folder = get_and_make_subdirectories(experiment_name,
                                                                                     top_statistics_folder,
                                                                                     top_images_folder)

            sample_frequency = experiment_config.getfloat("sample_frequency")
            window_length = experiment_config.getfloat("window_length")
            train_overlap = experiment_config.getfloat("train_overlap")
            test_overlap = experiment_config.getfloat("test_overlap")

            train_data_loader = Dl(sample_frequency=sample_frequency, window_length=window_length,
                                   degree_of_overlap=train_overlap)
            test_data_loader = Dl(sample_frequency=sample_frequency, window_length=window_length,
                                  degree_of_overlap=test_overlap)

            all_sensor_data = []
            all_label_data = []

            train_sensor_paths, train_label_paths = [], []
            test_sensor_paths, test_label_paths = [], []

            for s, c in zip(config_train_data_sets, config_train_sensor_codewords):
                temp_sensor_paths, temp_label_paths = get_eligible_subjects(
                    data_set_root=os.path.join(PROJECT_ROOT, "DATA", s),
                    codewords=c
                )
                train_sensor_paths += temp_sensor_paths
                train_label_paths += temp_label_paths

            for s, c in zip(config_test_data_sets, config_test_sensor_codewords):
                temp_sensor_paths, temp_label_paths = get_eligible_subjects(
                    data_set_root=os.path.join(PROJECT_ROOT, "DATA", s),
                    codewords=c
                )
                test_sensor_paths += temp_sensor_paths
                test_label_paths += temp_label_paths

            train_subject_ids = [os.path.dirname(l)[-3:] for l in train_label_paths]
            test_subject_ids = [os.path.dirname(l)[-3:] for l in test_label_paths]

            print("Loading training set")
            train_sensors = Parallel(n_jobs=N_JOBS)(
                delayed(load_sensors)(s, train_data_loader) for s in train_sensor_paths)
            train_labels = Parallel(n_jobs=N_JOBS)(
                delayed(load_labels)(s, train_data_loader) for s in train_label_paths)

            print("Loading test set")
            test_sensors = Parallel(n_jobs=N_JOBS)(
                delayed(load_sensors)(s, test_data_loader) for s in test_sensor_paths)
            test_labels = Parallel(n_jobs=N_JOBS)(delayed(load_labels)(s, test_data_loader) for s in test_label_paths)

            # Cleaning up
            tmp_sensors = []
            tmp_labels = []

            for sens, labs in zip(train_sensors, train_labels):
                clean_sens, clean_labs = remove_unwanted_activities(sens, labs, ACTIVITIES_TO_KEEP)
                tmp_sensors.append(clean_sens)
                tmp_labels.append(clean_labs)

            train_sensors = tmp_sensors
            train_labels = tmp_labels

            tmp_sensors = []
            tmp_labels = []

            for sens, labs in zip(test_sensors, test_labels):
                clean_sens, clean_labs = remove_unwanted_activities(sens, labs, ACTIVITIES_TO_KEEP)
                tmp_sensors.append(clean_sens)
                tmp_labels.append(clean_labs)

            test_sensors = tmp_sensors
            test_labels = tmp_labels

            train_sensor_dict = dict([(name, sen) for name, sen in zip(train_subject_ids, train_sensors)])
            train_label_dict = dict([(name, lab) for name, lab in zip(train_subject_ids, train_labels)])

            test_sensor_dict = dict([(name, sen) for name, sen in zip(test_subject_ids, test_sensors)])
            test_label_dict = dict([(name, lab) for name, lab in zip(test_subject_ids, test_labels)])

            cp = ClassifierPool(classifier_type="RandomForest", random_state=SEED, class_weight="balanced", n_estimators=N_TREES,
                                max_depth=MAX_DEPTH, n_jobs=N_JOBS)

            if ADAPTATION_TEST or BEST_INDIVIDUAL_TEST:
                print("Creating pool")
                for subject_id in train_subject_ids:
                    print(subject_id)
                    tmp_subject_sensors, tmp_subject_labels = train_sensor_dict[subject_id], train_label_dict[
                        subject_id]
                    cp.train_subject_classifier(subject_id, tmp_subject_sensors, tmp_subject_labels)

            if ADAPTATION_TEST:
                sub_test_name = "adaptation"
                print(sub_test_name)
                sub_test_statistics, sub_test_images = get_and_make_subdirectories(sub_test_name,
                                                                                   test_statistics_folder,
                                                                                   test_images_folder)

                overall_y_test, overall_y_pred = [], []

                for subject_id in test_subject_ids:
                    print(subject_id)
                    tmp_subject_sensors, tmp_subject_labels = test_sensor_dict[subject_id], test_label_dict[subject_id]
                    tmp_subject_sensors, tmp_subject_labels = remove_singletons(tmp_subject_sensors, tmp_subject_labels)
                    X_train, X_test, y_train, y_test = train_test_split(
                        tmp_subject_sensors, tmp_subject_labels, test_size=0.8, random_state=SEED,
                        stratify=tmp_subject_labels)

                    adapted = cp.mix_new_classifier_from_pool(X_train, y_train, [subject_id])
                    y_pred = adapted.predict(X_test)

                    overall_y_test.append(y_test)
                    overall_y_pred.append(y_pred)

                    title = sub_test_name + " " + subject_id
                    png_path = os.path.join(sub_test_images, subject_id + ".png")
                    json_path = os.path.join(sub_test_statistics, subject_id + ".json")
                    generate_and_save_confusion_matrix(y_test, y_pred, number_to_label_dict, png_path, title=title)
                    generate_and_save_statistics_json(y_test, y_pred, number_to_label_dict, json_path)

                overall_y_test, overall_y_pred = np.hstack(overall_y_test), np.hstack(overall_y_pred)
                subject_id = "overall"

                title = sub_test_name + " " + subject_id
                png_path = os.path.join(sub_test_images, subject_id + ".png")
                json_path = os.path.join(sub_test_statistics, subject_id + ".json")
                generate_and_save_confusion_matrix(overall_y_test, overall_y_pred, number_to_label_dict, png_path,
                                                   title=title)
                generate_and_save_statistics_json(overall_y_test, overall_y_pred, number_to_label_dict, json_path)

            if BEST_INDIVIDUAL_TEST:
                sub_test_name = "best_individual"
                print(sub_test_name)
                sub_test_statistics, sub_test_images = get_and_make_subdirectories(sub_test_name,
                                                                                   test_statistics_folder,
                                                                                   test_images_folder)

                overall_y_test, overall_y_pred = [], []

                for subject_id in test_subject_ids:
                    print(subject_id)

                    tmp_subject_sensors, tmp_subject_labels = test_sensor_dict[subject_id], test_label_dict[subject_id]
                    tmp_subject_sensors, tmp_subject_labels = remove_singletons(tmp_subject_sensors, tmp_subject_labels)

                    X_train, X_test, y_train, y_test = train_test_split(
                        tmp_subject_sensors, tmp_subject_labels, test_size=0.8, random_state=SEED,
                        stratify=tmp_subject_labels)
                    best = cp.find_best_existing_classifier(X_train, y_train, [subject_id])
                    y_pred = best.predict(X_test)

                    overall_y_test.append(y_test)
                    overall_y_pred.append(y_pred)

                    title = sub_test_name + " " + subject_id
                    png_path = os.path.join(sub_test_images, subject_id + ".png")
                    json_path = os.path.join(sub_test_statistics, subject_id + ".json")
                    generate_and_save_confusion_matrix(y_test, y_pred, number_to_label_dict, png_path, title=title)
                    generate_and_save_statistics_json(y_test, y_pred, number_to_label_dict, json_path)

                overall_y_test, overall_y_pred = np.hstack(overall_y_test), np.hstack(overall_y_pred)
                subject_id = "overall"

                title = sub_test_name + " " + subject_id
                png_path = os.path.join(sub_test_images, subject_id + ".png")
                json_path = os.path.join(sub_test_statistics, subject_id + ".json")
                generate_and_save_confusion_matrix(overall_y_test, overall_y_pred, number_to_label_dict, png_path,
                                                   title=title)
                generate_and_save_statistics_json(overall_y_test, overall_y_pred, number_to_label_dict, json_path)

            if ORDINARY_RFC_TEST:
                sub_test_name = "general_population"
                print(sub_test_name)
                sub_test_statistics, sub_test_images = get_and_make_subdirectories(sub_test_name,
                                                                                   test_statistics_folder,
                                                                                   test_images_folder)

                overall_y_test, overall_y_pred = [], []

                for subject_id in test_subject_ids:
                    png_path = os.path.join(sub_test_images, subject_id + ".png")
                    json_path = os.path.join(sub_test_statistics, subject_id + ".json")
                    print(subject_id)
                    global_pool_ids = sorted(list(set(train_subject_ids) - {subject_id}))
                    X_train = np.vstack([train_sensor_dict[x] for x in global_pool_ids])
                    y_train = np.hstack([train_label_dict[x] for x in global_pool_ids])
                    my_forest = Rfc(n_estimators=N_TREES, random_state=SEED, max_depth=MAX_DEPTH, n_jobs=N_JOBS,
                                    class_weight="balanced")
                    my_forest.fit(X_train, y_train)
                    print("Testing")
                    X_test = test_sensor_dict[subject_id]
                    y_test = test_label_dict[subject_id]

                    y_pred = my_forest.predict(X_test)

                    overall_y_test.append(y_test)
                    overall_y_pred.append(y_pred)

                    title = sub_test_name + " " + subject_id
                    generate_and_save_confusion_matrix(y_test, y_pred, number_to_label_dict, png_path, title=title)
                    generate_and_save_statistics_json(y_test, y_pred, number_to_label_dict, json_path)

                overall_y_test, overall_y_pred = np.hstack(overall_y_test), np.hstack(overall_y_pred)
                subject_id = "overall"

                title = sub_test_name + " " + subject_id
                png_path = os.path.join(sub_test_images, subject_id + ".png")
                json_path = os.path.join(sub_test_statistics, subject_id + ".json")
                generate_and_save_confusion_matrix(overall_y_test, overall_y_pred, number_to_label_dict, png_path,
                                                   title=title)
                generate_and_save_statistics_json(overall_y_test, overall_y_pred, number_to_label_dict, json_path)
