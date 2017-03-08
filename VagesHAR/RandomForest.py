from __future__ import print_function, division

import numpy as np
import os
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier as Rfc
from sklearn.model_selection import train_test_split

from DataLoader import DataLoader as Dl
from VagesHAR.ClassifierPool import ClassifierPool
from VagesHAR.statistics_stuff import generate_and_save_confusion_matrix
from statistics_stuff import generate_and_save_statistics_json

from datetime import datetime

seed = 0

ADAPTATION_TEST = True
BEST_INDIVIDUAL_TEST = True
ORDINARY_RFC_TEST = True

n_trees = 50
max_depth = None
n_jobs = -1
abs_vals = True
data_set = "TRAINING"
max_subjects = float("inf")

activities_to_keep = {1, 2, 4, 5, 6, 7, 8, 10, 13, 14}

relabel_dict = {11: 10}

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
    "Commute(standing)": 19
}

number_to_label_dict = dict([(label_to_number_dict[l], l) for l in label_to_number_dict])


def load_sensors(s_id, dl):
    thigh_file = os.path.join("..", "DATA", data_set, s_id, s_id + "_Axivity_THIGH_Right.csv")
    back_file = os.path.join("..", "DATA", data_set, s_id, s_id + "_Axivity_BACK_Back.csv")

    thigh_data = dl.read_sensor_data(thigh_file, abs_vals=abs_vals)
    back_data = dl.read_sensor_data(back_file, abs_vals=abs_vals)
    return np.hstack([thigh_data, back_data])


def load_labels(s_id, dl):
    label_file = os.path.join("..", "DATA", data_set, s_id, s_id + "_GoPro_LAB_All.csv")

    return dl.read_label_data(label_file, relabel_dict)


def remove_unwanted_activities(sensor_data, label_data, keep_set):
    indices_to_keep = [i for i, a in enumerate(label_data) if a in keep_set]
    sensor_data = sensor_data[indices_to_keep]
    label_data = label_data[indices_to_keep]
    return sensor_data, label_data


def get_and_make_subdirectories(sub_name, *dirs):
    new_dirs = [os.path.join(d, sub_name) for d in dirs]
    map(lambda x: os.makedirs(x) if not os.path.exists(x) else None, new_dirs)
    return new_dirs


if __name__ == "__main__":
    now = datetime.now()

    datetime_prefix = now.strftime("%Y%m%d_%H_%M_")

    top_statistics_folder = os.path.join(".", "statistics")
    top_images_folder = os.path.join(".", "images")

    test_name = "three_approaches"
    test_id = datetime_prefix + test_name

    test_statistics_folder, test_images_folder = get_and_make_subdirectories(test_id, top_statistics_folder,
                                                                             top_images_folder)
    sf = 100
    wl = 2.0
    overlap_dl = Dl(sample_frequency=sf, window_length=wl, degree_of_overlap=0.9)
    separate_dl = Dl(sample_frequency=sf, window_length=wl, degree_of_overlap=0.0)

    all_sensor_data = []
    all_label_data = []

    out_of_lab = data_set == "TRAINING"

    if out_of_lab:
        r_start, r_stop = 6, 23
        ignored = [7]
    else:
        r_start, r_stop = 1, 24
        ignored = [17]

    subject_names = []

    for i in range(r_start, r_stop):
        if len(subject_names) >= max_subjects:
            break
        if i in ignored:
            continue
        if out_of_lab:
            subject_names.append("{0:0>3}".format(i))
        else:
            subject_names.append("{0:0>2}".format(i) + "A")

    print("Loading sensors")
    overlap_sensors = Parallel(n_jobs=n_jobs)(delayed(load_sensors)(s, overlap_dl) for s in subject_names)
    separate_sensors = Parallel(n_jobs=n_jobs)(delayed(load_sensors)(s, separate_dl) for s in subject_names)

    print("Loading labels")
    overlap_labels = Parallel(n_jobs=n_jobs)(delayed(load_labels)(s, overlap_dl) for s in subject_names)
    separate_labels = Parallel(n_jobs=n_jobs)(delayed(load_labels)(s, separate_dl) for s in subject_names)

    tmp_sensors = []
    tmp_labels = []

    for sens, labs in zip(overlap_sensors, overlap_labels):
        clean_sens, clean_labs = remove_unwanted_activities(sens, labs, activities_to_keep)
        tmp_sensors.append(clean_sens)
        tmp_labels.append(clean_labs)

    overlap_sensors = tmp_sensors
    overlap_labels = tmp_labels

    tmp_sensors = []
    tmp_labels = []

    for sens, labs in zip(separate_sensors, separate_labels):
        clean_sens, clean_labs = remove_unwanted_activities(sens, labs, activities_to_keep)
        tmp_sensors.append(clean_sens)
        tmp_labels.append(clean_labs)

    separate_sensors = tmp_sensors
    separate_labels = tmp_labels

    overlap_sensor_dict = dict([(name, sen) for name, sen in zip(subject_names, overlap_sensors)])
    separate_sensor_dict = dict([(name, sen) for name, sen in zip(subject_names, separate_sensors)])
    overlap_label_dict = dict([(name, lab) for name, lab in zip(subject_names, overlap_labels)])
    separate_label_dict = dict([(name, lab) for name, lab in zip(subject_names, separate_labels)])

    cp = ClassifierPool(type="RandomForest", random_state=seed, class_weight="balanced", n_estimators=n_trees,
                        max_depth=max_depth, n_jobs=n_jobs)

    if ADAPTATION_TEST or BEST_INDIVIDUAL_TEST:
        print("Creating pool")
        for s_id in subject_names:
            print(s_id)
            sensor_data, label_data = overlap_sensor_dict[s_id], overlap_label_dict[s_id]
            cp.train_one(s_id, sensor_data, label_data)

    if ADAPTATION_TEST:
        sub_test_name = "adaptation"
        print(sub_test_name)
        sub_test_statistics, sub_test_images = get_and_make_subdirectories(sub_test_name, test_statistics_folder,
                                                                           test_images_folder)
        for s_id in subject_names:
            print(s_id)
            png_path = os.path.join(sub_test_statistics, s_id + ".png")
            json_path = os.path.join(sub_test_images, s_id + ".json")
            sensor_data, label_data = separate_sensor_dict[s_id], separate_label_dict[s_id]
            X_train, X_test, y_train, y_test = train_test_split(
                sensor_data, label_data, test_size=0.8, random_state=seed)
            adapted = cp.adapt(X_train, y_train, [s_id])
            y_pred = adapted.predict(X_test)
            title = sub_test_name + " " + s_id
            generate_and_save_confusion_matrix(y_test, y_pred, number_to_label_dict, png_path, title=title)
            generate_and_save_statistics_json(y_test, y_pred, number_to_label_dict, json_path)

    if BEST_INDIVIDUAL_TEST:
        sub_test_name = "best_individual"
        print(sub_test_name)
        sub_test_statistics, sub_test_images = get_and_make_subdirectories(sub_test_name, test_statistics_folder,
                                                                           test_images_folder)

        for s_id in subject_names:
            print(s_id)
            png_path = os.path.join(sub_test_statistics, s_id + ".png")
            json_path = os.path.join(sub_test_images, s_id + ".json")
            sensor_data, label_data = separate_sensor_dict[s_id], separate_label_dict[s_id]
            X_train, X_test, y_train, y_test = train_test_split(
                sensor_data, label_data, test_size=0.8, random_state=seed)
            best = cp.find_best_individual(X_train, y_train, [s_id])
            y_pred = best.predict(X_test)
            title = sub_test_name + " " + s_id
            generate_and_save_confusion_matrix(y_test, y_pred, number_to_label_dict, png_path, title=title)
            generate_and_save_statistics_json(y_test, y_pred, number_to_label_dict, json_path)

    if ORDINARY_RFC_TEST:
        sub_test_name = "general_population"
        print(sub_test_name)
        sub_test_statistics, sub_test_images = get_and_make_subdirectories(sub_test_name, test_statistics_folder,
                                                                           test_images_folder)
        for s_id in subject_names:
            png_path = os.path.join(sub_test_statistics, s_id + ".png")
            json_path = os.path.join(sub_test_images, s_id + ".json")
            print(s_id)
            train_subject_ids = sorted(list(set(subject_names) - {s_id}))
            X_train = np.vstack([overlap_sensor_dict[x] for x in train_subject_ids])
            y_train = np.hstack([overlap_label_dict[x] for x in train_subject_ids])
            my_forest = Rfc(n_estimators=n_trees, random_state=seed, max_depth=max_depth, n_jobs=n_jobs,
                            class_weight="balanced")
            my_forest.fit(X_train, y_train)
            print("Testing")
            X_test = separate_sensor_dict[s_id]
            y_test = separate_label_dict[s_id]

            y_pred = my_forest.predict(X_test)

            title = sub_test_name + " " + s_id
            generate_and_save_confusion_matrix(y_test, y_pred, number_to_label_dict, png_path, title=title)
            generate_and_save_statistics_json(y_test, y_pred, number_to_label_dict, json_path)
