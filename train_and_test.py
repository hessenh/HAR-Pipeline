from training import train
from testing import test
import gc
import random
import os


def train_and_test(training_subject_list=None, statistics_save_path=None, normalize_sensor_data=False,
                   testing_subjects_list=None, retrain=True):
    if retrain:
        train(subject_list=training_subject_list, normalize_sensor_data=normalize_sensor_data)
        gc.collect()
    test(subject_list=testing_subjects_list, statistics_save_path=statistics_save_path,
         normalize_sensor_data=normalize_sensor_data)
    gc.collect()


def train_on_single_subject():
    walk_of_training_folder = list(os.walk("./DATA/TRAINING"))
    all_training_subject_ids = walk_of_training_folder[0][1]
    for s_id in all_training_subject_ids:
        ids_for_session = [s_id]
        print("Selected subject for training:", s_id)
        output_file = "./single_subject_results/run_01/" + s_id + "_results.json"
        if not os.path.exists(output_file):
            train_and_test(training_subject_list=ids_for_session, statistics_save_path=output_file)


def graceful_degradation():
    walk_of_training_folder = list(os.walk("./DATA/TRAINING"))
    all_training_subject_ids = walk_of_training_folder[0][1]
    for i in range(3):
        round_folder = os.path.join("./graceful_degradation_results", "run" + format(i, "02"))
        if not os.path.exists(round_folder):
            os.makedirs(round_folder)
        for i in range(len(all_training_subject_ids), 0, -1):
            ids_for_session = random.sample(all_training_subject_ids, i)
            print("Selected subjects for training:", ids_for_session)
            output_file = os.path.join(round_folder, format(i, "02") + "_training_subjects.json")
            if not os.path.exists(output_file):
                train_and_test(training_subject_list=ids_for_session, statistics_save_path=output_file)
                with open(os.path.join(round_folder, "subjects.log"), "a") as f:
                    f.write(str(i) + ": " + str(ids_for_session) + "\n")

"""
def graceful_degradation():
    walk_of_training_folder = list(os.walk("./DATA/TRAINING"))
    all_training_subject_ids = walk_of_training_folder[0][1]
    for i in range(3):
        round_folder = os.path.join("./inlab_graceful_degradation_results", "run" + format(i, "02"))
        if not os.path.exists(round_folder):
            os.makedirs(round_folder)
        for i in range(len(all_training_subject_ids), 0, -1):
            ids_for_session = random.sample(all_training_subject_ids, i)
            print("Selected subjects for training:", ids_for_session)
            output_file = os.path.join(round_folder, format(i, "02") + "_training_subjects.json")
            if not os.path.exists(output_file):
                train_and_test(training_subject_list=ids_for_session, statistics_save_path=output_file)
                with open(os.path.join(round_folder, "subjects.log"), "a") as f:
                    f.write(str(i) + ": " + str(ids_for_session) + "\n")
"""


def normalization_comparison():
    train_and_test(statistics_save_path="./normalization_lowerback_results/normalized.json", normalize_sensor_data=True)
    train_and_test(statistics_save_path="./normalization_lowerback_results/unnormalized.json",
                   normalize_sensor_data=False)


def test_with_all_individual_subjects():
    walk_of_testing_folder = list(os.walk("./DATA/TESTING"))
    all_testing_subject_ids = walk_of_testing_folder[0][1]
    for i, subject in enumerate(all_testing_subject_ids):
        train_and_test(testing_subjects_list=[subject], retrain=i == 0,
                       statistics_save_path="./individual_upperback_testing/" + subject + ".json")

if __name__ == "__main__":
    graceful_degradation()