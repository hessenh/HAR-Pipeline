from __future__ import print_function
from training import train
from testing import test
import gc
import random
import os


def train_and_test(training_subject_list=None, statistics_save_path=None, normalize_sensor_data=False,
                   testing_subjects_list=None, retrain=True, testing_subjects_folder=None):
    if retrain:
        train(subject_list=training_subject_list, normalize_sensor_data=normalize_sensor_data)
        gc.collect()
    test(subject_list=testing_subjects_list, statistics_save_path=statistics_save_path,
         normalize_sensor_data=normalize_sensor_data, subjects_folder=testing_subjects_folder)
    gc.collect()


def train_on_single_subject(runs=5):
    walk_of_training_folder = list(os.walk("./DATA/TRAINING"))
    all_training_subject_ids = walk_of_training_folder[0][1]
    root_folder_path = os.path.join(".", "statistics", "single_subject_results")

    for i in range(runs):
        run_folder_name = "run_" + format(i, "02")
        run_folder_path = os.path.join(root_folder_path, run_folder_name)
        if not os.path.exists(run_folder_path):
            os.makedirs(run_folder_path)
        for s_id in all_training_subject_ids:
            ids_for_session = [s_id]
            print("Selected subject for training:", s_id)
            output_file = os.path.join(run_folder_path, s_id + "_results.json")
            if not os.path.exists(output_file):
                train_and_test(training_subject_list=ids_for_session, statistics_save_path=output_file)


def graceful_degradation(rounds=10):
    walk_of_training_folder = list(os.walk("./DATA/TRAINING"))
    all_training_subject_ids = walk_of_training_folder[0][1]
    for i in range(rounds):
        print("run" + format(i, "02"))
        round_folder = os.path.join("./statistics/graceful_degradation_results", "run" + format(i, "02"))
        if not os.path.exists(round_folder):
            os.makedirs(round_folder)
        for i in range(len(all_training_subject_ids), 0, -1):
            ids_for_session = random.sample(all_training_subject_ids, i)
            output_file = os.path.join(round_folder, format(i, "02") + "_training_subjects.json")
            if not os.path.exists(output_file):
                print("Selected", i, "subjects for training:", ids_for_session)
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
    root_folder = "./statistics/range_normalization"
    upper_folder = "./DATA/upperback_out_of_lab"
    lower_folder = "./DATA/lowerback_out_of_lab"

    for i in range(3):
        round_folder = os.path.join(root_folder, "run" + format(i, "02"))

        upper_all_name = "all_upper.json"
        lower_name = "lower.json"

        upper_statistics = os.path.join(round_folder, upper_all_name)
        lower_statistics = os.path.join(round_folder, lower_name)

        if not os.path.exists(round_folder):
            os.makedirs(round_folder)

        if not os.path.exists(upper_statistics):
            train_and_test(retrain=True, statistics_save_path=upper_statistics, testing_subjects_folder=upper_folder,
                           normalize_sensor_data=True)
        if not os.path.exists(lower_statistics):
            train_and_test(retrain=False, statistics_save_path=lower_statistics, testing_subjects_folder=lower_folder,
                           normalize_sensor_data=True)


def test_with_all_individual_subjects():
    walk_of_testing_folder = list(os.walk("./DATA/TESTING"))
    all_testing_subject_ids = walk_of_testing_folder[0][1]
    for i, subject in enumerate(all_testing_subject_ids):
        train_and_test(testing_subjects_list=[subject], retrain=i == 0,
                       statistics_save_path="./individual_upperback_testing/" + subject + ".json")


def test_upper_against_lower():
    root_folder = "./upper_against_lower_only_unduplicated_subjects"
    walk_of_upperback_folder = list(os.walk("./DATA/upperback_out_of_lab"))
    walk_of_lowerback_folder = list(os.walk("./DATA/lowerback_out_of_lab"))
    all_lowerback_ids = walk_of_lowerback_folder[0][1]
    all_upperback_ids = walk_of_upperback_folder[0][1]

    unique_ids = set(all_upperback_ids) - set(all_lowerback_ids)

    for i in range(3):
        round_folder = os.path.join(root_folder, "run" + format(i, "02"))
        upper_name = "upper.json"
        lower_name = "lower.json"
        upper_statistics = os.path.join(round_folder, upper_name)
        lower_statistics = os.path.join(round_folder, lower_name)
        if not os.path.exists(round_folder):
            os.makedirs(round_folder)
        if not os.path.exists(lower_statistics):
            train_and_test(retrain=True, statistics_save_path=upper_statistics,
                           testing_subjects_folder="./DATA/upperback_out_of_lab",
                           testing_subjects_list=unique_ids)
            """
            train_and_test(retrain=False, statistics_save_path=lower_statistics,
                           testing_subjects_folder="./DATA/lowerback_out_of_lab",
                           testing_subjects_list=all_lowerback_ids)
            """


def leave_one_out(training_folder=os.path.join(".", "DATA", "TRAINING")):
    # Before running this, all testing data must be added do DATA/TRAINING
    root_folder = "./statistics/leave_one_child_out"
    walk_of_training_folder = list(os.walk(training_folder))
    all_test_ids = set(walk_of_training_folder[0][1])
    all_in_lab_ids = {_ for _ in all_test_ids if "A" in _}
    all_out_of_lab_ids = all_test_ids - all_in_lab_ids

    for j in range(3):
        run_folder = "run_" + format(j, "02")

        # # Testing using only in-lab data as training
        # sub_folder = "in_lab"
        # statistics_folder = os.path.join(root_folder, sub_folder, run_folder)
        # if not os.path.exists(statistics_folder):
        #     os.makedirs(statistics_folder)
        #
        # for i, test_subject in enumerate(all_out_of_lab_ids):
        #     session_training_subjects = all_in_lab_ids.copy()
        #     current_statistics = os.path.join(statistics_folder, test_subject + ".json")
        #     retrain_only_for_first_subject = i == 0
        #     train_and_test(training_subject_list=session_training_subjects, statistics_save_path=current_statistics,
        #                    testing_subjects_list=[test_subject], retrain=retrain_only_for_first_subject,
        #                    testing_subjects_folder=training_folder)


        # # Testing with mix-in of in-lab
        # sub_folder = "mix"
        # statistics_folder = os.path.join(root_folder, sub_folder, run_folder)
        # if not os.path.exists(statistics_folder):
        #     os.makedirs(statistics_folder)
        #
        # for test_subject in all_out_of_lab_ids:
        #     session_training_subjects = all_test_ids.copy()
        #     session_training_subjects.remove(test_subject)
        #     current_statistics = os.path.join(statistics_folder, test_subject + ".json")
        #     train_and_test(training_subject_list=session_training_subjects, statistics_save_path=current_statistics,
        #                    testing_subjects_list=[test_subject], retrain=True, testing_subjects_folder=training_folder)

        # Only using out-of-lab
        sub_folder = "out_of_lab"
        statistics_folder = os.path.join(root_folder, sub_folder, run_folder)
        if not os.path.exists(statistics_folder):
            os.makedirs(statistics_folder)

        for test_subject in all_out_of_lab_ids:
            session_training_subjects = all_out_of_lab_ids.copy()
            session_training_subjects.remove(test_subject)
            current_statistics = os.path.join(statistics_folder, test_subject + ".json")
            train_and_test(training_subject_list=session_training_subjects,
                           statistics_save_path=current_statistics,
                           testing_subjects_list=[test_subject], retrain=True, testing_subjects_folder=training_folder)


def test_all_four_out_of_lab_sets(badly_synced=False, normalize=False):
    root_folder = os.path.join(".", "statistics", "existing_system")
    badly_synced_addition = "badly_synced" if badly_synced else ""
    if badly_synced:
        root_folder += "_badly_synced"
    upper_folder = os.path.join(".", "DATA", badly_synced_addition, "upperback_out_of_lab")
    lower_folder = os.path.join(".", "DATA", badly_synced_addition, "lowerback_out_of_lab")

    upper_folder_walk = list(os.walk(upper_folder))
    lower_folder_walk = list(os.walk(lower_folder))
    lower_006_012 = lower_folder_walk[0][1]
    upper_001_012 = upper_folder_walk[0][1]
    upper_001_005 = set(upper_001_012) - set(lower_006_012)
    upper_006_012 = lower_006_012

    for i in range(3):
        round_folder = os.path.join(root_folder, "run" + format(i, "02"))

        upper_001_012_name = "upper_001_012.json"
        lower_006_012_name = "lower_006_012.json"
        upper_001_005_name = "upper_001_005.json"
        upper_006_012_name = "upper_006_012.json"

        upper_statistics = os.path.join(round_folder, upper_001_012_name)
        upper_not_lower_statistics = os.path.join(round_folder, upper_001_005_name)
        upper_also_lower_statistics = os.path.join(round_folder, upper_006_012_name)
        lower_statistics = os.path.join(round_folder, lower_006_012_name)

        if not os.path.exists(round_folder):
            os.makedirs(round_folder)
            train_and_test(retrain=True, statistics_save_path=upper_statistics, testing_subjects_folder=upper_folder,
                           normalize_sensor_data=normalize)
            train_and_test(retrain=False, statistics_save_path=upper_not_lower_statistics,
                           testing_subjects_folder=upper_folder, testing_subjects_list=upper_001_005,
                           normalize_sensor_data=normalize)
            train_and_test(retrain=False, statistics_save_path=upper_also_lower_statistics,
                           testing_subjects_folder=upper_folder, testing_subjects_list=upper_006_012,
                           normalize_sensor_data=normalize)
            train_and_test(retrain=False, statistics_save_path=lower_statistics, testing_subjects_folder=lower_folder,
                           normalize_sensor_data=normalize)


if __name__ == "__main__":
    leave_one_out()
