from training import train
from testing import test
import gc
import random
import os


def train_and_test(training_subject_list=None, statistics_save_path=None, normalize_sensor_data=False):
    train(subject_list=training_subject_list, normalize_sensor_data=normalize_sensor_data)
    gc.collect()
    test(subject_list=None, statistics_save_path=statistics_save_path, normalize_sensor_data=normalize_sensor_data)
    gc.collect()


if __name__ == "__main__":
    walk_of_training_folder = list(os.walk("./DATA/TRAINING"))
    all_training_subject_ids = walk_of_training_folder[0][1]
    for s_id in all_training_subject_ids:
        ids_for_session = [s_id]
        print("Selected subject for training:", s_id)
        output_file = "./single_subject_results/run_01/" + s_id + "_results.json"
        if not os.path.exists(output_file):
            train_and_test(training_subject_list=ids_for_session, statistics_save_path=output_file)

if __name__ == "__main__":
    walk_of_training_folder = list(os.walk("./DATA/TRAINING"))
    all_training_subject_ids = walk_of_training_folder[0][1]
    for i in range(len(all_training_subject_ids), 0, -1):
        ids_for_session = random.sample(all_training_subject_ids, i)
        print("Selected subjects for training:", ids_for_session)
        with open("./graceful_degradation_results/subjects.log", "a") as f:
            f.write(str(i) + ": " + str(ids_for_session) + "\n")
        output_file = "./graceful_degradation_results/" + format(i, "02") + "_training_subjects.json"
        train_and_test(training_subject_list=ids_for_session, statistics_save_path=output_file)


"""
if __name__ == "__main__":
    train_and_test(statistics_save_path="./normalization_lowerback_results/normalized.json", normalize_sensor_data=True)
    train_and_test(statistics_save_path="./normalization_lowerback_results/unnormalized.json", normalize_sensor_data=False)
"""