import pandas as pd


def normalize(data_frame):
    means = data_frame.mean(axis=0)
    standard_deviations = data_frame.std(axis=0)
    for column, mean, standard_deviation in zip(data_frame, means, standard_deviations):
        data_frame[column] = (data_frame[column] - mean) / standard_deviation


def batch_normalize_files(glob_expression, output_folder):
    import glob
    import os

    file_paths = glob.glob(glob_expression)

    for i, path in enumerate(file_paths):
        print "Normalizing file", i + 1, "of", len(file_paths)

        data = pd.read_csv(path)
        normalize(data)

        file_name = path.split("/")[-1]
        subject_name = file_name.split("_")[0]

        head, tail = file_name.split(".")
        subject_output_folder = output_folder + "/" + subject_name
        output_path = subject_output_folder + "/" + "".join([head, "_normalized.", tail])

        if not os.path.exists(subject_output_folder):
            os.makedirs(subject_output_folder)

        data.to_csv(output_path)


if __name__ == "__main__":
    batch_normalize_files("./DATA/TRAINING/*/*Axivity*.csv", "./DATA/normalization_output")
