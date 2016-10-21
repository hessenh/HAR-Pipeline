import pandas as pd


def normalize(data_frame):
    means = data_frame.mean(axis=0)
    standard_deviations = data_frame.std(axis=0)
    for column, mean, standard_deviation in zip(data_frame, means, standard_deviations):
        data_frame[column] = (data_frame[column] - mean) / standard_deviation


if __name__ == "__main__":
    a = pd.read_csv("./DATA/TESTING/004/004_Axivity_BACK_Back.csv", header=None)
    normalize(a)
    a.to_csv("./004_Axivity_BACK_Back_normalized.csv", header=False, index=False)
