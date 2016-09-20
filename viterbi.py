import numpy as np
import TRAINING_VARIABLES
import pickle
import pandas as pd

V = TRAINING_VARIABLES.VARS()


def main():
    print run_viterbi()


def run_viterbi():
    start_probability = generate_start_probability()
    transition_dictionary = load_obj(V.VITERBI_TRANSITION_DICTIONARY_PATH)
    emission_probability = generate_emission_probability(V.VITERBI_PREDICTION_PATH_TESTING, start_probability)

    VITERBI_PATH = [{}]
    path = {}
    states = V.VITERBI_STATES

    for y in range(len(states)):
        VITERBI_PATH[0][states[y]] = start_probability[states[y]] + emission_probability[0][y]
        path[states[y]] = [states[y]]

    for t in range(1, len(emission_probability)):
        VITERBI_PATH.append({})
        newPath = {}
        for y in range(0, len(states)):
            (prob, state) = max(
                (VITERBI_PATH[t - 1][y0] + transition_dictionary[y0][states[y]] + emission_probability[t][y], y0) for y0
                in states)
            newPath[states[y]] = path[state] + [states[y]]
            VITERBI_PATH[t][states[y]] = prob

        path = newPath

    n = len(emission_probability) - 1
    (prob, state) = max((VITERBI_PATH[n][y], y) for y in states)
    end_state = state

    return path[state]


def load_obj(path):
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)


def generate_emission_probability(observation_path, start_probability):
    observations = pd.read_csv(observation_path, header=None, sep='\,', engine='python').as_matrix()

    states = V.VITERBI_STATES
    emission_probability = np.copy(observations)
    # For each observation
    for j in range(0, len(observations)):  # self.observations_length
        # Do it for evert state
        for i in range(0, len(states)):
            emission_probability[j][i] = observations[j][i] / np.exp(start_probability[states[i]])
        s = np.sum(emission_probability[j])
        for i in range(0, len(states)):
            emission_probability[j][i] = emission_probability[j][i] / s

        for i in range(0, len(states)):
            emission_probability[j][i] = np.log(emission_probability[j][i])
    return emission_probability


def generate_start_probability():
    start_probability = {}
    for i in range(len(V.VITERBI_STATES)):
        start_probability[V.VITERBI_STATES[i]] = np.log(1.0 / len(V.ACTIVITIES))
    return start_probability


def generate_transition_dictionary(transition):
    states = V.VITERBI_STATES
    # Create structure of matrix
    transition_probability = {}
    for i in range(0, len(states)):
        temp_dict = {}
        for j in range(0, len(states)):
            temp_dict[states[j]] = transition[i][j]  # /np.sum(transition[i])
        transition_probability[states[i]] = temp_dict

    # Divide the values in the matrix by the length of the observation
    for d in transition_probability:
        for key, value in transition_probability[d].items():
            transition_probability[d][key] = np.log(value)  # / labelsCount)

    return transition_probability


def generate_transition_matrix(matrix_type):
    transition_matrix = None

    if matrix_type == 'BW':
        transition_matrix = transition_matrix_baum_welch()

    if matrix_type == 'standard':
        transition_matrix = transition_matrix_standard()

    if matrix_type == 'combination':
        trans1 = transition_matrix_baum_welch()
        trans2 = transition_matrix_standard()
        transition_matrix = (trans1 + trans2) / 2

    transition_dictionary = generate_transition_dictionary(transition_matrix)
    with open(V.VITERBI_TRANSITION_DICTIONARY_PATH + '.pkl', 'wb') as f:
        pickle.dump(transition_dictionary, f, pickle.HIGHEST_PROTOCOL)


def transition_matrix_baum_welch():
    number_activities = len(V.ACTIVITIES)
    iterations = V.VITERBI_BAUM_WELCH_ITERATIONS

    predictions_path = V.VITERBI_PREDICTION_PATH_TRAINING

    predictions = pd.read_csv(predictions_path, header=None, sep='\,', engine='python').as_matrix()

    transition_matrix = np.zeros((number_activities, number_activities))
    transition_matrix = transition_matrix + 1.0 / number_activities

    # Convert predictions to log-values
    predictions = np.log(predictions)

    for i in range(0, iterations):
        transition_matrix = np.log(transition_matrix)

        forward_prob = forward(predictions, transition_matrix, number_activities)
        backward_prob = backward(predictions, transition_matrix, number_activities)

        for act1 in range(0, number_activities):
            for act2 in range(0, number_activities):
                a = 0
                b = 0
                for i in range(0, len(predictions) - 2):  # loop through seq

                    a = a + np.exp(forward_prob[i][act1] + backward_prob[i][act1] - backward_prob[0][act1])
                    b = b + np.exp(forward_prob[i][act1] + backward_prob[i + 1][act2] + transition_matrix[act1][act2] +
                                   predictions[i + 1][act2] - backward_prob[0][act1])

                a = backward_prob[0][act1] + np.log(a)

                b = backward_prob[0][act1] + np.log(b)

                transition_matrix[act1][act2] = np.exp(b - a)

        for i in range(0, number_activities):
            transition_matrix[i] = transition_matrix[i] / sum(transition_matrix[i])

    return transition_matrix


def transition_matrix_standard():
    number_activities = len(V.ACTIVITIES)

    actual_path = V.VITERBI_ACTUAL_PATH_TRAINING
    actual = pd.read_csv(actual_path, header=None, sep='\,', engine='python').as_matrix()

    transition_matrix = np.zeros((number_activities, number_activities))

    for i in range(0, len(actual) - 1):
        this_activity = np.argmax(actual[i])
        next_activity = np.argmax(actual[i + 1])

        transition_matrix[this_activity][next_activity] += 1

    # Normalizing
    row_sums = transition_matrix.sum(axis=1)
    transition_matrix = transition_matrix / row_sums[:, np.newaxis]

    # for i in range(0,number_activities):
    #	a = transition_matrix[i]/(np.sum(transition_matrix[i])*1.0)
    #	transition_matrix[i] = a.tolist()
    return transition_matrix


def forward(predictions_log, transition_log, number_activities):
    forward_prob = np.zeros((len(predictions_log), number_activities))

    forward_prob[0] = np.log(1.0 / number_activities)
    for t in range(1, len(forward_prob)):
        for act in range(0, number_activities):
            maxProb = 0
            maxProbIndex = 0
            prob = 0
            for prev_act in range(0, number_activities):
                if forward_prob[t - 1][prev_act] + transition_log[prev_act][act] + predictions_log[t][act] < maxProb:
                    maxProb = forward_prob[t - 1][prev_act] + transition_log[prev_act][act] + predictions_log[t][act]
                    maxProbIndex = prev_act

            for prev_act in range(0, number_activities):
                prob = prob + np.exp(
                    forward_prob[t - 1][prev_act] + transition_log[prev_act][act] + predictions_log[t][act] - maxProb)

            prob_t = maxProb + np.log(prob)

            forward_prob[t][act] = prob_t

    return forward_prob


def backward(predictions_log, transition_log, number_activities):
    backward_prob = np.zeros((len(predictions_log), number_activities))
    backward_prob[len(backward_prob) - 1] = np.log(1.0 / number_activities)
    for t in range(len(backward_prob) - 2, -1, -1):
        for act in range(0, number_activities):
            maxProb = 0
            maxProbIndex = 0
            prob = 0
            for next_act in range(0, number_activities):
                if backward_prob[t + 1][next_act] + transition_log[act][next_act] + predictions_log[t][act] < maxProb:
                    maxProb = backward_prob[t + 1][next_act] + transition_log[act][next_act] + predictions_log[t][act]
                    maxProbIndex = next_act
            for next_act in range(0, number_activities):
                prob = prob + np.exp(
                    backward_prob[t + 1][next_act] + transition_log[act][next_act] + predictions_log[t][act] - maxProb)

            prob_t = maxProb + np.log(prob)

            backward_prob[t][act] = prob_t

    return backward_prob


if __name__ == "__main__":
    main()
