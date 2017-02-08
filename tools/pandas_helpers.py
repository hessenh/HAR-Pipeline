import pandas as pd


def write_selected_columns_to_file(data_frame, columns, file_path, with_header=False, with_index=False, keep_rate=1,
                                   keep_only_complete_windows=False):
    max_index_to_keep = -1
    if keep_only_complete_windows:
        max_index_to_keep = data_frame.shape[0] // keep_rate * keep_rate
    data_frame_with_selected_columns = data_frame[columns][:max_index_to_keep:keep_rate]
    data_frame_with_selected_columns.to_csv(file_path, header=with_header, index=with_index)


def average_columns_and_write_to_file(data_frame, columns, file_path, with_header=False, with_index=False,
                                      window_size=100, average_only_complete_windows=True, float_format=None):
    data_frame_with_selected_columns = data_frame[columns]
    max_index_to_keep = data_frame_length = data_frame.shape[0]
    if average_only_complete_windows:
        max_index_to_keep = (data_frame_length // window_size) * window_size

    averages = []
    window_start = 0

    while window_start < max_index_to_keep:
        if not window_start // window_size % 100000:
            print window_start
        window_end = min(window_start + window_size, max_index_to_keep)
        means = data_frame_with_selected_columns[window_start:window_end].mean()
        averages.append(means.to_frame().transpose())
        window_start += window_size

    result = pd.concat(averages)
    result.to_csv(file_path, header=with_header, index=with_index, float_format=float_format)
