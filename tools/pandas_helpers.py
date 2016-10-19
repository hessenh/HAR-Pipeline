def add_to_column_in_data_frame(data_frame, column_name, added):
    data_frame[column_name] = data_frame.apply(lambda row: (row[column_name] + added), axis=1)


def subtract_from_column_in_data_frame(data_frame, column_name, subtracted):
    add_to_column_in_data_frame(data_frame, column_name, -subtracted)


def write_selected_columns_to_file(data_frame, columns, file_path, with_header=False, with_index=False):
    data_frame_with_selected_columns = data_frame[columns]
    data_frame_with_selected_columns.to_csv(file_path, header=with_header, index=with_index)