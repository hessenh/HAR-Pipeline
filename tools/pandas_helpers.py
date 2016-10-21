def write_selected_columns_to_file(data_frame, columns, file_path, with_header=False, with_index=False):
    data_frame_with_selected_columns = data_frame[columns]
    data_frame_with_selected_columns.to_csv(file_path, header=with_header, index=with_index)
