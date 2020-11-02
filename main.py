from DLC_functions import Head_direction

def convert_to_csv(H5_path, desired_path):
    """
    H5_path --> h5 files
    desired path to download to, needs to end with .csv
    """
    loaded_path = pd.read_hdf(defensive_H5)
    obstructive_title = loaded_path.keys()[0][0]
    extracted_tracks = loaded_path[obstructive_title]
    body_parts_list = list(extracted_tracks.columns.levels[0])
    DataFrame = pd.DataFrame(data=(range(0,len(extracted_tracks[extracted_tracks.keys()[0][0]][extracted_tracks.keys()[0][1]]))), columns=['FRAME'])
    for column in enumerate(extracted_tracks):
        column_data = []
        column_title = []
        if column[1][1] == 'likelihood':
            pass
        else:
            column_data.append(extracted_tracks[str(column[1][0])][str(column[1][1])])
            column_title = str(column[1][0] + column[1][1])
            DataFrame[column_title] = column_data[0]
    return DataFrame.to_csv(desired_path)

def run_dlc_spatial_transformation(DLC_configfile, video, dlc_string, H5_path, csv_destination):
    """
    DLC_string e.g. Final_videoDLC_resnet50_obj_locationSep27shuffle1_200000
    """
    deeplabcut.analyze_videos(DLC_configfile, [video])
    video_name = video.split('/')[-1][:-3]
    video_path = video.split('/')[:-1]
    H5_path = video_path + '/' + video_name + expected_dlc_string + 'h5'
    csv_path = H5_path.split('/')[:-1] + video_name + '.csv'
    convert_to_csv(H5_path, csv_path)
    eng = matlab.engine.start_matlab()
    matlab_output = eng.matlab_function(csv_path)
    head_direction = Head_direction(matlab_output)
    return head_direction.to_csv(csv_destination)



if __name__ == '__main__':
    run_dlc_spatial_transformation(DLC_configfile, video, dlc_string, H5_path, csv_destination)


