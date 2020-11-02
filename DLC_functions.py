import pandas as pd
import numpy as np
from scipy import signal
import math


def get_x_y_data(index):
    x_y_frame = pd.DataFrame({'x': array_of_smooths_x[index], 'y': array_of_smooths_y[index]})
    return x_y_frame

def load_tracks(H5_path):
    """
    loads H5 files in and returnsthe processed stracks
    returns dictionary of body parts containing dataframes with x/y coors
    """

    loaded_path = pd.read_hdf(H5_path)
    obstructive_title = loaded_path.keys()[0][0]
    extracted_tracks = loaded_path[obstructive_title]
    body_parts_list = list(extracted_tracks.columns.levels[0])
    dlc_output_list = list(extracted_tracks.columns.levels[1])
    body_coor = {}  # this is a dictionary of all body parts
    for body_part in body_parts_list:
        body_coor[body_part] = ([dlc_output_list[1]])
        body_coor[body_part].append(dlc_output_list[2])
    array_of_smooths_x = []
    array_of_smooths_y = []
    for key, vals in zip(body_coor.keys(), body_coor.values()):
        extracted_tracks[key].interpolate(limit=5)
        for values in vals:
            array_extract = extracted_tracks[key, values].to_numpy()
            savy_array = signal.savgol_filter(array_extract, 21, 3)
            if values == 'x':
                array_of_smooths_x.append(savy_array)
            else:
                array_of_smooths_y.append(savy_array)
    dictionary_of_dfs = {}
    for index, part in enumerate(body_parts_list):
        x_y_frame = get_x_y_data(index)
        dictionary_of_dfs[part] = (x_y_frame)

    return dictionary_of_dfs

def remove_parts_of_videoframe(extracted_tracks, x_or_y, threshold):
    """
    used to removed problematic parts of the videoframe, should be done in deeplabcut config
    pre training
    """
    for col in extracted_tracks.columns.levels[0]:
        extracted_tracks.loc[extracted_tracks[(col, x_or_y)] < threshold, [(col, 'x'), (col, 'y')]] = np.nan
        return extracted_tracks



def create_midline_array_oriented(l_cheek, r_cheek):
    midline_coor = np.add((l_cheek/2), (r_cheek/2))
    return midline_coor


def get_nov_obj_parameters(dictionary_of_obj):
    """
    takes in seperate data from dlc object recognition
    returns center and readius based on averages
    """
    tl_corner_x = dictionary_of_obj['tl_corner']['x']
    tl_corner_y = dictionary_of_obj['tl_corner']['y']
    tr_corner_x = dictionary_of_obj['tr_corner']['x']
    tr_corner_y = dictionary_of_obj['tr_corner']['y']
    ll_corner_x = dictionary_of_obj['ll_corner']['x']
    ll_corner_y = dictionary_of_obj['ll_corner']['y']
    lr_corner_x = dictionary_of_obj['lr_corner']['x']
    lr_corner_y = dictionary_of_obj['lr_corner']['y']
    data_x = {'tl_corner_x': tl_corner_x, 'tr_corner_x': tr_corner_x, 'll_corner_x': ll_corner_x,
              'lr_corner_x': lr_corner_x}
    data_y = {'tl_corner_y': tl_corner_y, 'tr_corner_y': tr_corner_y, 'll_corner_y': ll_corner_y,
              'lr_corner_y': lr_corner_y}
    aligned_x = pd.DataFrame(data_x)
    aligned_y = pd.DataFrame(data_y)
    aligned_x = aligned_x.median()
    aligned_y = aligned_y.median()
    center_x = aligned_x.mean()
    center_y = aligned_y.mean()

    distances = []
    for x, y in zip(aligned_x, aligned_y):
        distances.append(math.sqrt((center_x - x) ** 2 + (center_y - y) ** 2))
    radius = np.average(np.array(distances))
    return radius, center_x, center_y

def which_side_of_line(x_l_cheek, y_l_cheek, x_r_cheek, y_r_cheek, pointx, pointy):
    """
    returns False if coor are below the line
    returns True if coor is above the line
    d=(x−x1)(y2−y1)−(y−y1)(x2−x1)
    if d<0  then the point lies on one side of the line
    if d>0 then it lies on the other side
    If d=0 then the point lies exactly line.
    """
    d = ((pointx-x_l_cheek)*(y_r_cheek-y_l_cheek))-((pointy-y_l_cheek)*(x_r_cheek-x_l_cheek))
    if d<0:
        return 1
    if d>0:
        return 2
    if d==0:
        return 0

def checkCollision(xmid,ymid,xnose,ynose,x_obj,y_obj,radius,status=True):
    """
    Finding the distance of line
    from center
    ax + by + c = 0
    """
    x0 = x_obj
    y0 = y_obj
    x1 = xmid
    x2 = xnose
    y1 = ymid
    y2 = ynose
    numerator = ((y2-y1)*x0)-((x2-x1)*y0) + (x2*y1) - (y2*x1)
    denomenator = (y2-y1)**2+(x2-x1)**2
    dist =  (abs(numerator)/ math.sqrt(denomenator))
    if status == False:
        return dist
    elif radius >= dist:
        return True
    else:
        return False


def object_within_30_degrees_vision(midline_x, midline_y, x_nose, y_nose, x_l_cheek, \
                                    y_l_cheek, x_r_cheek, y_r_cheek, x_obj, y_obj, radius, distance_threshold):
    """
    first finds all the line values of the direction vector
    and ets the line values of the lateral head vector
    checks to see if the object is on the same side of the nose
    if thats the case applies check collisions, to see whether direction vecotr contacts objects
    """
    attending_to_obj = []
    object_distance_from_nose = []
    opposites = []
    angles_of_object = []

    for x1, y1, x2, y2, xnose, ynose, xmid, ymid in zip(x_l_cheek, y_l_cheek, x_r_cheek, y_r_cheek, x_nose, y_nose, \
                                                        midline_x, midline_y):

        opposite = checkCollision(xmid, ymid, xnose, ynose, x_obj, y_obj, radius, status=False)
        opposites.append(opposite)

        hypotenuse = math.sqrt((x_obj - xmid) ** 2 + (y_obj - ymid) ** 2)

        distance = math.sqrt((x_obj - xnose) ** 2 + (y_obj - ynose) ** 2)
        object_distance_from_nose.append(distance)

        intermediary = opposite / hypotenuse
        theta = math.asin(intermediary)

        nose_result = which_side_of_line(x1, y1, x2, y2, xnose, ynose)
        obj_result = which_side_of_line(x1, y1, x2, y2, x_obj, y_obj)

        if nose_result == obj_result:
            angles_of_object.append(math.degrees(theta))

            if distance < distance_threshold:
                attending_to_obj.append(True)
            else:
                attending_to_obj.append(False)
        else:
            angles_of_object.append(180 - (math.degrees(theta)))
            attending_to_obj.append(False)
    data = {'attending_to_obj': attending_to_obj, 'distance_from_nose': object_distance_from_nose,
            'degree_obj_FOV': angles_of_object}
    data_frame = pd.DataFrame(data=data)
    return data_frame

def Head_direction(H5file, H5object):
    """
    :param H5file:
    :param H5object:
    :return: dataframe containing information about the mouse head direction
    """
    dictionary_of_dfs = load_tracks(H5file)
    x_nose = dictionary_of_dfs['nose']['x']
    y_nose = dictionary_of_dfs['nose']['y']
    x_l_cheek = dictionary_of_dfs['l_cheek']['x']
    y_l_cheek = dictionary_of_dfs['l_cheek']['y']
    x_r_cheek = dictionary_of_dfs['r_cheek']['x']
    y_r_cheek = dictionary_of_dfs['r_cheek']['y']
    midline_x = create_midline_array_oriented(x_l_cheek, x_r_cheek)
    midline_y = create_midline_array_oriented(y_l_cheek, y_r_cheek)
    dictionary_of_obj = load_tracks (H5object)
    radius, center_x, center_y = get_nov_obj_parameters(dictionary_of_obj)
    data = object_within_30_degrees_vision(midline_x,midline_y,x_nose,y_nose,x_l_cheek,y_l_cheek,x_r_cheek,y_r_cheek,center_x,center_y,radius, 100)
    return data