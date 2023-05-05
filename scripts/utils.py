import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io, color, data
from skimage.morphology import erosion, dilation
import random
import os
import numpy as np
import cv2
import math
from frr import FastReflectionRemoval
import logging
import networkx as nx

LINES_RATE = 3

log_format = '%(asctime)s - [%(levelname)s] [%(module)s.%(funcName)s:%(lineno)d]: %(message)s'
logging.basicConfig(format=log_format, level=logging.INFO)
log = logging.getLogger(__name__)

# define constants
# horizontal_dim = image.shape[1]
bright_high = 180
bright_low = 160
normal = 110
normal_low = 70
dark_high = 45
dark_low = 35
car_direction = np.pi / 2
old_line_image = np.empty((720, 1280, 3))
old_car_offset = 0.0
old_angle = 0.0

alg = FastReflectionRemoval(h=0.1)


# to draw lines on the image
def add_weights(image, initial_image, alpha=0.8, beta=1, lambd=0):
    weighted_image = cv2.addWeighted(initial_image, alpha, image, beta, lambd)
    return weighted_image

# to get the lanes on the region of interest
def get_region_of_interest(edges, vertices):
    img = np.zeros_like(edges)
    if len(edges.shape) > 2:
        mask_color_ignore = (255, ) * edges.shape[2]
    else:
        mask_color_ignore = 255
    cv2.fillPoly(img, vertices, mask_color_ignore)
    masked_image = cv2.bitwise_and(edges, img)
    # io.imshow(img)
    return masked_image


# the coordinates for the line are calculated and drawn on the image
def draw_lines(image, lines, color=(255, 0, 0), thickness=10):
    left_slope, right_slope = [], []
    left_lane, right_lane = [], []
    first_shape = image.shape[0]
    test_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope > 0.4:
                right_slope.append(slope)
                right_lane.append(line)
            elif slope < -0.4:
                left_slope.append(slope)
                left_lane.append(line)
            first_shape = min(y1, y2, first_shape)
    if (len(left_lane) == 0) or (len(right_lane) == 0):
        # log.info('No lane detected')
        return 1, 1, 0
    slope_mean_left = np.mean(left_slope, axis=0)
    slope_mean_right = np.mean(right_slope, axis=0)
    mean_left = np.mean(np.array(left_lane), axis=0)
    mean_right = np.mean(np.array(right_lane), axis=0)
    if (slope_mean_left == 0) or (slope_mean_right == 0):
        # print('Not possible dividing by zero')
        return 1, 1, 0

        
    
# There was a code to calculate x coordinates in case of 2 lines are identified (the middle of the lane),
    x1_left = np.int_(left_lane[0][0][0])
    x2_left = np.int_(left_lane[0][0][2])
    x1_right = np.int_(right_lane[0][0][0])
    x2_right = np.int_(right_lane[0][0][2])
    
    if x1_left < x1_right:
        x1_left = int((x1_left + x1_right) / 2)
        x1_right = x1_left
        y2_left = y2_right = image.shape[0]
        try:
            y1_left = int((slope_mean_left * x1_left) + mean_left[0][1] - (slope_mean_left * mean_left[0][0]))
            y1_right = int((slope_mean_right * x1_right) + mean_right[0][1] - (slope_mean_right * mean_right[0][0]))
            y2_left = int((slope_mean_left * x2_left) + mean_left[0][1] - (slope_mean_left * mean_left[0][0]))
            y2_right = int((slope_mean_right * x2_right) + mean_right[0][1] - (slope_mean_right * mean_right[0][0]))
        except:
            y1_left = 0
            y1_right = 0
            y2_left = 0
            y2_right = 0
    else:
        y1_left = first_shape
        y1_right = first_shape
        y2_left = first_shape
        y2_right = first_shape
        
    x_middle = (x2_right - x2_left) / 2 + x2_left
    angle = (abs(np.arctan2(y2_right - y1_right, x2_right - x1_right)) + 
            abs(np.arctan2(y2_left - y1_left, x2_left - x1_left))) / 2
    
    new_image = np.array([x1_left, y1_left, x2_left, y2_left, x1_right, y1_right, x2_right, y2_right], dtype="float32")
    car_offset = np.int_(np.int_(image.shape[1])) / 2 - x_middle
    cv2.line(image, (int(new_image[0]), int(new_image[1])), (int(new_image[2]), int(new_image[3])), color, thickness)
    cv2.line(image, (int(new_image[4]), int(new_image[5])), (int(new_image[6]), int(new_image[7])), color, thickness)
    cv2.circle(image, (int(x_middle), image.shape[0] - 20), radius=10, color=(0, 255, 0), thickness=-1)
    return car_offset, angle, x_middle


# to make lines out of identified edges
def get_hough_lines(image, rho, theta, treshold, min_line_len, max_line_gap):
    try:
        lines = cv2.HoughLinesP(image, rho, theta, treshold, np.array([]), minLineLength=min_line_len, 
                           maxLineGap = max_line_gap)
    except: 
        return -1
    line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    m_lines = process_lines(lines)
    merged_lines = []
    for line in m_lines:
        merged_lines.append([[line[0][0], line[0][1], line[1][0], line[1][1]]])
    merged_lines = np.array(merged_lines)
    # here the merge for multiple lines is run and I output their coordinates
    car_offset, angle, x_middle = draw_lines(line_img, merged_lines)
    return line_img, car_offset, angle, x_middle

# here starts code for line merging
def get_lines(lines_in):
    try:
        return [l[0] for l in lines_in]
        
    except:
        return [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

def process_lines(lines):
    for line in get_lines(lines):
        leftx, boty, rightx, topy = line
    _lines = []
    for _line in get_lines(lines):
        _lines.append([(_line[0], _line[1]),(_line[2], _line[3])])
        
    # sort
    _lines_x = []
    _lines_y = []
    for line_i in _lines:
        orientation_i = math.atan2((line_i[0][1]-line_i[1][1]),(line_i[0][0]-line_i[1][0]))
        if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90+45):
            _lines_y.append(line_i)
        else:
            _lines_x.append(line_i)
            
    _lines_x = sorted(_lines_x, key=lambda _line: _line[0][0])
    _lines_y = sorted(_lines_y, key=lambda _line: _line[0][1])
        
    merged_lines_x = merge_lines_pipeline_2(_lines_x)
    merged_lines_y = merge_lines_pipeline_2(_lines_y)
    
    merged_lines_all = []
    merged_lines_all.extend(merged_lines_x)
    merged_lines_all.extend(merged_lines_y)
    return merged_lines_all

def merge_lines_pipeline_2(lines):
    super_lines_final = []
    super_lines = []
    min_distance_to_merge = 30
    min_angle_to_merge = 30
    
    for line in lines:
        create_new_group = True
        group_updated = False

        for group in super_lines:
            for line2 in group:
                if get_distance(line2, line) < min_distance_to_merge:
                    # check the angle between lines       
                    orientation_i = math.atan2((line[0][1]-line[1][1]),(line[0][0]-line[1][0]))
                    orientation_j = math.atan2((line2[0][1]-line2[1][1]),(line2[0][0]-line2[1][0]))

                    if int(abs(abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge: 
                        #print("angles", orientation_i, orientation_j)
                        #print(int(abs(orientation_i - orientation_j)))
                        group.append(line)

                        create_new_group = False
                        group_updated = True
                        break
            
            if group_updated:
                break

        if (create_new_group):
            new_group = []
            new_group.append(line)

            for idx, line2 in enumerate(lines):
                # check the distance between lines
                if get_distance(line2, line) < min_distance_to_merge:
                    # check the angle between lines       
                    orientation_i = math.atan2((line[0][1]-line[1][1]),(line[0][0]-line[1][0]))
                    orientation_j = math.atan2((line2[0][1]-line2[1][1]),(line2[0][0]-line2[1][0]))

                    if int(abs(abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge: 
                        #print("angles", orientation_i, orientation_j)
                        #print(int(abs(orientation_i - orientation_j)))

                        new_group.append(line2)

                        # remove line from lines list
                        #lines[idx] = False
            # append new group
            super_lines.append(new_group)
        
    
    for group in super_lines:
        super_lines_final.append(merge_lines_segments1(group))
    
    return super_lines_final


def merge_lines_segments1(lines, use_log=False):
    if(len(lines) == 1):
        return lines[0]
    
    line_i = lines[0]
    
    # orientation
    orientation_i = math.atan2((line_i[0][1]-line_i[1][1]),(line_i[0][0]-line_i[1][0]))
    
    points = []
    for line in lines:
        points.append(line[0])
        points.append(line[1])
        
    if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90+45):
        
        #sort by y
        points = sorted(points, key=lambda point: point[1])
        
        # if use_log:
            # print("use y")
    else:
        
        #sort by x
        points = sorted(points, key=lambda point: point[0])
        
        # if use_log:
            # print("use x")
    
    return [points[0], points[len(points)-1]]


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
# https://stackoverflow.com/questions/32702075/what-would-be-the-fastest-way-to-find-the-maximum-of-all-possible-distances-betw
def lines_close(line1, line2):
    dist1 = math.hypot(line1[0][0] - line2[0][0], line1[0][0] - line2[0][1])
    dist2 = math.hypot(line1[0][2] - line2[0][0], line1[0][3] - line2[0][1])
    dist3 = math.hypot(line1[0][0] - line2[0][2], line1[0][0] - line2[0][3])
    dist4 = math.hypot(line1[0][2] - line2[0][2], line1[0][3] - line2[0][3])
    
    if (min(dist1,dist2,dist3,dist4) < 100):
        return True
    else:
        return False


def lineMagnitude (x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2)+ math.pow((y2 - y1), 2))
    return lineMagnitude

def DistancePointLine(px, py, x1, y1, x2, y2):
    #http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
    LineMag = lineMagnitude(x1, y1, x2, y2)
    if LineMag < 0.00000001:
        DistancePointLine = 9999
        return DistancePointLine
    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)
    if (u < 0.00001) or (u > 1):
        #// closest point does not fall within the line segment, take the shorter distance
        #// to an endpoint
        ix = lineMagnitude(px, py, x1, y1)
        iy = lineMagnitude(px, py, x2, y2)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    else:
        # Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = lineMagnitude(px, py, ix, iy)
    return DistancePointLine

def get_distance(line1, line2):
    dist1 = DistancePointLine(line1[0][0], line1[0][1], 
                              line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist2 = DistancePointLine(line1[1][0], line1[1][1], 
                              line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist3 = DistancePointLine(line2[0][0], line2[0][1], 
                              line1[0][0], line1[0][1], line1[1][0], line1[1][1])
    dist4 = DistancePointLine(line2[1][0], line2[1][1], 
                              line1[0][0], line1[0][1], line1[1][0], line1[1][1])
    
    
    return min(dist1,dist2,dist3,dist4)

def apply_hsv_mask(img):
  # Convert image to HSV color space
  hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  # Define HSV color range to extract
  lower_range = np.array([23, 0, 151])
  upper_range = np.array([120, 252, 255])

  # Create a mask of pixels within the HSV range
  mask = cv2.inRange(hsv_img, lower_range, upper_range)

  # Change the pixels in the mask to white
  thresholded_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]

  # Invert the mask
  inverted_mask = cv2.bitwise_not(thresholded_mask)

  # Set the pixels outside the mask to black
  output_img = cv2.bitwise_and(img, img, mask=thresholded_mask)
  output_img[np.where((output_img == [0,0,0]).all(axis=2))] = [255,255,255]
  output_img[inverted_mask != 0] = [0,0,0]

  return output_img

def remove_reflection(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to the grayscale image
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Detect edges in the blurred image using Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate the edges to fill in gaps
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Create a mask from the dilated edges
    _, mask = cv2.threshold(dilated_edges, 1, 255, cv2.THRESH_BINARY_INV)

    # Apply the mask to the original image
    result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    return result


def find_line_lane(nr_frame, image):
    # transform to gray scale and hue scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_gray_image = cv2.GaussianBlur(gray_image, (5, 5), 3)

    # this version, hsv mask
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, bright_high], dtype=np.uint8)
    upper_white = np.array([80, 80, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv_image, lower_white, upper_white)
    t_image = cv2.bitwise_and(image, image, mask=mask)
    # print(image.shape)
    # print(image)
    # cv2.imshow('img', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    global old_angle
    global old_line_image
    global old_car_offset

    import os
    if nr_frame % LINES_RATE == 0:

        # identify edges
        # canny_edges = cv2.Canny(t_image, 50, 150, apertureSize=3)
        # cv2.imshow('canny', canny_edges)

        laplacian_edges = cv2.Laplacian(blur_gray_image, cv2.CV_16S, ksize=3)
        laplacian_edges = cv2.convertScaleAbs(laplacian_edges)

        # Optional: Apply a threshold to enhance the detected edges
        _, laplacian_edges = cv2.threshold(laplacian_edges, 30, 255, cv2.THRESH_BINARY)

        # print(canny_edges)
        # print(laplacian_edges)

        imshape = image.shape
        lower_left = [0+imshape[1]/10, imshape[0]]
        lower_right = [imshape[1]-imshape[1]/10, imshape[0]]
        top_left = [imshape[1] / 2 - imshape[1] / 4, imshape[0] / 2 + imshape[0] / 6]
        top_right = [imshape[1] / 2 + imshape[1] / 4, imshape[0] / 2 + imshape[0] / 6]
        
        # identify vertices
        vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]
        
        # get region of interest image
        # roi_image = get_region_of_interest(canny_edges, vertices)
        roi_image = get_region_of_interest(laplacian_edges, vertices)
        cv2.imshow('roi_image', roi_image)
        theta = np.pi / 180

        line_image, car_offset, angle, x_middle = get_hough_lines(roi_image, 4, theta, 120, 20, 70)
        if (car_offset == 1) and (angle == 1):
            return -1, -1, -1, 0
        relative_angle = (angle - car_direction) * 180 / np.pi
        results = cv2.addWeighted(image, 0.8, line_image, 1, 0)
        old_line_image = line_image
        old_car_offset = car_offset
        old_angle = angle
    else:
        try:
            results = cv2.addWeighted(image, 0.8, old_line_image, 1, 0)
        except:
            results = image.copy()
        return old_car_offset, old_angle, results, 0
    return car_offset, relative_angle, results, x_middle



#### VOVA PART ####

from heapq import heappop, heappush

def find_shortest_path_length(G, start, end):
    return nx.shortest_path_length(G, start, end)

def shortest_path_without_weights(G, start, points_to_visit):
    #find the shortest path between the start and one of the points to visit
    paths = {}
    #add the points to visit as keys to the paths dictionary

    for points in points_to_visit:
        if nx.has_path(G, start, points):
            shortest_path = nx.shortest_path(G, start, points)
            paths[points] = len(shortest_path)
    # print(paths)
    #find the key with the smallest value in the dictionary paths
    value = min(paths, key=paths.get)
    pointers = [start]+points_to_visit[points_to_visit.index(value):]+points_to_visit[:points_to_visit.index(value)] + ['85']
    new_path = []
    new_path.append(start)
    for i in range(len(pointers)-1):

        new_path.extend(nx.shortest_path(G, pointers[i], pointers[i+1])[1:])


    return new_path


# create a simple dictionary with keys named priority_sign, traffic and stop and values 0, 0 and False respectively


sign_dict = {'parking_sign': [181, 177, 162, 165], 
             'pedestrian_sign': [171, 265, 295, 276, 92, 95], 
             'roundabout_sign': [230, 301, 342], 
             'no_entry_sign':[468, 15, 16], # kinda nonsense?
             'priority_sign':[63, 23, 36, 41],
             'one_way_road_sign':[8, 7, 426], 
             'stop_sign':[45, 54, 59, 61, 25, 72, 34, 467],   # 
             'highway_sign':[49, 343],
             'crossed_highway_sign':[426, 374, 338], 
             'traffic_lights':[77, 4, 2, 6],
             'blockade':[141]}



point_to_sign = {
    181: 'parking_sign',
    177: 'parking_sign',
    162: 'parking_sign',
    165: 'parking_sign',
    171: 'pedestrian_sign',
    265: 'pedestrian_sign',
    295: 'pedestrian_sign',
    276: 'pedestrian_sign',
    92: 'pedestrian_sign',
    95: 'pedestrian_sign',
    230: 'roundabout_sign',
    301: 'roundabout_sign',
    342: 'roundabout_sign',
    468: 'no_entry_sign',
    15: 'no_entry_sign',
    16: 'no_entry_sign',
    63: 'priority_sign',
    23: 'priority_sign',
    36: 'priority_sign',
    41: 'priority_sign',
    8: 'one_way_road_sign',
    7: 'one_way_road_sign',
    427: 'one_way_road_sign',
    45: 'stop_sign',
    54: 'stop_sign',
    59: 'stop_sign',
    61: 'stop_sign',
    25: 'stop_sign',
    72: 'stop_sign',
    34: 'stop_sign',
    467: 'stop_sign',
    49: 'highway_sign',
    343: 'highway_sign',
    426: 'crossed_highway_sign',
    374: 'crossed_highway_sign',
    338: 'crossed_highway_sign',
    77: 'green_light',
    4: 'green_light',
    2: 'green_light',
    6: 'green_light',
    141: 'blockade'
}


# should think do I put them like what I should see, or really where they are as it is


# here are all the intersections from the map (all the nodes that are in the center of an intersection,
# reminder, in the same intersection point there are 3-4 nodes, one for each direction)
intersections = ['9', '10', '11', '12', '28', '29', '30','37', '38', '39', '46','47','48' ,'55','56','57', '64','65','66',  '73','74','75', '82','83','84','305','267','302', '312'] #312 idk, is highway   #  '19', '20', '21' tehre only way straight, no intersection