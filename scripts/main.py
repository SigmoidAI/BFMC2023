import cv2
import numpy as np
from utils import find_line_lane, shortest_path_without_weights, intersections, point_to_sign
from openvino_detection import init_model, predict
import logging
import json
import time
import networkx as nx
import matplotlib.pyplot as plt
import pyzed.sl as sl
from ultralytics import YOLO
import random
import torch
from car_actions import *
import trafficlights
import vehicletovehicle
# from livetraffic.livetraffic import *
import threading
# from test_quantized_model import *

# variables that may need to be changed
STARTING_NODE = '86'
NODES_TO_VISIT = ['78', '40', '105','113', '111', '71', '62']

# Constants
PATH_TO_GRAPHML = r'./files/Competition_track.graphml'
PATH_TO_YOLO = r'./models/semifinal_model_2.pt'   #r'./models/best_based_on_8n.pt'
MODEL_NAME = 'semifinal_model_1'
PATH_TO_VIDEO = r'files\semi_final_track_1_right.avi'
FRAME_FREQUENCY = 30
CLASS_NAMES = ['crossed_highway_sign', 'green_light', 'highway_sign', 'no_entry_sign','one_way_road_sign', 'parking_sign','pedestrian_sign', 'priority_sign', 'red_light', 'roundabout_sign','stop_sign', 'yellow_light', 'car', 'pedestrian', 'roadblock']
COLORS = [(92,164,100),(0,255,0),(65,174,68), (51,51,255),(255,0,0), (204,0,0),(255,153,51),(51,255,255),(0,0,255), (192,192,192),(0,0,204), (0,255,255),(0,0,0), (204,229,255),(0,128,255)]
DISPLAY_MESSAGE = ["CROSSED HIGHWAY, SPEED 1", 'GO', 'HIGHWAY, SPEED 2', 'NO ENTRY','ONE WAY ROAD', 'DO PARKING, SPEED 0.5','CROSSWALK, LOOK OUT, SPEED 0.5', 'PRIORITY', 'RED LIGHT, STOP', 'ROUNDABOUT','STOP', 'YELLOW, wait', 'CAR', 'PEDESTRIAN', 'OBSTACLE!']

# Set the font scale and thickness
font_scale = 1
thickness = 2

# Set the text to be written
text = "Hello, World!"

# Set the position of the text
x = 50
y = 50

# Camera vars
cap = None 
runtime = None 
left = None 

# variables for algorithm
current_intersection_index = -1
current_index = 0
past_current_index = 0
last_seen_label = 'random'
direction = 'None'
shortest_path = []
intersection_to_go = []
turning_signs = []
signs_indexes_on_the_path = []
angle_values = [0, 0, 0, 0, 0]
G = None
frames = 0
last_timestamp = 0

prev_offset = 0
prev_angle = 0
prev_line = 0

model = None
compiled_model = None
device = 'cuda'

# Flags
go_to_old_road = False # can calculate it from points to visit
output_video = False


# Set the font type
font = cv2.FONT_HERSHEY_SIMPLEX

log_format = '%(asctime)s - [%(levelname)s] [%(module)s.%(funcName)s:%(lineno)d]: %(message)s'
logging.basicConfig(format=log_format, level=logging.INFO)
log = logging.getLogger(__name__)


def get_turning_point_signs(turning_points):
    signs = []
    for point in turning_points:
        if point_to_sign.get(int(point)):
            signs.append(point_to_sign.get(int(point)))
    return signs

def delete_other_roundabout_intersections(intersections):
    i = 0
    for x in intersections:
        if x in ['305','267','302']:
            while i<len(intersections)+1 and intersections[i+1] in ['305','267','302'] :
                del intersections[i+1]
        i+=1
    return intersections

def get_turn_direction():
    global past_current_index, current_index, last_seen_label, direction, shortest_path

    past_current_index = current_index
    next_nodes = shortest_path[current_index:current_index+10]
    for node_idx in range(len(next_nodes)):
        if next_nodes[node_idx] in intersections:
            intersection_node = next_nodes[node_idx]
            prev_node = shortest_path[current_index-1]
            if last_seen_label=='roundabout_sign':
                next_node = shortest_path[current_index+node_idx+9]
            else:
                next_node = shortest_path[current_index+node_idx+1]
            intersection_node_coord = (float(G.nodes[intersection_node]['x']), float(G.nodes[intersection_node]['y']))
            prev_node_coord = (float(G.nodes[prev_node]['x']), float(G.nodes[prev_node]['y']))
            next_node_coord = (float(G.nodes[next_node]['x']), float(G.nodes[next_node]['y']))
            # Calculate the angle between the 3 points wth intersection node as the center
            angle = np.arctan2(prev_node_coord[1] - intersection_node_coord[1], prev_node_coord[0] - intersection_node_coord[0]) - np.arctan2(next_node_coord[1] - intersection_node_coord[1], next_node_coord[0] - intersection_node_coord[0])
            angle = np.rad2deg(angle)
            if angle < 0:
                angle += 360
            if angle < 140:
                direction = 'right'
            elif angle > 220:
                direction = 'left'
            else:
                direction = 'straight'

            # log.info('direction: ' + direction + ' angle: ' + str(angle) + ' sign: ' + last_seen_label)
            # log.info(DISPLAY_MESSAGE)
            return direction

    
def get_shortest_path():
    global G

    G = nx.read_graphml(PATH_TO_GRAPHML)
    pos=nx.fruchterman_reingold_layout(G)
    shortest_path = shortest_path_without_weights(G, STARTING_NODE, NODES_TO_VISIT)    

    # shortest_path = ['78', '87', '45', '48','40','90', '54', '57', '49', '308', \
    # '309', '310', '311', '312', '375', '376', '377', '378', '379', '380', '381', '382', '383', '384', \
    # '385', '386', '387', '388', '389', '390', '391', '392', '393', '394', '395', '396', '397', '398', \
    # '338', '339', '340', '341', '342', '305', '306', '307', '267', '268', '269', '270', '271', '272', \
    # '273', '274', '275', '276', '277', '278', '279', '280', '281', '282', '283', '284', '285', '286', \
    # '287', '23', '28', '24', '147', '18', '21', '13', '145', '61', '65', '58', '129', '130', '131', '132', \
    # '133', '72', '75', '69', '110', '2', '9', '5', '112', '34', '38', '31', '114', '115', '116', '117', \
    # '118', '27', '30', '22', '288', '289', '290', '291', '292', '293', '294', '295', '296', '297', '298', \
    # '299', '300', '301', '302', '303', '304', '305', '306', '231', '232', '233', '234', '235', '236', '237', \
    # '238', '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252', \
    # '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266', '172', \
    # '173', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', \
    # '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '63', '66', '58', '129', '130', \
    # '131', '132', '133', '72', '75', '67', '95', '96', '97', '81', '84', '76', '85']


    # '86','77','82','78', '87', '45', '48','40','90', '54',
    # shortest_path = ['100','4', '9', '5', '112', '34', '38', '35', '107','108','109', '52', '57', '49', '308', \
    # '309', '310', '311', '312', '375', '376', '377', '378', '379', '380', '381', '382', '383', '384', \
    # '385', '386', '387', '388', '389', '390', '391', '392', '393', '394', '395', '396', '397', '398', \
    # '338', '339', '340', '341', '342', '305', '306', '231', '232', '233', '234', '235', '236', \
    # '237', '238', '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', \
    # '251', '252', '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', \
    # '265', '266', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', \
    # '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', \
    # '63', '66', '58', '129', '130', '131', '132', '133', '72', '75', '69', '110', '2', '9', '8', '139', \
    # '140', '141', '137', '138', '16', '19', '17', '146', '25', '28', '22', '288', '289', '290', '291', '292', \
    # '293', '294', '295', '296', '297', '298', '299', '300'] 
     

    # , '48','40','90', '54',
    shortest_path = ['86','77','82','78', '87', '45', '46', '42', '98', '99', '100','4', '9', '7', '134',
    '135', '136', '137', '138', '16', '19', '13', '145', '61', '65', '62', '148', '149',
    '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161',
    '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '198', '199',
    '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211',
    '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223',
    '224', '225', '226', '227', '228', '229', '230', '267', '268', '269', '270', '271',
    '272', '273', '274', '275', '276', '277', '278', '279', '280', '281', '282', '283',
    '284', '285', '286', '287', '23', '28', '26', '119', '120', '121', '122', '123', '32',
    '38', '33', '107', '108', '109', '52', '57', '53', '91', '41', '46', '44', '89', '79',
    '82', '76','85']              
  
    # log.info(f"Starting point is {STARTING_NODE}")
    # log.info(f"Generated path is {shortest_path}")

    return shortest_path

def transform_value(old_value, old_min=400, old_max=780, new_min=1, new_max=9):
    new_value = (old_value - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
    
    # Ensure the new value is within the new interval limits
    new_value = max(new_min, min(new_value, new_max))
    
    return new_value


def frame_process(img):
    # img = cv2.resize(img, (360,640))
    # time1 = time.time()
    global frames, text, last_seen_label, turning_signs, signs_indexes_on_the_path, last_timestamp, direction, prev_offset, prev_angle, current_intersection_index, G, current_index
    # log.info(f'Is processed frame with number: {frames}')

    # get img width and height
    img_h, img_w, _ = img.shape
    img_area = img_h * img_w
    label_class = None
    # Perform object detection using YOLOv5
    with torch.no_grad():
        results = model([img], verbose=False) #
        # results = predict(img)

    # Draw the bounding boxes and class labels on the frame
    for bbox in results:
        boxes = bbox.boxes
        for box_i in range(len(boxes)):
            x_min, y_min, x_max, y_max = boxes.xyxy[box_i]
            confidence = boxes.conf[box_i]
            label = boxes.cls[box_i]
            box_area = (x_max - x_min) * (y_max - y_min)

    # for box in results['det']:
    #     x_min, y_min, x_max, y_max, confidence, label = box
    #     # print(x_min, y_min, x_max, y_max, confidence, label)
    #     # calculate area of the box
    #     box_area = (x_max - x_min) * (y_max - y_min)

            if confidence > 0.5:
                color = COLORS[int(label)]
                label_class = CLASS_NAMES[int(label)]
                label = f"{CLASS_NAMES[int(label)]} {confidence:.2f}"
                # log.info(f'Was detected: {label}')
                if label_class in ['green_light','red_light', 'yellow_light']:
                    threshold = 0.006
                else:
                    threshold = 0.0035
                if box_area>threshold*img_area and last_seen_label != label_class:
                    last_seen_label = label_class
                    last_timestamp = time.time()
                    if label_class == signs_indexes_on_the_path[0] and label_class not in ['car'] :
                        del signs_indexes_on_the_path[0]
                        if label_class ==turning_signs[0]:
                            del turning_signs[0]
                            # last_seen_label = label_class
                            current_intersection_index+=1
                            current_index = shortest_path.index(intersection_to_go[current_intersection_index])
                            direction = get_turn_direction()
                        action, text = get_action(last_seen_label, direction)
                        last_timestamp = time.time()

                cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
                cv2.putText(img, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    car_offset, relative_angle, line_image, x_middle = find_line_lane(frames, img)
    if (car_offset == -1) and (relative_angle == -1) and (line_image == -1):
        car_offset = prev_offset
        relative_angle = prev_angle
        # line_image = cv2.addWeighted(img, 0.8, prev_line, 1, 0)
        line_image = img
    else:
        prev_offset = car_offset 
        prev_angle = relative_angle
        prev_line = line_image

    new_value = transform_value(x_middle)
    car_change_rotation(int(round(new_value)))


    if time.time() - last_timestamp > 5:
        last_seen_label = 'random'
        direction = ''

    cv2.putText(line_image, text, (x,y), font, font_scale, (255, 0, 0), thickness)
    cv2.putText(line_image, direction, (int(img_w/2), int(img_h/2) ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 2)
    cv2.imshow('frame', cv2.resize(line_image, (720, 480)))

    time2 = time.time()
    # log.info(f'Frame processing took {1/(time2-time1)} gfgbngfbngdbngfbgf')

    # log.info(f'Car offset is {car_offset}')
    # log.info(f'Relative angle is {relative_angle}')
    # print("Frame: {}, Car offset: {}, Relative angle: {}".format(frame, car_offset, relative_angle))


def initialize_program():
    global model, shortest_path, intersection_to_go, turning_points, current_intersection_index, turning_signs, signs_indexes_on_the_path, last_seen_label, last_timestamp, current_index, direction

    # open_port() #TODO
    model = YOLO(PATH_TO_YOLO)
    # model_path = fr"./models/{MODEL_NAME}_openvino_int8_model/{MODEL_NAME}.xml"
    # init_model(model_path)


    shortest_path = get_shortest_path()
    #intersections to pass through in calculated shortest path
    intersection_to_go = [x for x in shortest_path if x in intersections]
    intersection_to_go = delete_other_roundabout_intersections(intersection_to_go)

    signs_on_the_path = [x for x in shortest_path if point_to_sign.get(int(x))]
    signs_indexes_on_the_path = [point_to_sign.get(int(x)) for x in shortest_path if point_to_sign.get(int(x))]
    turning_points = [x for x in signs_on_the_path if x not in ['49','338', '171', '265', '295', '276', '92', '95','8' '7' '427','468', '15', '16','181', '177', '162', '165']]
    if '311' in turning_points and go_to_old_road==False:
        turning_points.remove('311')
        intersection_to_go.remove('312')
    index_34 = [i for i, x in enumerate(turning_points) if x =='34']
    for i in index_34:
        if i<len(turning_points)-1 and turning_points[i+1] == '301':
            turning_points.insert(i+1, '27')

    turning_signs = get_turning_point_signs(turning_points)

    #current intersection node number
    # current_intersection = intersection_to_go[current_intersection_index]

def init_camera():
    global cap, runtime, left, res
    cap = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = True
    
    
    runtime = sl.RuntimeParameters()
    left = sl.Mat()
    res = sl.Resolution(1280, 720)
    err = cap.open(init_params)
    return cap


def line_process(live_camera = True):
    global frames, runtime, cap, left, res, img_w, img_h, img_area, prev_offset, prev_angle, prev_line, current_index, direction, angle_values # TODO not sure if should remove something here

    if live_camera:
        cap = init_camera()

        if output_video:
            height = 720 
            width = 1280 
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter("output_video_camera.mp4", fourcc, 30.0, (width, height), isColor=True)
        
        while True:

            err_code = cap.grab(runtime)
            if err_code != sl.ERROR_CODE.SUCCESS:
                break
            frames += 1

            cap.retrieve_image(left, sl.VIEW.LEFT, resolution=res)

            if frames%FRAME_FREQUENCY == 0:

                img = cv2.cvtColor(left.get_data(), cv2.COLOR_RGBA2RGB)
                img = frame_process(img)

                if output_video:
                    out.write(img)
                # cv2.imshow('ZED', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    else:
        # Get video capture
        cap = cv2.VideoCapture(PATH_TO_VIDEO)

        # If output_video is True, save results as video
        if output_video:
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter("output_video.mp4", fourcc, 30.0, (width, height), isColor=True)

        # Loop through video frames
        while True:
            ret, img = cap.read()
            if not ret:
                break
            frames += 1
            if frames%FRAME_FREQUENCY == 0:
                img = frame_process(img)

                if output_video:
                    out.write(img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
    cap.release()
    if output_video:
        out.release()
    cv2.destroyAllWindows()


### for traffic lights ###
def runTrafficLightListener():

    # Semaphore colors list
    colors = ['red','yellow','green']   

    # Get time stamp when starting tester
    start_time = time.time()
    # Create listener object
    Semaphores = trafficlights.trafficlights()
    # Start the listener
    Semaphores.start()
    # Wait until 60 seconds passed
    while (time.time()-start_time < 60):
        # Clear the screen
        print("\033c")
        print("Example program that gets the states of each\nsemaphore from their broadcast messages\n")
        # Print each semaphore's data
        print("S1 color " + colors[Semaphores.s1_state] + ", code " + str(Semaphores.s1_state) + ".")
        print("S2 color " + colors[Semaphores.s2_state] + ", code " + str(Semaphores.s2_state) + ".")
        print("S3 color " + colors[Semaphores.s3_state] + ", code " + str(Semaphores.s3_state) + ".")
        print("S4 color " + colors[Semaphores.s4_state] + ", code " + str(Semaphores.s4_state) + ".")
        time.sleep(0.5)
    # Stop the listener
    Semaphores.stop()

### for vehicle listner ###
def runVehicleListener():

    # Get time stamp when starting tester
    start_time = time.time()
    # Create listener object
    vehicle = vehicletovehicle.vehicletovehicle()
    # Start the listener
    vehicle.start()

    # Wait until 60 seconds passed
    while (time.time()-start_time < 60):
        # Clear the screen
        print("\033c")
        # Print each received msg
        print("ID ", vehicle.ID, ", coor ", vehicle.pos)
        time.sleep(0.5)
    # Stop the listener
    vehicle.stop()



if __name__=="__main__":
    initialize_program()
    # runTrafficLightListener()
    # runVehicleListener()
    # beacon = 23456
    # id = 120
    # serverpublickey = 'publickey_livetraffic_server.pem'
    # clientprivatekey = 'privatekey_client.pem'
    
    # gpsStR, gpsStS = Pipe(duplex = False)

    # envhandler = EnvironmentalHandler(id, beacon, serverpublickey, gpsStR, clientprivatekey)
    # envhandler.start()
    # time.sleep(5)
    # for x in range(1, 10):
    #     time.sleep(random.uniform(1,5))
    #     a = {"obstacle_id": int(random.uniform(0,25)), "x": random.uniform(0,15), "y": 			random.uniform(0,15)}
    #     gpsStS.send(a)
        
    # envhandler.stop()
    # envhandler.join()
    line_process(live_camera = False)