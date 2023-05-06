import logging
import serial
import threading
import time
import numpy as np

ser = None
log_format = '%(asctime)s - [%(levelname)s] [%(module)s.%(funcName)s:%(lineno)d]: %(message)s'
logging.basicConfig(format=log_format, level=logging.INFO)
log = logging.getLogger(__name__)

last_actions = {'direction':None,
                'speed': None,
                'rotation': None,
                'action': None,
                'key': None}

def open_port():
    global ser
    ser = serial.Serial('/dev/ttyACM0', 9600) # Open serial port at 9600 bps
    ser.flushInput()

def change_car_speed(speed, time = 0):
    global ser, last_action
    if speed==2:
        last_actions['speed'] = '2'
        last_actions['key'] = b'o'
    elif speed==1:
        last_actions['speed'] = '1'
        last_actions['key'] = b'F'
    elif speed==0.5:
        last_actions['speed'] = '0.5'
        last_actions['key'] = b'f'
    elif speed==0:
        last_actions['speed'] = '0'
        last_actions['key'] = b'0'
    if ser:
        ser.write(last_actions['key'])

    return "Changing car speed to " + str(speed)

def change_speed_pedestrian(speed, time_s):
    global ser, last_action
    last_actions['speed'] = str(speed)
    last_actions['direction'] = 'forward'
    # last_actions['action'] = 'turn'
    last_actions['key'] = b'f'
    if ser:
        ser.write(last_actions['key'])
    time.sleep(time_s)
    last_actions['speed'] = '1'
    last_actions['action'] = 'turn'
    last_actions['direction'] = 'forward'
    last_actions['key'] = b'F'
    if ser:
        ser.write(last_actions['key'])

def stop_car():
    global ser, last_action
    last_actions['speed'] = '0'
    last_actions['direction'] = 'stop'
    last_actions['key'] = b's'

    if ser:
        ser.write(last_actions['key'])

def stop_car_temp(time_s, direction):
    global ser, last_action
    last_actions['speed'] = '0'
    last_actions['direction'] = 'stop'
    last_actions['action'] = 'turn'
    last_actions['key'] = b's'
    if ser:
        ser.write(last_actions['key'])
    time.sleep(time_s)
    last_actions['speed'] = '1'
    last_actions['action'] = 'turn'
    last_actions['direction'] = 'forward'
    last_actions['key'] = b'F'
    if ser:
        ser.write(last_actions['key'])
    time.sleep(1)
    action, text = car_change_direction(direction)

def overpass_roadblock(roadblock_min, roadblock_max, image):
    middlecamera = np.int(np.int_(image.shape[1])) / 2
    if roadblock_min < middlecamera < roadblock_max:
        car_change_rotation_temp(2, 'turn', 2)
        # shortest_path = ['86','77','82','78', '87', '45', '46', '42', '98', '99', '100','4', '9', '7', '134',
        # '140', '141', '142', '143', '15', '19', '17', '146', '25', '00', '26', '119', '120', '121', '122',
        # '123', '32', '30', '33', '113', '6', '90', '1', '111', '70', '75', '71', '124', '125', '126', '127',
        # '128', '59', '65', '62', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157',
        # '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '198', '199',
        # '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211',
        # '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223',
        # '224', '225', '226', '227', '228', '229', '230', '267', '268', '269', '270', '271',
        # '272', '273', '274', '275', '276', '277', '278', '279', '280', '281', '282', '283',
        # '284', '285', '286', '287', '23', '28', '26', '119', '120', '121', '122', '123', '32',
        # '38', '33', '107', '108', '109', '52', '57', '53', '91', '41', '46', '44', '89', '79',
        # '82', '76','85']
        # return shortest_path


def car_change_direction(direction):
    global ser, last_action
    if direction=='left': #forward left
        last_actions['direction'] = 'forward-left'
        last_actions['action'] = 'turn'
        last_actions['rotation'] = '1'
        last_actions['key'] = b'G'
    elif direction=='right': # forward right
        last_actions['direction'] = 'forward-right'
        last_actions['action'] = 'turn'
        last_actions['rotation'] = '9'
        last_actions['key'] = b'I'
    elif direction=='straight':
        last_actions['direction'] = 'forward'
        last_actions['action'] = 'turn'
        last_actions['rotation'] = '5'
        last_actions['key'] = b'f'
    if ser:
        ser.write(last_actions['key'])
        time.sleep(2)
        last_actions['direction'] = 'forward'
        last_actions['action'] = 'turn'
        last_actions['rotation'] = '5'
        last_actions['key'] = b'f'

        ser.write(last_actions['rotation'])
        ser.write(last_actions['key'])
    # log.info("Changing direction to " + direction)
    return 'turn', "Changing direction to " + direction

def car_change_rotation(angle, action):
    global ser, last_action
    # print('HERE I AM RPINTING ANGLE', angle, ser)
    if last_actions['action']!='turn':
        last_actions['rotation'] = str(angle)
        last_actions['key'] = str(angle).encode()
        print("Changing rotation to " + str(angle))
        if ser:
            ser.write(last_actions['key'])
    # log.info("Changing rotation to " + angle)
    
    return "Changing rotation to " + str(angle)


def car_change_rotation_temp(angle, action, time_s):
    global ser, last_action
    # print('HERE I AM RPINTING ANGLE', angle, ser)
    if last_actions['action']!='turn':
        last_actions['rotation'] = str(angle)
        last_actions['key'] = str(angle).encode()
        last_actions['action'] = action
        if ser:
            ser.write(last_actions['key'])
        time.sleep(time_s)

        last_actions['rotation'] = str(8)
        last_actions['key'] = str(8).encode()
        last_actions['action'] = action
        if ser:
            ser.write(last_actions['key'])
        time.sleep(time_s//2)
        last_actions['rotation'] = str(5)
        last_actions['key'] = str(angle).encode()
        last_actions['action'] = None
        if ser:
            ser.write(last_actions['key'])
    # log.info("Changing rotation to " + angle)
    # print(last_actions)
    return "Changing rotation to " + str(angle)

def do_parking():
    global ser, last_action
    # log.info("Parking the car in an empty parking lot")
    last_actions['action'] = 'parking'
    last_actions['key'] = b'p'
    if ser:
        ser.write(last_actions['key'])
    return "Parking the car in an empty parking lot"

def do_roundabout_rotation(direction):
    # log.info("Entering roundabout and rotating to " + direction)
    return "Entering roundabout and rotating to " + direction

def change_lane(direction):
    # log.info("Changing Lane to " + direction)
    return "Changing Lane to " + direction


def get_action(last_seen_label, direction):
    text = None
    action = None
    if last_seen_label == 'crossed_highway_sign':
        text = change_car_speed(speed = 1)
        threading.Thread(target=car_change_direction, args=(direction)).start()
        action = 'turn'
        # , text2 = car_change_direction(direction)
        # text+= '\n'+ text2
    elif last_seen_label == 'highway_sign':
        text = change_car_speed(speed = 2)
    elif last_seen_label == 'green_light':
        text = change_car_speed(speed = 1)
        threading.Thread(target=car_change_direction, args=(direction)).start()
        action = 'turn'
        # text += '\n'+ text2
    elif last_seen_label == 'yellow_light':
        text = change_car_speed(speed = 0.5)
    elif last_seen_label == 'red_light':
        text = change_car_speed(speed = 0)
    elif last_seen_label == 'no_entry_sign':
        text = "seen no entry sign"
    elif last_seen_label == 'one_way_road_sign':
        text = None
        # text = car_change_direction(direction)
    elif last_seen_label == 'parking_sign':
        text = change_car_speed(speed = 0.5)
        text += '\n'+ do_parking()
    elif last_seen_label == 'pedestrian_sign':
        threading.Thread(target=change_speed_pedestrian, args=(0.5, 3)).start()
    elif last_seen_label == 'pedestrian':
        threading.Thread(target=change_speed_pedestrian, args=(0, 10)).start()
    elif last_seen_label == 'priority_sign':
        threading.Thread(target=car_change_direction, args=(direction)).start()
        text = 'priority sign, turning'
        action = 'turn'
    elif last_seen_label == 'horizontal_line':
        threading.Thread(target=car_change_direction, args=(direction)).start()
        action = 'turn'
        text = 'found intersection, turning'
    elif last_seen_label == 'roundabout_sign':
        text = do_roundabout_rotation(direction = direction) # direction = 'left' or 'right' or 'straight'
        # action, text2, car_change_direction(direction)
        # text += '\n'+ text2
    elif last_seen_label == 'stop_sign':
        threading.Thread(target=stop_car_temp, args=(3, direction)).start()
        # text = change_car_speed(speed = 0, time=2) #change speed to stop only for 2 seconds
        # text += '\n'+ text2

    elif last_seen_label == 'car':
        text = None
    elif last_seen_label == 'roadblock':
        text = change_lane(direction)
    elif last_seen_label == 'random':
        text = None

    # print(last_actions)
    return action, text