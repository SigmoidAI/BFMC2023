import logging
import serial


ser = None
log_format = '%(asctime)s - [%(levelname)s] [%(module)s.%(funcName)s:%(lineno)d]: %(message)s'
logging.basicConfig(format=log_format, level=logging.INFO)
log = logging.getLogger(__name__)


def open_port():
    global ser
    ser = serial.Serial('/dev/ttyACM0', 115200) # Open serial port at 9600 bps


def change_car_speed(speed, time = 0):
    # should put global vars
    log.info("Changing car speed to " + str(speed))
    return "Changing car speed to " + str(speed)


def car_change_direction(direction):
    log.info("Changing direction to " + direction)
    return "Changing direction to " + direction

def do_parking():
    log.info("Parking the car in an empty parking lot")
    return "Parking the car in an empty parking lot"

def do_roundabout_rotation(direction):
    log.info("Entering roundabout and rotating to " + direction)
    return "Entering roundabout and rotating to " + direction

def change_lane(direction):
    log.info("Changing Lane to " + direction)
    return "Changing Lane to " + direction

def get_action(last_seen_label, direction):
    if last_seen_label == 'crossed_highway_sign':
        text = change_car_speed(speed = 1)
        text+=car_change_direction(direction)
    elif last_seen_label == 'highway_sign':
        text = change_car_speed(speed = 2)
    elif last_seen_label == 'green_light':
        text = change_car_speed(speed = 1)
        text += car_change_direction(direction)
    elif last_seen_label == 'yellow_light':
        text = change_car_speed(speed = 0.5)
    elif last_seen_label == 'red_light':
        text = change_car_speed(speed = 0)
    elif last_seen_label == 'no_entry_sign':
        pass
    elif last_seen_label == 'one_way_road_sign':
        text = car_change_direction(direction)
    elif last_seen_label == 'parking_sign':
        text = change_car_speed(speed = 0.5)
        text += do_parking()
    elif last_seen_label == 'pedestrian_sign':
        text = change_car_speed(speed = 0.5)
    elif last_seen_label == 'priority_sign':
        text = car_change_direction(direction)
    elif last_seen_label == 'roundabout_sign':
        text = do_roundabout_rotation(direction = direction) # direction = 'left' or 'right' or 'straight'
        text += car_change_direction(direction)
    elif last_seen_label == 'stop_sign':
        text = change_car_speed(speed = 0, time=2) #change speed to stop only for 2 seconds
        text += car_change_direction(direction)
    elif last_seen_label == 'pedestrian':
        text = change_car_speed(speed = 0)
    elif last_seen_label == 'car':
        pass
    elif last_seen_label == 'roadblock':
        change_lane()
    return None, text