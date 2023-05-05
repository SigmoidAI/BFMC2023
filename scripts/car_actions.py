import logging
import serial


ser = None
log_format = '%(asctime)s - [%(levelname)s] [%(module)s.%(funcName)s:%(lineno)d]: %(message)s'
logging.basicConfig(format=log_format, level=logging.INFO)
log = logging.getLogger(__name__)


def open_port():
    global ser
    ser = serial.Serial('/dev/ttyACM0', 9600) # Open serial port at 9600 bps
    ser.flushInput()

def change_car_speed(speed, time = 0):
    global ser
    if ser:
        if speed==2:
            ser.write(b'o')
        elif speed==1:
            ser.write(b'F')
        elif speed==0.5:
            ser.write(b'f')
    return "Changing car speed to " + str(speed)


def car_change_direction(direction):
    global ser
    if ser:
        if direction=='left': #forward left
            ser.write(b'G')
        elif direction=='right': # forward right
            ser.write(b'I')
        elif direction=='straight':
            ser.write(b'f')
    # log.info("Changing direction to " + direction)
    return "Changing direction to " + direction

def car_change_rotation(angle):
    global ser
    if ser:
        ser.write(str(angle))
    # log.info("Changing rotation to " + angle)
    return "Changing rotation to " + str(angle)

def do_parking():
    global ser
    # log.info("Parking the car in an empty parking lot")
    if ser:
        ser.write(b'p')
    return "Parking the car in an empty parking lot"

def do_roundabout_rotation(direction):
    # log.info("Entering roundabout and rotating to " + direction)
    return "Entering roundabout and rotating to " + direction

def change_lane(direction):
    # log.info("Changing Lane to " + direction)
    return "Changing Lane to " + direction

def get_action(last_seen_label, direction):
    text = None

    if last_seen_label == 'crossed_highway_sign':
        text = change_car_speed(speed = 1)
        # text+= '\n'+ car_change_direction(direction)
    elif last_seen_label == 'highway_sign':
        text = change_car_speed(speed = 2)
    elif last_seen_label == 'green_light':
        text = change_car_speed(speed = 1)
        text += '\n'+ car_change_direction(direction)
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
        text = change_car_speed(speed = 0.5)
    elif last_seen_label == 'priority_sign':
        text = car_change_direction(direction)
    elif last_seen_label == 'roundabout_sign':
        text = do_roundabout_rotation(direction = direction) # direction = 'left' or 'right' or 'straight'
        text += '\n'+ car_change_direction(direction)
    elif last_seen_label == 'stop_sign':
        text = change_car_speed(speed = 0, time=2) #change speed to stop only for 2 seconds
        text += '\n'+ car_change_direction(direction)
    elif last_seen_label == 'pedestrian':
        text = change_car_speed(speed = 0)
    elif last_seen_label == 'car':
        text = None
    elif last_seen_label == 'roadblock':
        text = change_lane(direction)
    elif last_seen_label == 'random':
        text = None

    return None, text