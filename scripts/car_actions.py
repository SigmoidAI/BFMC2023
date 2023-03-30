import logging

log_format = '%(asctime)s - [%(levelname)s] [%(module)s.%(funcName)s:%(lineno)d]: %(message)s'
logging.basicConfig(format=log_format, level=logging.INFO)
log = logging.getLogger(__name__)

def change_car_speed(speed):
    # should put global vars
    log.info("Changing car speed to " + str(speed))
    pass

def car_change_direction(direction):
    log.info("Changing direction to " + direction)
    pass

def do_parking():
    log.info("Parking the car in an empty parking lot")
    pass

def do_roundabout_rotation(direction):
    log.info("Entering roundabout and rotating to " + direction)
    pass

def change_lane(direction):
    log.info("Changing Lane to " + direction)
    pass

def get_action(last_seen_label, direction):
    if last_seen_label == 'crossed_highway_sign':
        change_car_speed(speed = 1)
        car_change_direction(direction)
    elif last_seen_label == 'highway_sign':
        change_car_speed(speed = 2)
    elif last_seen_label == 'green_light':
        change_car_speed(speed = 1)
        car_change_direction(direction)
    elif last_seen_label == 'yellow_light':
        change_car_speed(speed = 0.5)
    elif last_seen_label == 'red_light':
        change_car_speed(speed = 0)
    elif last_seen_label == 'no_entry_sign':
        pass
    elif last_seen_label == 'one_way_road_sign':
        car_change_direction(direction)
    elif last_seen_label == 'parking_sign':
        change_car_speed(speed = 0.5)
        do_parking()
    elif last_seen_label == 'pedestrian_sign':
        change_car_speed(speed = 0.5)
    elif last_seen_label == 'priority_sign':
        car_change_direction(direction)
    elif last_seen_label == 'roundabout_sign':
        do_roundabout_rotation(direction = direction) # direction = 'left' or 'right' or 'straight'
        car_change_direction(direction)
    elif last_seen_label == 'stop_sign':
        change_car_speed(speed = 0, time=2) #change speed to stop only for 2 seconds
        car_change_direction(direction)
    elif last_seen_label == 'pedestrian':
        change_car_speed(speed = 0)
    elif last_seen_label == 'car':
        pass
    elif last_seen_label == 'roadblock':
        change_lane()