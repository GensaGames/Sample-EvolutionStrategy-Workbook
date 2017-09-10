import cv2 as cv2
import numpy as np
import time
import sys
import math
from threading import Thread

IMAGE_WINDOW_NAME = 'Color Moving Evolution'
IMAGE_WINDOW_SIZE = 512

NOISE_POINTS_SIZE = 30
LEARN_RATE = 0.5
SIGMA = 2


# ################################################
# Create a blank image, a window
# and bind the function to window
# noinspection PyRedundantParentheses,PyPep8Naming
def start_window():
    print "---Started show image---"
    global active_image
    active_image = np.zeros((IMAGE_WINDOW_SIZE,
                             IMAGE_WINDOW_SIZE, 3), np.uint8)
    active_image[:] = tuple(reversed((255, 255, 255)))

    cv2.namedWindow(IMAGE_WINDOW_NAME)
    cv2.setMouseCallback(IMAGE_WINDOW_NAME, draw_circle, active_image)

    while (True):
        cv2.imshow(IMAGE_WINDOW_NAME, active_image)
        if cv2.waitKey(20) & 0xFF == 27:
            active_image = None
            sys.exit(2)


# Mouse callback function
def draw_circle(event, x, y, flags, param):
    global input_location

    if event == cv2.EVENT_LBUTTONUP:
        print "---Drawing info updated!---"
        if input_location is not None:
            cv2.circle(active_image, input_location, 10,
                       tuple(reversed((255, 255, 255))), -1)
        input_location = (x, y)
        cv2.circle(param, input_location, 10,
                   tuple(reversed((220, 20, 20))), -1)


# ################################################
# Main function with starting
# learning and drawing temp points
# noinspection PyRedundantParentheses
def start_learning():
    print "---Started learning---"
    global active_image
    time.sleep(1)

    while (True):
        time.sleep(0.1)
        if active_image is None:
            sys.exit(2)
        learn_step()


# Main function for starting learning.
# noinspection PyTypeChecker
def learn_step():
    global active_image, state_location, state_location_noise
    if state_location is None or input_location is None \
            or not state_location:
        start_center_point = IMAGE_WINDOW_SIZE / 2
        update_learn_location(start_center_point, start_center_point)
        return

    rewards = []
    for i in range(NOISE_POINTS_SIZE):
        noise = state_location_noise[i]
        rewards.append(get_rewards(noise[0], noise[1]))

    reward_mean = np.sum(rewards) / NOISE_POINTS_SIZE
    reward_std = np.std(rewards) + .00001
    new_state = [0, 0]

    print ("")
    print ("Reward avg: " + str(reward_mean)
           + " Reward std: " + str(reward_std))

    for i in range(NOISE_POINTS_SIZE):
        noise = state_location_noise[i]
        new_state[0] += noise[0] * (rewards[i] - reward_mean) / reward_std
        new_state[1] += noise[1] * (rewards[i] - reward_mean) / reward_std

    print ("Total new State X: " + str(int(new_state[0]))
           + " State Y: " + str(int(new_state[1])))

    avr_new_state_x = state_location[0] + int(new_state[0] * LEARN_RATE / (NOISE_POINTS_SIZE * SIGMA))
    avr_new_state_y = state_location[1] + int(new_state[1] * LEARN_RATE / (NOISE_POINTS_SIZE * SIGMA))

    print ("Apply new X: " + str(avr_new_state_x)
           + " Y: " + str(avr_new_state_y))

    print ("Previous  X: " + str(state_location[0])
           + " Y: " + str(state_location[1]))

    update_learn_location(avr_new_state_x, avr_new_state_y)


# Drawing Main point and Noise with some percentage from Image Size
# Redraw also previous point to white color, as clearing to White. 
def update_learn_location(x, y):
    global active_image, state_location, state_location_noise

    if state_location_noise is not None:
        for i in state_location_noise:
            cv2.circle(active_image, i, 1,
                       tuple(reversed((255, 255, 255))), -1)
    state_location_noise = []

    for i in range(NOISE_POINTS_SIZE):
        max_rand = int(IMAGE_WINDOW_SIZE * 0.05)
        location = (x + np.random.randint(-1 * max_rand, max_rand),
                    y + np.random.randint(-1 * max_rand, max_rand))

        state_location_noise.append(location)
        cv2.circle(active_image, location, 1,
                   tuple(reversed((190, 190, 190))), -1)

    if state_location is not None:
        cv2.circle(active_image, state_location, 5,
                   tuple(reversed((255, 255, 255))), -1)

    state_location = (x, y)
    cv2.circle(active_image, (x, y), 5,
               tuple(reversed((49, 49, 79))), -1)


# Base method for checking distance between two points.
# Greater value is better.
def get_rewards(x, y):
    if input_location is None:
        return 0

    return IMAGE_WINDOW_SIZE - math\
        .sqrt((x - input_location[0]) ** 2 + (y - input_location[1]) ** 2)


# ################################################
active_image = None
input_location = None

state_location = None
state_location_noise = None

thread_window = Thread(target=start_window)
thread_window.start()

thread_learn = Thread(target=start_learning)
thread_learn.start()
