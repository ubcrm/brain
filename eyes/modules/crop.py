import json
import os
import pathlib
import cv2
import numpy as np
import copy

ASSETS_DIR = pathlib.Path(os.path.dirname(__file__)) / 'assets'
CORNERS_FILE = ASSETS_DIR / 'corners.json'
WINDOW_NAME = 'Assign Field Boundaries'
LINE_COLOR = (0, 255, 0)
DELAY = 20

field_bounds = []
mouse_position = None
busy_bound = False
count_bounds = 0


def crop(frame, com):
    if com.frame_count == 1:
        saved_corners = load_corners(com)
        com.field_corners = saved_corners if saved_corners is not None else set_corners(frame, com)

    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.drawContours(mask, [com.field_corners.astype(np.int32)], -1, 255, -1)
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    return frame


def load_corners(com):
    if not os.path.exists(CORNERS_FILE):
        return None

    with open(CORNERS_FILE, 'r') as file:
        try:
            corner_data = json.load(file)
        except json.decoder.JSONDecodeError:
            print('Failed to load field corner values. Please reassign them.')
            return None
    corners = com.orig_to_frame(list(corner_data.values()))
    return corners


def set_corners(frame, com):
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse_event)

    while count_bounds < 4:
        frame_copy = copy.deepcopy(frame)
        if busy_bound:
            cv2.line(frame_copy, tuple(field_bounds[count_bounds][0]), tuple(mouse_position), LINE_COLOR, 1)
        cv2.imshow(WINDOW_NAME, frame_copy)
        cv2.waitKey(DELAY)
    cv2.destroyWindow(WINDOW_NAME)

    corners = calculate_corners(field_bounds)
    corner_data = {i: com.frame_to_orig(corners[i]).tolist() for i in range(4)}
    with open(CORNERS_FILE, 'w+') as file:
        json.dump(corner_data, file, indent=2)
    return corners


def on_mouse_event(event, x, y, *args):
    global busy_bound, field_bounds, count_bounds, mouse_position
    mouse_position = [x, y]

    if event == cv2.EVENT_LBUTTONDOWN:
        field_bounds.append([[x, y], None])
        busy_bound = True
    elif event == cv2.EVENT_LBUTTONUP:
        field_bounds[count_bounds][1] = [x, y]
        count_bounds += 1
        busy_bound = False


def calculate_corners(bounds):
    corners = []
    lines = []

    for p1, p2 in bounds:
        a = p1[1] - p2[1]
        b = p2[0] - p1[0]
        c = p2[0] * p1[1] - p1[0] * p2[1]
        lines.append([a, b, c])

    for i in range(4):
        l1, l2 = lines[i], lines[(i + 1) % 4]
        delta = l1[0] * l2[1] - l1[1] * l2[0]
        dx = l1[2] * l2[1] - l1[1] * l2[2]
        dy = l1[0] * l2[2] - l1[2] * l2[0]

        if delta != 0:
            x = dx / delta
            y = dy / delta
            corners.append([x, y])
    corners = order_corners(np.array(corners, dtype=np.int32))
    return corners


def order_corners(corners, flip=False):
    xs = corners[:, 0]
    ys = corners[:, 1]
    order = [np.argmin(ys), np.argmax(xs), np.argmax(ys), np.argmin(xs)]
    reordered = corners[order]

    if flip:
        reordered = np.roll(reordered, 2)
    return reordered
