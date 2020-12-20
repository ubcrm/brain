from modules.transform import transform
from modules.detect import detect
from modules.crop import crop
import cv2
import numpy as np
import json
import time

FRAME_SCALE = 2
DEBUG_SCALE = 2
MODE_RUN, MODE_DEBUG, MODE_TEST = 0, 1, 2
FRAME_DELAY = 0  # (ms)


def eyes(source, mode=MODE_DEBUG, data_file=None):
    com = Common()
    com.debug = (mode == MODE_DEBUG)
    com.test = (mode == MODE_TEST)

    if com.test:
        if data_file is None:
            raise RuntimeError('Provide data file to test against.')
        if type(source) is int:
            raise RuntimeError('Provide video source containing test footage.')
        print('Testing implementation...')
        coords_data = []
        failed_count = 0
        start_time = time.time()

    capture = cv2.VideoCapture(source)
    successful, frame = capture.read()
    if not successful:
        raise RuntimeError(f'Failed to read from video source "{source}".')
    com.frame_count = 0
    com.top_left = np.array([0, 0])
    com.orig_dims = np.array([capture.get(3), capture.get(4)]).astype(int)
    frame_dims = tuple(np.round(com.orig_dims / FRAME_SCALE).astype(int))
    debug_dims = tuple(np.round(com.orig_dims / DEBUG_SCALE).astype(int))

    while True:
        successful, frame = capture.read()
        if not successful:
            break
        if com.debug:
            com.debug_frame = cv2.resize(frame, debug_dims)
            print(f'\n----- Frame {com.frame_count} -----')
        com.frame_count += 1
        frame = cv2.resize(frame, frame_dims)

        frame = crop(frame, com)
        centers = detect(frame, com)
        coords = transform(centers, com)

        if com.debug:
            cv2.imshow('Debug Output', com.debug_frame)
            if cv2.waitKey(FRAME_DELAY) == ord('q'):
                break
        elif com.test:
            if len(coords) == 0:
                failed_count += 1
            else:
                coords_data.append(coords[0])

    if com.test:
        end_time = time.time()
        print(f'Time per frame: {(end_time - start_time) / com.frame_count * 1E3:.1f} ms')
        print('Deviations in (x, y): {:.2f}, {:.2f}'.format(*calc_deviations(coords_data, data_file)))
        print(f'Failed detection ratio: {failed_count / com.frame_count:.2f}')
    cv2.destroyAllWindows()


def calc_deviations(data, data_file):
    with open(data_file, 'r') as file:
        absolute = json.load(file)
        absolute = list(absolute.values())[1:]  # skip first frame

    measured = np.array(data)
    deviations = np.divide(np.subtract(absolute, measured), absolute) ** 2
    deviations = np.sqrt(np.average(deviations, axis=0))
    return deviations


class Common:
    def __init__(self):
        self.debug, self.test = None, None
        self.frame_count = None
        self.debug_frame = None
        self.orig_dims = None
        self.field_corners = None
        self.frame_scale = FRAME_SCALE

    def orig_to_frame(self, point, dtype=np.int):
        return (np.array(point) / FRAME_SCALE).astype(dtype)

    def frame_to_orig(self, point, dtype=np.int):
        return (np.array(point) * FRAME_SCALE).astype(dtype)

    def orig_to_debug(self, point, dtype=np.int):
        return (np.array(point) / DEBUG_SCALE).astype(dtype)
