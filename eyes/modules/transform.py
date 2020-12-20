import cv2
import numpy as np

FIELD_DIMS = (8.08, 4.48)  # meters
THICKNESS = 1
FONT = 0
FONT_SIZE = 0.46
LABEL_OFFSET = (7, 4)

transform_matrix = None
field_dims = None


def transform(centers, com):
    if com.frame_count == 1:
        global transform_matrix, field_dims

        tl, tr, br, bl = com.field_corners
        width, height = FIELD_DIMS
        new_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        transform_matrix = cv2.getPerspectiveTransform(com.field_corners.astype(np.float32), new_corners)

    coords = []
    for center in centers:
        x, y, z = transform_matrix @ np.array([*center, 1])
        coords.append([x / z, y / z])

    if com.debug:
        for center, coord in zip(centers, coords):
            label_position = tuple([sum(p) for p in zip(center, LABEL_OFFSET)])
            cv2.putText(com.debug_frame, str(np.round(coord, 2)), label_position, FONT, FONT_SIZE, (0, 255, 0))
    return coords
