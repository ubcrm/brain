import cv2

ROBOT_THRESHOLD = 28
MERGE_DIM_REL = 20 / 1080
ERODE_DIM_REL = 10 / 1080
CONTOUR_SIZE_THRESHOLD_REL = 60 / 1080

frame_prev = None
merge_kernel = None
erode_kernel = None
contour_area_threshold = None


def detect(frame, com):
    global frame_prev
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.inRange(frame, 1, ROBOT_THRESHOLD)

    if com.frame_count == 1:
        global erode_kernel, merge_kernel, contour_area_threshold
        merge_dim = com.orig_to_frame(MERGE_DIM_REL * com.orig_dims[1])
        erode_dim = com.orig_to_frame(ERODE_DIM_REL * com.orig_dims[1])
        merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_dim, merge_dim))
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_dim, erode_dim))

        contour_area_threshold = com.orig_to_frame(CONTOUR_SIZE_THRESHOLD_REL * com.orig_dims[1]) ** 2
        frame_prev = frame
        return []

    detection = cv2.bitwise_or(cv2.subtract(frame, frame_prev), cv2.subtract(frame_prev, frame))
    # detection = cv2.morphologyEx(detection, cv2.MORPH_OPEN, erode_kernel, iterations=4)
    detection = cv2.morphologyEx(detection, cv2.MORPH_CLOSE, merge_kernel, iterations=6)
    detection = cv2.morphologyEx(detection, cv2.MORPH_OPEN, erode_kernel, iterations=4)
    centers = find_contour_centers(detection)
    frame_prev = frame

    if com.debug:
        print(f'Found {len(centers)} robot centers.')
        for center in centers:
            cv2.circle(com.debug_frame, tuple(center), 3, (0, 255, 0), -1)
    return centers


def find_contour_centers(mask):
    contours, _ = cv2.findContours(mask, 1, 2)
    centers = []

    for contour in contours:
        if contour_area_threshold < cv2.contourArea(contour):
            moments = cv2.moments(contour)
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            centers.append([cx, cy])
    return centers
