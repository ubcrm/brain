import cv2
import numpy as np
from matplotlib import pyplot as plt

image = None
warped_image = None
width = None
height = None

# https://drive.google.com/drive/folders/1ANInuRB7Rk2yDQNqE2lO5rtYl3UEQ3Im - numbers below from rules manual
arena_length = 8490  # mm
eff_arena_length = 8080  # mm (without walls)
arena_wdith = 4890  # mm
eff_arena_wdith = 4480  # mm (without walls)

'''
Function Overview: calculates the transformation matrix for use in the "transform_pixel" function
Inputs: the path of the current image to analyze as a string
Outputs: the transformation matrix as a numpy array
Requirements: The first run of this function will return something meaningless, during the first run, hover your cursor
              over the perspective view of the arena that appears, then fill out the points of each of the corners of 
              the arena 
Notes: Using matplotlib flips bgr to rgb I think - because blue and red colours swap sides of the arena
'''
def calculate_matrix(image_path):
    global image
    global warped_image
    global width
    global height
    #load image:
    frame = cv2.imread(image_path)
    image = np.copy(frame)

    arena_l_w_ratio = arena_length / arena_wdith
    eff_arena_l_w_ratio = eff_arena_length / eff_arena_wdith

    # "expected" corners of arena - input these values after hovering your cursor over the image that appears
    T_L_X_avg = 1100
    T_L_Y_avg = 135
    T_R_X_avg = 1635
    T_R_Y_avg = 198
    B_L_X_avg = 335
    B_L_Y_avg = 560
    B_R_X_avg = 1160
    B_R_Y_avg = 960

    # draw both lines on image - this isn't necessary, but a nice visual
    cv2.line(frame, (T_L_X_avg, T_L_Y_avg), (B_L_X_avg, B_L_Y_avg), (255, 0, 0), 3)
    cv2.line(frame, (T_R_X_avg, T_R_Y_avg), (B_R_X_avg, B_R_Y_avg), (255, 0, 0), 3)

    widthA = np.sqrt(((B_R_X_avg - B_L_X_avg) ** 2) + ((B_R_Y_avg - B_L_Y_avg) ** 2))
    widthB = np.sqrt(((T_R_X_avg - T_L_X_avg) ** 2) + ((T_R_Y_avg - T_L_Y_avg) ** 2))
    minWidth = min(int(widthA), int(widthB))

    heightA = np.sqrt(((T_R_X_avg - B_R_X_avg) ** 2) + ((T_R_Y_avg - B_R_Y_avg) ** 2))
    heightB = np.sqrt(((T_L_X_avg - B_L_X_avg) ** 2) + ((T_L_Y_avg - B_L_Y_avg) ** 2))
    minHeight = min(int(heightA), int(heightB))

    # try minWidth as ratio we found of arena length/width - works almost the same
    minWidth = int(minHeight / eff_arena_l_w_ratio)

    # set up numpy arrays before creating matrix with findHomography function
    dst = np.float32([
        [minWidth, 0],
        [minWidth, minHeight],
        [0, 0],
        [0, minHeight]]) #, dtype="float32") -> this was necessary for testing in google collab

    original_lines = np.float32([
        [T_R_X_avg, T_R_Y_avg],
        [B_R_X_avg, B_R_Y_avg],
        [T_L_X_avg, T_L_Y_avg],
        [B_L_X_avg, B_L_Y_avg]])

    # calculate matrix H: https://zbigatron.com/mapping-camera-coordinates-to-a-2d-floor-plan/
    h, status = cv2.findHomography(original_lines, dst)
    width = minWidth
    height = minHeight

    # can comment out these lines after you have a "consistent" camera feed
    plt.imshow(frame)
    plt.show()
    warped_image = cv2.warpPerspective(image, h, (width, height))
    plt.imshow(warped_image)
    plt.show()

    return h

'''
Overview: Transforms a pixel from perspective to birds-eye view and maps to real world (competition area) 
          coordinate system
Inputs: Receives a transformation matrix from the "calculate_matrix" function and a numpy array of pixels
Outputs: Outputs a list of transformed pixels in mm (Sorry trent, taking in a numpy array and outputting a list is 
         weird, but I don't think its a big issue given that you'll put them in JSON format anyway)
Requirements: To get a useful result, ensure points given are actually located in the field of view, otherwise, you'll
              get transformed values that are outside the competition area
'''
def transform_pixel(tr_matrix,points_list):
    transformed_points = []

    for points in points_list:
        print(points.ravel())
        # for meaningful visuals, make clean copies each run
        #image_copy = np.copy(image)
        #warped_copy = np.copy(warped_image)

        test_pixel_x, test_pixel_y = points.ravel()

        # test_pixel must be in this format below for cv2.perspectiveTransform to work
        test_pixel = np.array([[[test_pixel_x, test_pixel_y]]], dtype='float32')
        #perspective_mask = cv2.circle(image_copy, (test_pixel_x, test_pixel_y), 5, (0, 255, 0), 10)
        #plt.imshow(perspective_mask)
        #plt.show()

        transformed_coor = cv2.perspectiveTransform(test_pixel, tr_matrix)
        tr_coor_x = int(transformed_coor.ravel()[0])
        tr_coor_y = int(transformed_coor.ravel()[1])

        #birds_eye_mask = cv2.circle(warped_copy, (tr_coor_x, tr_coor_y), 5, (0, 255, 0), 10)
        #plt.imshow(birds_eye_mask)
        #plt.show()

        real_X = int((tr_coor_x / width) * eff_arena_wdith)
        real_Y = int((tr_coor_y / height) * eff_arena_length)

        transformed_points.append([real_X,real_Y])

    # can comment out print statements later
    print("untransformed points: " + str(points_list))
    print("transformed points: " + str(transformed_points))
    print("effective arena boundaries: " + str((eff_arena_wdith,eff_arena_length)))
    print("these test pixels are supposed to be in the 1000x1000 squares of each corner")
    return transformed_points

# I have this main function here to test functionality
if __name__ == '__main__':
    # for testing purposes, implemented with this image in the same folder as this .py file
    image_path = "./Simulation0-frame0-camera0.png"
    matrix = calculate_matrix(image_path)
    random_points_array = np.array([[1462,220],[1050,680]])
    transformed_points = transform_pixel(matrix,random_points_array)
