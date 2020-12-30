import cv2
TrDict = {'csrt': cv2.TrackerCSRT_create,
          'kcf': cv2.TrackerKCF_create,
          'boosting': cv2.TrackerBoosting_create,
          'mil': cv2.TrackerMIL_create,
          'tld': cv2.TrackerTLD_create,
          'medianflow': cv2.Tracker,
          'mosse': cv2.TrackerMOSSE_create} #different types of trackers, will be used to compare performances for tracking

trackers = cv2.MultiTracker_create() #decide which tracker to
v = cv2.VideoCapture(r'C:\Users\sayem\Downloads\Clip2.mp4') #find video to capture
#getting the frame rate of a video(later will be used to find the velocity of the robots)
fps = v.get(cv2.CAP_PROP_FPS)
time_elapsed = 1/fps
ret, frame = v.read()
k = 2
for i in range(k):
    cv2.imshow('Frame', frame)
    bbi = cv2.selectROI('Frame', frame)
    tracker_i = TrDict['csrt']()
    trackers.add(tracker_i, frame, bbi)
(success, boxes) = trackers.update(frame)
while True: #   Keep reading frames
    prev_boxes = boxes
    ret,frame = v.read() #read every frame in video
    if not ret:
        break #this is when the stream ends :(
    (success, boxes) = trackers.update(frame)
    for index, box in enumerate(boxes):
        (x,y,w,h) = [int(a) for a in box] #returns the bottom corner of the tracking box along with the width and height of the box
        (cx, cy) = int(x+w/2), int(y+h/2)
        (vx,vy) = (x+w/2 - (prev_boxes[index,0]+prev_boxes[index,2]/2))/time_elapsed, (y+h/2 - (prev_boxes[index,1]+prev_boxes[index,3]/2))/time_elapsed
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #x,y is coordinates of the bottom left corner of the box while the other point is the top left corner of the box
        cv2.circle(frame,(cx,cy), 3, (0, 255, 0), -1)
        loc = cx,cy
        cv2.putText(frame, str(loc), (x,y),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
v.release()
cv2.destroyAllWindows()