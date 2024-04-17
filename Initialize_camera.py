import cv2
from cv2 import SimpleBlobDetector
import collections
import numpy as np
import time, datetime

# Initializing webcam
cap = cv2.VideoCapture(0)

class MotionDetector():

    def __init__(self):
        self.frame_counter=0
        self.recent_frames = collections.deque(maxlen=7)
        self.recent_center_points = collections.deque(maxlen=7)
        self.second_avg_frames = collections.deque(maxlen=10)
        self.recent_difference = collections.deque(maxlen=5)
        self.recent_brightness = collections.deque(maxlen=3) # save last bright_std_mean. difference gives brightness increase/deacrease.
        self.ts2dt = lambda timestamp: datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        self.change_array = np.array([[]]) #add the frames relevant to this change. CHANGE THIS TO JUST REGULAR ARRAY.
        # self.frame_time = 

    def image_process(self, frame):
        # if time.time() - (self.frame_time) > 0.05: # if it's taking too much time to process the frame, skip it
        #     return None
        self.frame_counter+=1
        image_fg = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        image_fg = cv2.resize(image_fg, (320, 240))
        
        
        
        self.recent_frames.append(image_fg)
        avg_frame=np.min(self.recent_frames, axis=0).astype(np.int8)
        if self.frame_counter%30==0:
            self.second_avg_frames.append(avg_frame)
        # image = np.mean(self.recent_frames, axis=0).astype(np.uint8)
        
        image = np.abs(image_fg - avg_frame)###
        
        
        
        #image = np.abs(image - np.mean(self.second_avg_frames, axis=0).astype(np.int16))
        
        image_bg = cv2.medianBlur(image.astype(np.uint8), 13)
        # image_bg = image_bg[image_bg > 10]
        ret, thresh = cv2.threshold(image_bg, np.median(image_bg[image_bg > 10]), 255, cv2.THRESH_BINARY)
        
        thresh_cp = thresh.copy()
        
        
        
        # detect the contours on the binary image using cv2.ChAIN_APPROX_SIMPLE
        # draw contours on the original image for `CHAIN_APPROX_SIMPLE`
        
        
        # contours,hierarchy = cv2.findContours(thresh, 1, 2)
        contours1, hierarchy1 = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image = cv2.drawContours(image_bg, contours1, -1, 30, 2, cv2.LINE_AA)
        if len(contours1) == 0:
            self.recent_center_points.clear()
            return image
        cnt = max(contours1, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(image,(x,y),(x+w,y+h),128,2)
        center_point = (x+(w//2),y+(h//2))
        
        if center_point == (160, 120):
            return image
        print(center_point)
        cv2.circle(image,center_point,thickness=3,color=255,radius=3)
        
        self.recent_center_points.append(center_point)
        
        # print(np.average(self.recent_center_points[0,:].astype(np.int8)))
        # print(np.mean(self.recent_center_points, axis=0)[0])
        total = [total := i[0] for i in self.recent_center_points]
        if len(total) > 7:
            return image
        
        ix = center_point[0] - np.average(total).astype(np.int8)
        print(ix) #what ix is at the end of the motion is what matters. or we could get the average of ix over the range of movement.
        # avg_center=np.average(self.recent_center_points[0,:]).astype(np.int8)
        # print(avg_center)
        # ix = center_point[0] - avg_center[0]

        
        # self.recent_difference.append(abs(image))
        # diff_std =  np.std(image)
        # recent_diff_std = np.std(self.recent_difference)
        # bright_std_mean = np.std(np.mean(self.recent_frames, axis=0).astype(np.uint8))
        
        # self.recent_brightness.append(bright_std_mean)
        # # Print timestamp and image means to the camera.txt log files.
        # # print(self.ts2dt(time.time()),' Mean diff=', np.mean(self.recent_difference), 'Mean recent 10=', np.mean(np.mean(self.recent_frames, axis=0).astype(np.uint8)))
        # print (self.ts2dt(time.time()), 'Difference Std = ', np.std(image), 'Recent Diff Std = ', np.std(self.recent_difference), np.std(np.mean(self.recent_frames, axis=0).astype(np.uint8)))
        
        # # Write timestamp and image means to the log files.
        # # self.fp_labelfile.write('%.2f,hhhhh \n'%(time.time()))
        # # self.fp_logfile.write('%.2f,%.2f,%.2f\n'%(time.time(), 'Difference Std = ', np.std(image), 'Recent Diff Std = ',  np.std(self.recent_difference)))
        # # self.fp_logfile.write(self.ts2dt(time.time()), 'Difference Std = ', np.std(image), 'Recent Diff Std = ',  np.std(self.recent_difference))
        
        # if self.recent_brightness[-1] - self.recent_brightness[0]> 0.05 * self.recent_brightness[0]: # needs optimization
        #     print ("Brightness increased")
        #     time.sleep(0.75)
        # elif abs(self.recent_brightness[0] - self.recent_brightness[-1])> 0.05 * self.recent_brightness[0]: # needs optimization
        #     print ("Brightness decreased")
        #     time.sleep(0.75)
            
        # if diff_std >= 3 and recent_diff_std > diff_std:
        #     self.change_array = np.append(self.change_array, image)
        #     print ("Change detected")
        #     if self.change_array.size/(320*240) <= 5 and np.std(self.recent_brightness) > 2:
        #         print("Light On and Deviation is = ", np.std(self.recent_brightness))
        
        # if diff_std == 0 and recent_diff_std > diff_std:
        #     # if self.change_array.size <= 5:
        #     #     print("Light On")
        #     print ("change array size: ",self.change_array.size/(320*240))
        #     self.change_array = np.array([[]])
        
        return image
    
    def image_capture(self):
        ret, frame = cap.read()
        if ret:
            self.frame_time = cap.get(cv2.CAP_PROP_POS_MSEC)
            # print((time.time() - cap.get(cv2.CAP_PROP_POS_MSEC))/1000)
        
        image = self.image_process(frame)
        return image
        
    def image_show(self):
        while True:
            # Capture frame-by-frame
            
            image = self.image_capture()
            
            # if image is None:
            #     return
            
            # Display the resulting frame
            cv2.imshow('Webcam', image)

            # Check for the 'q' key to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

x = MotionDetector()
x.image_show()

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()