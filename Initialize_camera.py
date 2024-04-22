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
        self.change_array = [] #add the frames relevant to this change. CHANGE THIS TO JUST REGULAR ARRAY.
        self.num = 0
        self.total = 0
        self.ix = 0
        # self.frame_time = 


    def background_subtraction(self, frame):
        self.recent_frames.append(frame)
        avg_frame=np.min(self.recent_frames, axis=0).astype(np.int8)
        if self.frame_counter%30==0:
            self.second_avg_frames.append(avg_frame)
        # image = np.mean(self.recent_frames, axis=0).astype(np.uint8)
        
        return np.abs(frame - avg_frame)###
    
    
    def contour_process(self, frame):
        ret, thresh = cv2.threshold(frame, np.median(frame[frame > 10]), 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_ct = cv2.drawContours(frame, contours, -1, 30, 2, cv2.LINE_AA)
        if len(contours) == 0:
            return image_ct, None
        
        cnt = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(image_ct,(x,y),(x+w,y+h),128,2)
        center_point = (x+(w//2),y+(h//2))
        
        return image_ct, center_point
        
        
    def counter_process(self):
        # print ("no movement")
        self.total = [total := i[0] for i in self.change_array]
        
        self.ix = np.average(self.total[0:len(self.total)//2]).astype(np.int8) - np.average(self.total[len(self.total)//2:len(self.total)]).astype(np.int8)
        movement_size = len(self.total)
        
        # if movement_size > 0:
        #     print(movement_size)
        if movement_size > 0:
            print (self.total[-1], self.ix)
            if self.ix > 40 and self.total[-1]>240: #ix change positive + noise, last movement at the end of the frame
                self.num += 1
                print ("person in", self.ix, self.num)
                # time.sleep(1)
            
            elif self.ix < -40 and self.total[-1]<80: #ix change negative - noise, last movement at the beggening of the frame
                self.num -= 1
                print ("person out", self.ix, self.num)
                
            self.recent_center_points.clear()
            self.change_array.clear()
            
            
    def image_process(self, frame):
        # if time.time() - (self.frame_time) > 0.05: # if it's taking too much time to process the frame, skip it
        #     return None
        
        self.frame_counter+=1
        image_fg = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        image_fg = cv2.resize(image_fg, (320, 240))

        image_bs = self.background_subtraction(image_fg)
        image_med_b = cv2.medianBlur(image_bs.astype(np.uint8), 13)
        
        image_bs, center_point = self.contour_process(image_med_b)
        
        if center_point == None:
            return image_bs
        
        # still need to account one person stops at the middle. so record the end position and compare later movement if starts at the same place. 
        elif center_point == (160, 120):
            
            self.counter_process()
            return image_bs
        
        # print(center_point)
        cv2.circle(image_bs,center_point,thickness=3,color=255,radius=3)
        
        # self.recent_center_points.append(center_point)
        self.change_array.append(center_point)
        
        # print(np.average(self.recent_center_points[0,:].astype(np.int8)))
        # print(np.mean(self.recent_center_points, axis=0)[0])
        if len(self.change_array) < 10:
            return image_bs
        
        return image_bs
    
    
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