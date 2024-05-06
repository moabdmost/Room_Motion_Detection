""" 
This is the class implementation for object passage counter using motion detection techniques based on image processing with OpenCV.

Author: Mohamed Mostafa
Author: John Matsudaira
"""

import os
import time, datetime
import numpy as np
import collections
import cv2

# Initializing webcam
cap = cv2.VideoCapture(0)

class MotionDetector():

    def __init__(self, width=320, height=240, edge_thresh=0.25, ix_factor=1/8, thesh_cut=10, blur_size=17, n_stored_frames=7):
        """
        Initializes the MotionDetector object with default or user-defined parameters.

        Parameters:
            width (int): The width of the video frame.
            height (int): The height of the video frame.
            edge_thresh (float): The threshold for motion detection edge.
            ix_factor (float): Factor used for motion detection velocity.
            thesh_cut (int): Threshold for background subtraction to binary image conversion.
            blur_size (int): Size of the blur for image processing.
            n_stored_frames (int): Number of frames stored in the recent_frames deque.
        
        Returns:
            None
        """
        
        self.frame_counter=0
        self.rs_width = width
        self.rs_height = height
        self.mid_point = (self.rs_width / 2, self.rs_height / 2)
        MOVEMENT_EDGE_THRESH = edge_thresh
        self.low_edge = MOVEMENT_EDGE_THRESH * self.rs_width
        self.high_edge = (1 - MOVEMENT_EDGE_THRESH) * self.rs_width
        # self.ix_thresh = self.rs_width * ix_factor # could be used to limit detection of motion to be based on a certain velocity.
        self.thresh_cut = thesh_cut # the threshold for converting the background subtraction to a binary image.
        self.blur_size = blur_size
        self.recent_frames = collections.deque(maxlen=n_stored_frames)
        
        self.ts2dt = lambda timestamp: datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        self.change_array = [] # contains the frames relevant to this change.
        self.state = 0
        self.direction = 0
        self.num = 0
        self.all_x = 0
        self.ix = 0
        if os.path.exists('log.txt') == True:
            os.remove('log.txt')


    def background_subtraction(self, frame):
        """
        Perform background subtraction on a given frame.

        Parameters:
            self: The object itself.
            frame: The input frame for background subtraction.

        Returns:
            Numpy image array: The result of the background subtraction.
        """
        
        self.recent_frames.append(frame)
        min_frame=np.min(self.recent_frames, axis=0).astype(np.int8)
        
        return np.abs(frame - min_frame)
    
    
    def contour_process(self, frame):
        """
        Process the contours in a given frame.

        Parameters:
            self: The object itself.
            frame: The input frame for contour processing.

        Returns:
            Tuple: A tuple containing the processed frame with contours drawn and the center point of the largest contour.
                   If no contours are found, the processed frame is returned with a value of None for the center point.
        """
        
        # threshold for values greater than the median above thresh_cut (10), since most of the bs_image is dark and has low values.
        _, thresh = cv2.threshold(frame, np.median(frame[frame > self.thresh_cut]), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_ct = cv2.drawContours(frame, contours, -1, 30, 2, cv2.LINE_AA)
        if len(contours) == 0:
            return image_ct, None
        
        cnt = max(contours, key = cv2.contourArea) # find the largest contour
        x,y,w,h = cv2.boundingRect(cnt) # x,y is top left of the bounding rectangle
        cv2.rectangle(image_ct,(x,y),(x+w,y+h),128,2)
        center_point = (x+(w//2),y+(h//2))
        
        return image_ct, center_point
        
    
    def state_transition(self, num_change, state_change, direction_change, movement, q):
        """
        A function that applies required changes in movement data and prints the movement information.

        Parameters:
            self: The object itself.
            num_change: The change in the number of objects.
            state_change: The change of state.
            direction_change: The change in the direction.
            movement: The movement information to be printed.
            q: The side/quantile information to be printed.

        """
        self.num = self.num + num_change
        self.state = state_change
        self.direction = direction_change
        if num_change != 0:
            print (movement + self.ts2dt(time.time())," Counter: ", self.num)
        else:
            print(movement + q)  
    
    
    def state_machine(self, center_point):
        """
        A function that processes the movement based on the cross-movement center points.
        """
        
        self.all_x = [x_coord := i[0] for i in self.change_array] #Pull x coordinates from the change array
        # Difference between average change in x coordinates of the first and second halfs of the array.
        self.ix = np.average(self.all_x[0:len(self.all_x)//2]).astype(np.int16) - np.average(self.all_x[len(self.all_x)//2:len(self.all_x)]).astype(np.int16)
        
        if center_point == self.mid_point: #No movement. One contour of the whole frame and the center point is at the middle point of the frame.
            self.change_array.clear()
            center_point = (0,0)
        else:
            self.log_process(center_point[0])
        
        center_point_x = center_point[0]
        
        if self.direction == 2:
            center_point_x = self.rs_width - center_point_x
        
        if self.state == 0 : # Initial state
            if 0 < center_point_x <= self.low_edge and self.ix  < 0 :
                self.state_transition(num_change = 0, state_change = 1, direction_change = 1, movement = "ENTERED ", q = "1ST RIGHT SIDE")
                
            elif self.rs_width >= center_point_x > self.high_edge and self.ix  > 0:
                self.state_transition(num_change = 0, state_change = 1, direction_change = 2, movement = "ENTERED ", q = "1ST LEFT SIDE")            
        
        elif self.state == 1 : # 1st side of the frame
            
            if self.low_edge < center_point_x < self.high_edge:
                self.state_transition(num_change = 0, state_change = 2, direction_change = self.direction, movement = "SPOTTED IN ", q = "MIDDLE")
                
            elif center_point == (0,0):
                self.state_transition(num_change = 0, state_change = 0, direction_change = 0, movement = "RESET ", q = " ")
                
        elif self.state == 2 : # Middle of the frame
            
            if self.rs_width > center_point_x > self.high_edge:
                self.state_transition(num_change = 0, state_change = 3, direction_change = self.direction, movement = "ENTERED ", q = "OPPOSITE SIDE")  
                
            elif 0 < center_point_x < self.low_edge:
                self.state_transition(num_change = 0, state_change = 1, direction_change = self.direction, movement = "BACK TO ", q = "1ST SIDE") 
                
        elif self.state == 3: # 3rd side of the frame
            if center_point == (0,0):
                
                if self.direction == 1:
                    self.state_transition(num_change = 1, state_change = 0, direction_change = 0, movement = "IN ", q = " ")
                    
                if self.direction == 2:
                    self.state_transition(num_change = -1, state_change = 0, direction_change = 0, movement = "OUT ", q = " ")
                    
            elif self.low_edge < center_point_x < self.high_edge:
                self.state_transition(num_change = 0, state_change = 2, direction_change = self.direction, movement = "BACK TO ", q = "MIDDLE") 
    
            
    def log_process(self, center_point):
        """
        Logs the motion data, current state, and direction.

        Parameters:
            center_point (tuple): The coordinates of the center point of the motion detected.

        """
        with open('log.txt', 'a') as f:
            f.write(str(time.time()) + ', ' + str(center_point) + ', ' + str(self.state) + ', ' + str(self.direction) + '\n')
    
    
    def image_process(self, frame):
        """
        Process the input frame by converting to grayscale, resizing, performing background subtraction, and contour processing.
        
        Parameters:
            self: The object itself.
            frame: The input frame to be processed.
        
        Returns:
            numpy image array: The processed image after background subtraction, blurring, and contour processing.
        """
        
        self.frame_counter+=1
        image_fg = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        image_fg = cv2.resize(image_fg, (self.rs_width, self.rs_height))

        image_bs = self.background_subtraction(image_fg)
        image_med_b = cv2.medianBlur(image_bs.astype(np.uint8), self.blur_size)
        
        image_bs, center_point = self.contour_process(image_med_b)
        
        if center_point == None: #No contours (motion) on the background subtracted image
            self.change_array.clear()
            return image_med_b
        
        if center_point != self.mid_point:
            cv2.circle(image_med_b,center_point,thickness=3,color=255,radius=3)
        
        self.state_machine(center_point)
        
        self.change_array.append(center_point)
                
        return image_med_b
    
    
    def image_capture(self):
        """
        Captures an image from the camera and processes it.
        The function then calls the `image_process()` method to process the captured frame and returns the processed image.

        Returns:
            numpy.ndarray: The processed image.
        """
        
        ret, frame = cap.read()
        if ret:
            self.frame_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        image = self.image_process(frame)
        return image
        
        
    def image_show(self):
        """
        Displays the captured frames from the camera in a window.
        """
        while True:
            
            # Capture frame-by-frame
            image = self.image_capture()
            
            # Display the resulting frame
            cv2.imshow('Webcam', image)

            # Check for the 'q' key to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


#Setting the parameters.
width=320
height=240
edge_thresh=0.25
ix_factor=0.1
thesh_cut=10
blur_size=21
n_stored_frames=15

#Initialize the MotionDetector object
Motion = MotionDetector(width, height, edge_thresh, ix_factor, thesh_cut, blur_size, n_stored_frames)
Motion.image_show()

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()