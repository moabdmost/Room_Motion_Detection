import cv2
import collections
import numpy as np
import time, datetime

# Initializing webcam
cap = cv2.VideoCapture(0)

class MotionDetector():

    def __init__(self, width=320, height=240, edge_thresh=0.25, ix_factor=1/8, thesh_cut=10, blur_size=17, n_stored_frames=7):
        
        self.frame_counter=0
        self.rs_width = width
        self.rs_height = height
        self.mid_point = (self.rs_width / 2, self.rs_height / 2)
        MOVEMENT_EDGE_THRESH = edge_thresh
        self.low_edge = MOVEMENT_EDGE_THRESH * self.rs_width
        self.high_edge = (1 - MOVEMENT_EDGE_THRESH) * self.rs_width
        self.ix_thresh = self.rs_width * ix_factor
        self.thresh_cut = thesh_cut
        self.blur_size = blur_size
        self.recent_frames = collections.deque(maxlen=n_stored_frames)
        
        # self.recent_center_points = collections.deque(maxlen=7)
        self.ts2dt = lambda timestamp: datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        self.change_array = [] #contains the frames relevant to this change.
        self.num = 0
        self.all_x = 0
        self.ix = 0
        


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
        ret, thresh = cv2.threshold(frame, np.median(frame[frame > self.thresh_cut]), 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_ct = cv2.drawContours(frame, contours, -1, 30, 2, cv2.LINE_AA)
        if len(contours) == 0:
            return image_ct, None
        
        cnt = max(contours, key = cv2.contourArea) # find the largest contour
        x,y,w,h = cv2.boundingRect(cnt) # x,y is top left of the bounding rectangle
        cv2.rectangle(image_ct,(x,y),(x+w,y+h),128,2)
        center_point = (x+(w//2),y+(h//2))
        
        return image_ct, center_point
        
        
    def counter_process(self):
        """
        A function that processes the movement based on the array of cross-movement center points.
        """
        # print ("no movement")
        self.all_x = [x_coord := i[0] for i in self.change_array] #pull x coordinates from the change array
        
        # Difference between average change in x coordinates of the first and second halfs of the array.
        self.ix = np.average(self.all_x[0:len(self.all_x)//2]).astype(np.int16) - np.average(self.all_x[len(self.all_x)//2:len(self.all_x)]).astype(np.int16)
        movement_size = len(self.all_x)
        last_movement = np.average(self.all_x[(len(self.all_x)*3)//4:len(self.all_x)]).astype(np.int16)
        
        if movement_size > 0:
            #print(movement_size) 
            # print (self.all_x[-1], self.ix)
            
            print ("avg last movement: ",last_movement," ix velocity:",self.ix)
            if self.ix < -self.ix_thresh and (last_movement>self.high_edge): #ix change -> positive + noise, last movement at the end of the frame
                self.num += 1
                print ("IN ", self.ts2dt(time.time()), " Direction: ", self.ix, " Counter: ", self.num)
                # self.change_array.clear()
                # time.sleep(1)
            
            elif self.ix > self.ix_thresh and (last_movement<self.low_edge): #ix change -> negative - noise, last movement at the beginning of the frame
                # print(-self.ix_thresh)
                self.num -= 1
                print ("OUT ", self.ts2dt(time.time()), " Direction: ", self.ix, " Counter: ", self.num)
                # self.change_array.clear()
                
            # self.recent_center_points.clear()
            self.change_array.clear()
            
            
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
        
        
        
        if center_point == None: #no contours (motion) on the background subtracted image
            return image_med_b
        
        # still need to account one person stops at the middle. so record the end position and compare later movement if starts at the same place. 
        elif center_point == self.mid_point: #One contour of the whole frame and the center is at the middle of the frame, meaning movement stopped.
            if len(self.change_array) < 20: #not enough for judgement
                return image_med_b
            
            self.counter_process()
            return image_med_b
        
        # print(center_point)
        cv2.circle(image_med_b,center_point,thickness=3,color=255,radius=3)
        
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
            # print((time.time() - cap.get(cv2.CAP_PROP_POS_MSEC))/1000)
        
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
ix_factor=1/4
thesh_cut=10
blur_size=17
n_stored_frames=7

#Initialize the MotionDetector object
Motion = MotionDetector(width, height, edge_thresh, ix_factor, thesh_cut, blur_size, n_stored_frames)
Motion.image_show()

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()