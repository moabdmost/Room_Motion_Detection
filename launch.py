from motion_detection import MotionDetector

#Setting the parameters
width=320
height=240
edge_thresh=0.25
ix_factor=0.1
thesh_cut=10
blur_size=21
n_stored_frames=7

# Initialize the MotionDetector object
Motion = MotionDetector(width, height, edge_thresh, ix_factor, thesh_cut, blur_size, n_stored_frames)

# Start the webcam feed and display motion detection results
Motion.image_show()