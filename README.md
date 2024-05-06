# Object Passage Counter using Motion Detection with OpenCV

## Introduction
The MotionDetector class is designed to detect motion in a video stream captured by a webcam. It performs background subtraction, contour processing, and state transition analysis to determine the movement of objects across predefined areas within the frame. This can be useful for applications such as counting the number of people entering or exiting a room, monitoring traffic flow, or analyzing crowd movement. Additionally, `plot_motion.py` allows for the visualization of motion data through scatter plots using NumPy and Matplotlib.

## Features
- Real-time motion detection using webcam feed.
- Adjustable parameters for fine-tuning detection sensitivity.
- Object tracking and counting based on movement direction and position within the frame.
- Logging of motion data, including timestamps, object positions, and movement states.
- Visualization of motion data through scatter plots using NumPy and Matplotlib

  ## Authors
- Mohamed Mostafa
- John Matsudaira

![image](https://github.com/moabdmost/Room_Motion_Detection/assets/72043625/d0c60052-f207-47d1-bfa0-8ccccff11b6c)
![image](https://github.com/moabdmost/Room_Motion_Detection/assets/72043625/709684b9-5c87-4c73-8eda-37c5c0d0130f)

## Requirements
- Python 3.11
- OpenCV (`pip install opencv-python`)

## Usage
1.  Run `launch.py` OR
2. Import the `MotionDetector` class from the `motion_detector` module.
3. Initialize the `MotionDetector` object with desired parameters.
4. Call the `image_show()` method to start the webcam feed and display motion detection results.

```python
from motion_detector import MotionDetector

# Initialize the MotionDetector object
Motion = MotionDetector()

# Start the webcam feed and display motion detection results
Motion.image_show()
```

## Parameters
- `width` (int): The width of the video frame.
- `height` (int): The height of the video frame.
- `edge_thresh` (float): The threshold for motion detection edge.
- `ix_factor` (float): Factor used for motion detection velocity.
- `thesh_cut` (int): Threshold for background subtraction to binary image conversion.
- `blur_size` (int): Size of the blur for image processing.
- `n_stored_frames` (int): Number of frames stored in the recent_frames deque.

## Methods
- `background_subtraction(frame)`: Perform background subtraction on a given frame.
- `contour_process(frame)`: Process the contours in a given frame.
- `state_transition(num_change, state_change, direction_change, movement, q)`: Apply required changes in movement data and print the movement information.
- `state_machine(center_point)`: Process the movement based on the cross-movement center points.
- `log_process(center_point)`: Logs the motion data, current state, and direction.
- `image_process(frame)`: Process the input frame by converting to grayscale, resizing, performing background subtraction, and contour processing.
- `image_capture()`: Captures an image from the camera and processes it.
- `image_show()`: Displays the captured frames from the camera in a window.

## License
This project is licensed under the [MIT License](LICENSE).
