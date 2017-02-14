**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images. (But make sure 
we are using same type of cameras with same configurations)
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image. 
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./media/output_images/undistorted.png "Undistorted"
[image2]: ./media/output_images/test1.png "Road Transformed"
[image3]: ./media/output_images/binary_combo.png "Binary Example"
[image4]: ./media/output_images/warped_straight_lines.png "Warp Example"
[image5]: ./media/output_images/color_fit_lines.png "Fit Visual"
[image6]: ./media/output_images/example_output.png "Output"
[video1]: ./media/videos/proj_video_processed.mp4 "Video"

---

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in "get_camera_mtx_dist()" function in python file located in src/utility_funcs.py. 

I started by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. 
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for 
each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended 
with  a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be  
appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Then, I used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients 
by evoking the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image by using
`cv2.undistort()` function in cell 5 of pipleline.ipynb and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
For getting a distortion-corrected image, I reused the camera matrix and distortion coefficients got from the chess 
board undistortion and apply them to un-distort the test image by using cv2.undistort()  
 
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color (grayscale and S channel in HLS colorspace in specific) and gradient thresholds (Sober 
x, Sober y, gradient magnitude and gradient direction to be specific) to generate a binary image (thresholding 
functions are called inside detect_lane_line_pipeline function in pipeline.ipynb).  Here's an example of my output for 
this step. 
![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

In cell 3 of pipeline.ipynb, I defined all constant variables needed in the project. I also defined  
The code for my perspective transform is line 10 'warped = cv2.warpPerspective(threshold_img, M, (x,y))' in 
detect_lane_line_pipeline function inside pipeline.ipynb file. the cv2.warpPerspective function takes as inputs an 
image (`threshold_img`), as well as source (`src`) and destination (`dst`) points.  I chose to hardcode the source 
and destination points in the following manner:

```
# defined in cell 3 of pipeline.ipynb
x = 1280
y = 720
offset_top = 590
offset_btm = 200
offset_dst = 300 
src = np.float32([(offset_top, 450), (x-offset_top, 450), (x-offset_btm, y), (offset_btm, y)])
dst = np.float32([(offset_dst, 0), (x-offset_dst, 0), (x-offset_dst, y), (offset_dst, y)])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 590, 450      | 300, 0        | 
| 690, 450      | 980, 0      |
| 1080, 720     | 980, 720      |
| 200, 720      | 300, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After got the warped image from perspective transform, I wrote a function named get_sliding_window() in 
src/utility_funcs.py to get all interested patches in the image. In the get_sliding_window function, I first generate
a histogram on the bottom 1/3 of the warped image, then get 2 highest peaks which standing for 2 lanes in the bottom
of the image by using a function called get_bottom_lane_marks function. 
  
Then starting from the bottom 2 lanes 
  marks, I used sliding windows (height = 120px) to detect lane marks above the already detected lane parts (only areas 
  around detected marks are checked to save some computation). By repeating the sliding windows row by row, finally I
   checked the whole image and draw the 2 lane lines. The whole step is done by find_lane_patches() function in 
   src/utility_funcs.py. 
   
Up to this point, all interested patches(width=120, height=120) are detected, then I get all 4 corner positions from
all patches collected. I used numpy.polyfit() to fit my lane lines with a 2nd order polynomial. The code for this 
step is from line 13 to line 23 in detected_lane_line_pipeline function inside pipeline.ipynb. 

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Since we already got polynomials for the 2 lane lines, we can calculate the curvature for the lines in the image. After
 we get the curvature in the image, we can transfer it back into real world. As our test image is taken on U.S. 
 road, and we can assume our interested lines are about 30 meters long and 3.7 meters wide(Not exactly, but roughly). 
 Therefore, we can calculate how long a pixel stands for. I defined ym_per_pix and xm_per_pix for this purpose in pipeline.ipynb.
  For the offset from lane center, I assumed camera is amounted at the center of the car. Therefore, I get the offset
   of lane center with image center, then convert it back to real world offset in meters. Lines of code are in  
   cal_offset_from_lane_center function in src/utility_funcs.py.  
 
 The code for this step is in get_real_world_curvature() function in src/utility_funcs.py file. 
 

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in line 49 of detect_lane_line_pipeline() function in pipeline.ipynb by using cv2
.warpPerspective() function with a inverse matrix from previous perspective transform. Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/EqzrqlN2y6s)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- Problems/issues:
    - In perspective transform, source and destination polygon positions are hard coded. It will be inaccurate if the
     road is very curvy. 
    - Used only one polynomial fit for a single lane line. If the road is twisted (left turn then right turn in a 
    short distance), one polynomial fit will not be sufficient. 
    - Assumed road of interest is 3.7m wide and 30m long, which may not always be that case. 

- Pipeline likely fail:
    - Road is too curvy.
    - Road is twisted. 
    - Very narrow lane or too wide lane.
    - Very Strong light. 
    - Lane lines are not distinguishable.  

- Ways to make it more robust and future improvement:
    - Using lane line class (already used in my implementation). By using the lane class, we can keep track of 
    detected lane lines in previous frame. Therefore, we can be more confident if current detection is consistent 
    with previous data. 
    - Add a confidence level attributes and functions in lane line class. For new frame, we detect its lane lines and 
    estimate how confident we are about the prediction.
    - Dynamically decide perspective transform parameters and auto detect road width.
