**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run vehicle detection pipeline on a video stream (start with the test_video.mp4 and later implement on full 
project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./media/output_images/draw_boxes.png
[image5]: ./media/output_images/bboxes_and_heat.png
[image6]: ./media/output_images/labels_map.png
[image7]: ./media/output_images/final_output.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in function called get_hog_features() (lines 6 through 23) of the file 
called `colorspace_gradient_utils.py` located in './src' directory. This function takes an input image and apply 
feature.hog() function from scikit-image library. I set all parameters required for it as global variables in config
.py file for easy configuration and experiments. 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.feature.hog()` parameters (`orientations`, 
`pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters (color space, HOG orient, HOG pixels per cell, HOG cells per block, HOG 
channels) and test them in my trained SVM and look for the best combination in the test score. I used the best 
parameters from my experiments in the config.py file. 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in the function called get_classifier_n_scaler() (line 47 to line 58)in the file 
called `train_classifier.py` located in './src/vehicle_detection' directory. 

This function first check whether the classifier file already exists in the directory. It'll use a already trained 
classifier if it exists in the directory. If not, it will train a new 
classifier by the function named train_classifier() in the same file. I set all parameters required for training a new 
classifier as global variables in config.py file for easy configuration and experiments. 

I trained a linear SVM using some sample car and non-car images from KITTI dataset (not full dataset), then I tested 
it in GTI dataset and found the test accuracy is not satisfactory. I also test it on the videos, and it cannot 
perform well. So I randomly take some images from GTI dataset into the training dataset and remove them from test 
dataset, the classifier performs better in the test dataset and also test videos, so I used this classifier for the 
final processing pipeline.   

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for implementing a sliding window search is in the function called detect_vehicle_by_clf() in 
./track_vehicle_single_frame.py file. I defined 3 searching scale windows (rectangles), for near car, middle distance 
car and cars far away. Their searching scales, window_sizes and window overlappings are defined in config.py as global 
variables. 

detect_vehicle_by_clf() function will first gets all sliding windows' coordinate locations, then it will exam whether
 a sliding window contains a car or not by using search_windows() function in the same file. If a sliding window is 
 predicted as a car, its coordinate location will be added to hot_windows list. 

For what scales to search and overlap windows, I first draw grids in test images and see how cars locate in the test 
images. I then decide to define 3 ranges: near, middle and far. Then I estimated how many pixels a car occupies in 
the 3 ranges. 

After got an estimated size, I defined overlap rate. The ideal overlap rate will make sure all pixels 
in the same range(near, middle, far) are equally searched. So the overlap rate can only be 0.5(search a pixel twice), 
0.67(search a pixel three times), 0.8(4 times) etc. Also, I want to emphasize nearby pixels since detecting nearby cars
 is more critical for a safe driving. 

After defining several sets of search ranges, window sizes and overlap rates, I tested them on my test images and 
test videos and picked the set of parameters which return the best results.  

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to try to minimize 
false positives and reliably detect cars?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of 
color in the feature vector, which provided a nice result.  I fine-tuned parameters to extract representing features, 
get an optimal classifier and conduct a useful sliding windows searching. For detecting vehicles in videos, I 
recorded positive detections in recent frames and combine current positive detection with recent ones to eliminate 
false positive.  

Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a 
heatmap and then thresholded (number of threshold are defined in config.py) that map to identify vehicle positions.  I 
then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob
 corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

