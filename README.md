**Vehicle Detection Project**

### Demonstration 
You can see the demonstration video on [YouKu(优酷)](http://v.youku.com/v_show/id_XMjUzNDgyMDk0MA) or on [YouTube](https://youtu.be/qlgsulEVgxo) by click the gif below.

[![Demo CountPages alpha](./media/output.gif)](https://youtu.be/qlgsulEVgxo)

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run vehicle detection pipeline on a video stream (start with the test_video.mp4 and later implement on full 
project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./media/output_images/car_not_car.png
[image2]: ./media/output_images/hogs.png

[image4]: ./media/output_images/draw_boxes.png
[image5]: ./media/output_images/bboxes_and_heat.png
[image6]: ./media/output_images/labels_map.png
[image7]: ./media/output_images/final_output.png
[video1]: ./project_video.mp4

### Demonstration 

### How to execute the project:
python main.py

####1. How did I extract HOG features from the training images?

The code for this step is contained in function called get_hog_features() (lines 6 through 23) of the file called 
`colorspace_gradient_utils.py` located in './src' directory. This function takes an input image and apply feature.hog
()  function from scikit-image library. I set all parameters required for it(orientation numbers, pixel per cell, 
pixel 
per bolck) as global variables in config.py file for easy configuration and fast experiments. 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.feature.hog()` parameters (`orientations`, 
`pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them 
to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and
 `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how I settled on your final choice of HOG parameters.

I tried various combinations of parameters (color space, HOG orientation numbers, HOG pixels per cell, HOG cells per 
block, HOG channels used) and test them in my trained SVM and look for the best combination in the test score. I put
 the best parameters from my experiments in the config.py file for easy modification. So far I'm using the `YCrCb` 
 color space Y channel and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`.

####3. Describe how (and identify where in your code) I trained a classifier using your selected HOG features (and 
color features if you used them).

The code for this step is contained in the function called get_classifier_n_scaler() (line 47 to line 58)in the file 
called `train_classifier.py` located in './src/vehicle_detection' directory. 

This function first check whether the classifier file already exists in the directory. It'll use a already trained 
classifier if it exists in the directory. If not, it will train a new classifier by the function named 
train_classifier() in the same file. I set all parameters required for training a new 
classifier as global variables in config.py file for easy configuration and fast experiments. 

For the model training process, I first extract features from [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) images by using 
extract_features() function located in src/vehicle_detection/feature_extraction.py file. I combined and normalized HOG features (from Y channel in YCrCb 
color space), image bin spatial features and color histogram features as my image features. Then I used a 
recursive feature elimination and cross-validated selection (RFECV) to get the best 30% features. After that, 
I trained a linear SVM model using the selected features, then tested it in test dataset. I fine tuned the parameters
 for RFECV and linear SVM to get the highest score on my test dataset. 
 
 After fine tuned the parameters in my classifier, I used the parameters to re-trained my classifier using all images 
 available (including the test images) to get a better classifier. 

###Sliding Window Search

####1. Describe how (and identify where in my code) I implemented a sliding window search.  How did I decide what 
scales to search and how much to overlap windows?

The code for implementing a sliding window search is in the function called detect_vehicle_by_clf() in 
./track_vehicle_single_frame.py file. I defined 3 searching scale windows (rectangles), for near car, middle distance 
car and cars far away. Their searching scales, window_sizes and window overlappings are defined in config.py as global 
variables. 

detect_vehicle_by_clf() function will first gets all sliding windows' coordinate locations, then it will exam whether
 a sliding window contains a car or not by using search_windows() function (The trained linear SVC classifier) in the 
 same file. If a sliding window is predicted as a car, its coordinate location will be added to hot_windows
 (car_windows) list. 

For what scales to search and overlap windows, I first draw grids in test images and see how cars locate in the test 
images. I then decide to define 3 ranges: near, middle and far. Then I estimated how many pixels a car occupies in 
the 3 ranges. (All parameters defined in config.py file)

After got an estimated size, I defined overlap rate. The ideal overlap rate will make sure all pixels 
in the same range(near, middle, far) are equally searched. So the overlap rate can only be 0.5(search a pixel twice), 
0.67(search a pixel three times), 0.8(4 times) etc. Also, I want to emphasize nearby pixels since detecting nearby cars
 is more critical for a safe driving. 

After defining several sets of search ranges, window sizes and overlap rates, I tested them on my test images and 
test videos and picked the set of parameters which return the best results (Defined in config.py file).  

![alt text][image3]

####2. Show some examples of test images to demonstrate how my pipeline is working.  What did I do to try to 
minimize false positives and reliably detect cars?

Ultimately I searched on three scales using YCrCb Y-channel HOG features plus spatially binned color and histograms of 
color in the feature vector, which provided a nice result. I fine-tuned parameters to extract representing features, 
get an optimal classifier and conduct a useful sliding windows searching. For detecting vehicles in videos, I 
recorded positive detections in recent frames and combine current positive detection with recent ones to eliminate 
false positive. The code for recording recent frame hotmap is defined in 
src/vehicle_detection/VehicleDetectionHeatMap.py file. The number for tracked recent frames and heatmap threshold is 
defined in config.py file. 

For optimize the performance of my classifier, I designed a searching strategy which can reduce sliding windows 
searched. I try to search extensively in near range, since it's more safety critical. I also used larger sliding 
window for near range, as near cars appears bigger in images. I reduced sliding windows overlapping rate, and defined
 relatively bigger sliding window in all ranges to reduce searches needed. 

Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./media/videos/vehicle_tracking_and_lane_line_marking.mp4) or 
on [YouTube](https://youtu.be/qlgsulEVgxo) 


####2. Describe how (and identify where in my code) I implemented some kind of filter for false positives and some 
method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a 
heatmap and then thresholded (number of threshold are defined in config.py) that map to identify vehicle positions.  I 
then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob
 corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is a sample heatmap:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from the last frame:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues I faced in my implementation of this project.  Where will my 
pipeline likely fail?  How to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
In my implementation, there are mainly 4 parts: Extract representative features, train an optimal classifier, design 
an effective sliding window searching strategy and apply filters to reject false positives.  
- Problems/issues:
    - Training vehicle dataset is relatively small. 17760 vehicle and non-vehicle training images are used to get a 
    useful classifier. If a larger dataset is used for training classifier, we can get a classifier with lower error 
    rate. Then, the number of sliding window can be largely reduced to speed up the training. Also, the false 
    positive will be less. 
    - I get the final bounding box by combining hotmaps from recent frames, and it works fine for vehicle in the same
     direction. But it worked poorly if a car is coming from the opposite direction, since its coordinate location 
     will be quite different from frame to frame. A different filter strategy is needed for this case. 
    - My output still have some false positives. A better sliding window setting will improve the result.
- Where the pipeline likely fail:
    - For cars coming from different direction (mentioned above). 
    - The slope is steep, as the searching range and vehicle size will change. 
    - Strong/dark light. 
- How to make it more robust:
    - A better sliding window strategy. 
    - Larger vehicle and non-vehicle training set.
    - Better filter strategy, such as weighted average for recent frames. 
- More after thoughts:
    - Fine tune all related parameters in the pipeline step is essential to get a useful pipeline. And the optimal 
    parameters are gotten from lots of experiments, therefore how fast one can perform experiments and improve the 
    pipeline from the experiment results is critical to get a working pipeline.   
