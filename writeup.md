
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/carposition.png
[image2]: ./output_images/data_vis.png
[image3]: ./output_images/data_vis2.png 
[image4]: ./output_images/prediction1.png
[image5]: ./output_images/im1.png
[image6]: ./output_images/heatmap.png
[image7]: ./examples/output_bboxes.png
[video1]: ./output_video.mp4

### [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
#### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

This will serves as the writeup/ README.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the IPython notebook. The functions were obtained from the Udacity course material.

All the `vehicle` and `non-vehicle` images were read in.  Here is an example of one of each of the `vehicle` and `non-
vehicle` classes:

`vehicle` data visualization:

<img src="https://github.com/BVG85/Project-5-Vehicle-Detection-Tracking/blob/master/output_images/data_vis.png" width="200" height="200" /> 

`non-vehicle` data visualization:

<img src="https://github.com/BVG85/Project-5-Vehicle-Detection-Tracking/blob/master/output_images/data_vis2.png" width="200" height="200" />

The HSV color space was used initially, as good results were obtained during the lessons with this color space. However after an initial review this was changed to YCrCb with better results. Other parameters such as `orientations`, `pixels_per_cell`, `cells_per_block`, `hog_channel`, `spacial size` and `hist_bins` were explored.  

A linear SVC was used and the following results were obtained
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 6156
11.7 Seconds to train SVC...
Test Accuracy of SVC =  0.9882


#### 2. Explain how you settled on your final choice of HOG parameters.

Various combinations of parameters and color spaces were explored. These are the final parameters used:

```python
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb ## HSV
orient = 9  # HOG orientations ## 18
pix_per_cell = 8 # HOG pixels per cell ## 8
cell_per_block = 2 # HOG cells per block ## 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [350, 650] # Min and max in y to search in slide_window()
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

A linear SVM was trained using HOG and color features, with parameters mentioned above.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search was performed over a single image. Parameters were adjusted till results were satisfactory.
The implementation over a single image can be seen below:

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The `find_cars` function was used, with a YCrCb color space, utilizing all the channels of HOG features, as well as spatially binned color and histogtams of color. The cells_per_step were reduced to increase the overlap. Here are some example images:

![alt text][image5]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_video_reSub.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Heat maps and thresholding were utilised to minimize false positives. `scipy.ndimage.measurements.label()` was used to identify individual blobs in the heatmap. 

Here's an example result showing the heatmap and resulting bounding box:
![alt text][image6]

![alt text][image1]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I struggled to generate a robust pipeline. False positives still show up in the video feed. Also, at times the pipeline fails to identify true positives. More exploration will be done with thresholding, as well as different colorspaces.

