
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
[video1]: ./project_video.mp4

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

The HSV color space was used, as good results were obtained during the lessons with this color space. Other parameters such as `orientations`, `pixels_per_cell`, `cells_per_block`, `hog_channel`, `spacial size` and `hist_bins` were explored.  

A linear SVC was used and the following results were obtained

Using: 18 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 11448
3.22 Seconds to train SVC...
Test Accuracy of SVC =  0.9932


#### 2. Explain how you settled on your final choice of HOG parameters.

Various combinations of parameters and color spaces were explored. These are the final parameters used:

```python
color_space = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb ## HSV
orient = 18  # HOG orientations ## 18
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

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

