# Vehicle Detection and Tracking

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this project, the goal is to write a software pipeline to detect vehicles in a video.

The steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png

[image8]: ./output_images/heat_map_2.png
[image9]: ./output_images/heat_map_1.png
[image10]: ./output_images/find_cars_3.png
[image11]: ./output_images/find_cars_2.png
[image12]: ./output_images/find_cars_1.png
[image13]: ./output_images/sliding_window_3.png
[image14]: ./output_images/sliding_window_2.png
[image15]: ./output_images/sliding_window_1.png
[image16]: ./output_images/color_bin_vis_2.png
[image17]: ./output_images/color_bin_vis_1.png
[image18]: ./output_images/color_bin_vis.png
[image19]: ./output_images/hog_sample_2.png
[image20]: ./output_images/hog_sample_1.png
[image21]: ./output_images/training_sample.png
[image22]: ./output_images/color_hist_vis.png


[video1]: ./project_video.mp4


---

### Loading and Visualizing the data

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images. A snapshot of the training set

![alt text][image21]

### Defining a function to return HOG features and visualization

The **`get_hog_features`** function takes in an image and computes the Histogram of Oriented Gradient (HOG) features in it using the **`hog()`** function from the [scikit-image](http://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=feature%20hog#skimage.feature.hog) package. Below is the visualization of the **`get_hog_features`** function.

![alt text][image19]

![alt text][image20]

### Defining a function to compute Color Histogram features and visualizing the results

The **`color_hist`** function computes Color Histogram features labeled **`hist_features`**. This function returns concatenated color channels by default and separate color channels if the **`vis == True`** flag is called. Below is the visualization of the **'R' 'G' and 'B'** channels from a random `car_image`.

![alt text][image22]

### Defining a function to return Spatial Binning of Color features and visualizing the results

The **`bin_spatial`** function takes in an image, a color space, and a new image size and returns a feature vector. Useful for extracting color features from low resolution images. Below is an example of spatially binned color features extracted from an image before and after resizing. 

![alt text][image18]

![alt text][image17]

![alt text][image16]

#### 3. Train a classifier

I trained a linear SVM using...

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

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

