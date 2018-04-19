# Vehicle Detection and Tracking

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this project, the goal is to write a software pipeline to detect vehicles in a video.

The steps of this project are the following:

* Load Training Data
* Visualize training set data
* Define Method to Convert Image to Histogram of Oriented Gradients (HOG)
* Define a function to compute Color Histogram features
* Define a function to return Spatial Binning of Color features
* Define a function to extract features from a list of images
* Train and test the HOG Support Vector Classifier
* Train and test the Color Histogram Support Vector Classifier
* Train and the SVM classifier using all the features
* Sliding Window Implementation
* Adding Heatmaps and Bounding Boxes
* Pipeline to detect cars in a given frame
* Process video using the defined pipeline
* Add lane detection along with object detection and tracing

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

![alt text][image17]

![alt text][image16]

### Training and testing the HOG Support Vector Classifier and the Color Histogram Support Vector Classifier

I totally trained a 3 linear SVM classifiers.

After extracting HOG and color features from the **`car_images`** and **`noncar_images`** I test the accuracy of the SVC by comparing the predictions on labeled `X_train` data. 

Test Accuracy of HOG based SVC is 97.38%.
Test Accuracy of Color Histogram based SVC is 96.68%.

### Defining a function to extract features from a single image window

The **`single_img_features`** function is very similar to the *`extract_features`* function. One extracts HOG and color features from a list of images while the other extracts them from one image at a time. The extracted features are passed on to the **`search_windows`** function which searches windows for matches defined by the classifier. The following parameters were used to extact feautures of `cars` and `noncars` from the datasets.

```python

color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 10  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 64   # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```
Test Accuracy of SVC is 98.87%

### Sliding Window Search Implementation

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

The **`slide_window`** function takes in an image, start and stop positions, window size and overlap fraction and returns a list of bounding boxes for the search windows, which will then be passed to draw boxes. Below is an illustration of the **`slide_window`** function with adjusted `y_start_stop` values [400, 656].

![alt text][image13]

Different winows size is used to find cars in varying size because on its distance from the car.

![alt text][image14]

![alt text][image15]

---

### Video Implementation

### Defining a function that can extract features using HOG sub-sampling and make predictions

The **`find_cars`** function extracts the HOG and color features, scales them and then makes predictions. Using multiple scale values allows for more accurate predictions. I have combined scales of **`1.0, 1.5`** and **`2.0`** with their own `ystart` and `ystop` values to lower the ammount of false-postive search boxes. 

![alt text][image10]

![alt text][image11]

![alt text][image12]

### Adding Heatmaps and Bounding Boxes

The **`add_heat`** function creates a map of positive "car" results found in an image by adding all the pixels found inside of search boxes. More boxes means more "hot" pixels. The **`apply_threshold`** function defines how many search boxes have to overlap for the pixels to be counted as "hot", as a result the "false-positve" search boxes can be discarded. The **`draw_labeled_bboxes`** function takes in the "hot" pixel values from the image and converts them into labels then draws bounding boxes around those labels. Below is an example of these functions at work.

![alt text][image8]

![alt text][image9]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

