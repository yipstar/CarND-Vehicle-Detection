**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

All jupyter notebook cells referenced below are contained in `./Vehicle Detection.ipynb`.

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

My main function for training the classifier lives in the `./search_classify.py` file in the function `train_classifier` (lines 112-189). Lines 118-119 read in the filenames for the `cars` and `notcars` arrays respectively. Lines 127-153 extract features for both the cars and notcars images by calling out to the `extract_features` function in the file `lesson_functions.py`. The `extract_features` function is just a direct copy from the lesson code. The only change I made was to use cv2.imread everywhere instead of mpimg.imread for reading in images so I didn't run into pixel range issues.

Cell 4 contains a random example of both a car and noncar image.

I then explored different color spaces on random example car images to see if a strong pattern stood out and would give me an intuition that a particular color space would make a good basis for the feature vector creation.

Cells 5-7 show 3 examples of cars plotted in 3d in RGB, HSV, HLS, and LUV color spaces. The LUV color space showed a clear similar pattern in 3d space of pixels and so it became a strong candidate for the color space I would eventually use for feature extraction.

I then explored 2 different sets of `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) on the U channel of the LUV converted color space example image.

Cell 14 shows hog features with the parameters presented in the lesson (`orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)). Cell 15 used (`orientations=8`, `pixels_per_cell=(16, 16)` and `cells_per_block=(4, 4)`.

####2. Explain how you settled on your final choice of HOG parameters.

Cell 14 (the parameters presented in the lesson) were clearly superior and I choose to stick with them and move on further into the project, and if needed do more exploration later should my pipeline prove to be inaccurate. In addition I decided to use the test performance of the trained classifier as my basis for choosing which specific feature parameters to use.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The `train_classifier` function (lines 112-189)  in `./search_classify.py` is where in the code I trained my classifier. I trained a linear SVM using the LUV color space and ALL the color channels for the hog channel. I also used spatial features downsampling to (32,32) image size, and color histogram features using 32 bins as described in the lesson. The total number of features was 8460. This combination of features gave me the best test accuracy against the test set at 99.47%. As mentioned above I tried a variety of different color spaces and parameter settings and used the test performance metric as my basis for choosing this combiation of feature parameters.

Cell 16 shows the classifer's test accuracy.

### Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The `tracking_pipeline` function (lines 191-270 in `search_classify.py`) contains the code the takes an image frame, the trained classifier, and the feature parameters and performs the sliding window search. I used 4 different window sizes [(64, 64), (96, 96), (128, 128), (192,192) ]. All of them overlapped at 75 or 80%. The smaller window sizes were restricted to search only around the horizon y coordinates. Each window size searched a larger portion of the bottom half of the image then its predecssor size, since smaller windows would mach cars farther away from the camera (closer to the horizon) and larger windows would match cars closer the camera (the bottom of the image).

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Cells 17-19 contains examples of the hot windows found from the sliding window process on 3 of the test images. I tried different combinations of window sizes and areas to optimize the performance of my classifier.

To improve the accuracy of my classifier I trained several SVM's using different color spaces and feature parameters and choose the one described here which gave me the best accuracy on the test set.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

My final video lives in `project_video.mp4` and can be viewed in Cell 33.

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

My `tracking_pipeline` function (Lines 230-271) recorded the postions of positive detections in each frame of the video. From these hot windows I created a heatmap and then thresholded that map as described in the lesson. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  Assuming each blob corresponded to a vehicle, bounding boxes were drawn to cover the area of each blob detected.

Cells 21 and 23 show examples of 2 heatmaped test images.

Cells 24 and 26 show examples of these 2 test images with the final bounding boxes obtained by using `scipy.ndimage.measurements.label()` on the heatmapped images.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My approach was to first get a working pipeline based on the code provided in the lesson. After some data exploration to decide what parameters to try for feature extraction, I trained serval Linear SVM classifiers and choose the paramters that corresponded to the highest test accuracy on the test set. My choices for what size sliding windows to use were based on intuition, and fine tuning to try and reduce the number of windows while still getting good hot window results on test images. I will admit I ended up using a large number of windows and the performance of the approach was quite slow. Generating the project video took a considerable amount of time. More effort into reducing the number of sliding windows while still acheiving good results would go a long way to speeding up the entire pipeline.

To make pipeline more robust I'd like to try keeping track of heatmap detections in each frame and averaging them so the bounding boxes are less jittery. I'd also like to better optimize the number of sliding windows used.

My pipeline currently fails when 2 cars are very close to each other and the heatmaps for each car overlap creating one big bounding box.

