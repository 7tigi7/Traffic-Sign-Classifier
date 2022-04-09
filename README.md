# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs data set.
The dataset contains 32(Weight) x 32(Height) x 3(RGB) images.

- The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32*32*3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
[Data set]: ./examples/datasetGenerated.jpg "Data set"

These bar charts showing the distributions of training validation and test datas.
[Distributions]: ./examples/distrib.jpg "Distributions"

As you can see the dataset is not balanced, some classes has a few around 250 some classes has more than 1500 or 2000 data.
It is important to pay attention to this, because this way our estimate can be tilted in one direction.

### Design and Test a Model Architecture

#### Image Pre Processing

As a first step, I decided to convert the images to grayscale because the neural network works easier on grayscaled images.
When a computer sees an image, it will see a matrix of pixel values. In this case it will see a (32 x 32 x 3)3 dimension of numbers. The last dimension is the color and these numbers have a value from 0 to 255. These dimensions only make the calculation difficult, so we simplify the picture.
As a result we get (32 x 32 x 1) dimension pictures.

Here is an example of a traffic sign image before and after grayscaling.

Before
[Original]: ./examples/30original.jpg "Original"

After
[Grayscaled]: ./examples/30gray.jpg "Grayscaled"

And as second step I normalized the image data because minimally the image data should be normalized so that the data has mean zero and equal variance. For image data, (pixel - 128)/ 128 is a quick way to approximately normalize the data and can be used in this project.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscaled image 						| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| RELU					|												|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling        	| 2x2 stride, valid padding, outputs 5x5x16     |
| Flatten				| outputs 400  									|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Fully connected		| outputs 84  									|
| RELU					|												|
| Fully connected		| outputs 43  									|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

My model is based on a LeNet neural network
[Lenet](https://engmrk.com/lenet-5-a-classic-cnn-architecture)

To train the model, I used dropout to make it more effective.

read more about dropout
[Dropout](https://tf-lenet.readthedocs.io/en/latest/tutorial/dropout_layer.html)

The input for LeNet-5 is a 32×32 grayscale image which passes through the first convolutional layer with 6 feature maps or filters having size 5×5 and a stride of one. So after pre processing the image dimensions changing from 32x32x1 to 28x28x6 and so on as described above.
For dropout i tried so many numbers to set the keep_prob variable. My trying range was 0.1 to 0.9 and I found 0.4 the one of the best values to set.

There are a few more variables that should not be missed.

1. rate
Rate means the learning rate. If this number is small the learning is slow, so if you set this number very little you should raise the number of epochs or raise the number of training data to make the accuracy higher.
My choice was 0.0007

2. EPOCHS
EPOCHS is the number of epochs.That is a hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset. One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters,weights. An epoch is comprised of one or more batches.
My choice was 50

3. BATCH_SIZE
This variable is a number of  inputs we are using in one pass(forward and backward) or in one weight updation of model
My choice was 128

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.940
* validation set accuracy of 0.961 
* test set accuracy of 0.994

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I chose the LeNet model because I already knew this model from the number recognition example given by udacity.
LeNet is a convolutional neural network structure proposed by Yann LeCun et al. in 1989. In general, LeNet refers to lenet-5 and is a simple convolutional neural network. Convolutional neural networks are a kind of feed-forward neural network whose artificial neurons can respond to a part of the surrounding cells in the coverage range and perform well in large-scale image processing.
Read more
[Lenet](https://en.wikipedia.org/wiki/LeNet)

* What were some problems with the initial architecture?

Some parameters has to be changed, the dimensions, the output has to be one of 43 classes.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting.

I only changed that i set a dropout on activation 3 and 4. It was enough to increase the accuracy well.

* What are some of the important design choices and why were they chosen?
For example, why might a convolution layer work well with this problem?
How might a dropout layer help with creating a successful model?

CNNs are used for image classification and recognition because of its high accuracy. ... The CNN follows a hierarchical model which works on building a network, like a funnel, and finally gives out a fully-connected layer where all the neurons are connected to each other and the output is processed.

Why it is beneficial to use pre trained models?
Simply put, a pre-trained model is a model created by some one else to solve a similar problem. Instead of building a model from scratch to solve a similar problem, you use the model trained on other problem as a starting point. For example, if you want to build a self learning car.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

[5 German Traffic Signs]: ./examples/fivegerman.jpg "5 German Traffic Signs"

I resized all of my image to 32x32 pixels and i put all the signs in all image to the center.

These images are not very hard to identify bacause of the good quallity, there is no "smog" on the images and all of them made at good light conditions. Of course we can challange the network if we want but here that was not in the scope.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No entry      		| No entry   									| 
| Priority road     	| Priority road 								|
| Keep left				| Keep left										|
| Stop	      		    | Stop		   			 	             		|
| Speed limit (30km/h)	| Speed limit (30km/h)							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

[Top 5 softmax]: ./examples/top5softmax.jpg "Top 5 softmax"


| Probability           |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0      		        | No entry   									| 
| 1.0     	            | Priority road 								|
| 0.84				    | Keep left										|
| 1.0	      		    | Stop		   			 	             		|
| 1.0	                | Speed limit (30km/h)							|


For the third image it is only 0.84. It might be the sign size and the thick black edges causes it.

### Discussion

Here some links related to this image classifier project
[Classifiers](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) Classifiers
[Keras](https://keras.io/) Keras

Try out this project, modify the values that i mentioned above and lets see where it goes for you!
Thanks for reading!
