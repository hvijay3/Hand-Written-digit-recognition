# Hand-Written-digit-recognition
Implemented neural network for hand written digit recognition
Multilayer  Perceptron – MNIST  Dataset 
Harshit Vijayvargia 
Department of Computer & Information Science, University of Florida
Gainesville, USA 
hvijayvargia@ufl.edu 


 
Abstract—The purpose of this report is to explain the implementation of a back propagation neural network to solve Handwritten digit recognition problem. MNIST data set which is a standard database of 10 handwritten digits is used for training the neural network. Throughout my implementation, I have experimented several methods for optimizing the performance of my neural network such as down sampling, dimensionality reduction, generalization, techniques, enhanced gradient descent methods and cross-validation .  I have empirically justified by analyzing classification results on validation and test data set that these techniques have improved the classification performance both in terms of accuracy and efficiency.

Introduction 
If an image of some animal in distorted form is presented to us then we can tell what is it, but if the same image is processed through an algorithm which has not processed it before then it is highly likely that the algorithm may fail. The reason behind this are the 80 billion neurons in our brain which work together in a complicated manner to process information when we see something. Machines on the other hand are restricted in such capabilities. This system of neurons is the inspiration behind creation of Artificial neural network which train themselves according to data.

Our focus in this report is on pattern recognition especially handwritten digits recognition which has a great practical value in real world. But why we need a Machine learning algorithm for recognizing hand written digits? 

 Machine Learning for Hand Written digits: 
If the digits in the MNIST dataset were not hand written or they followed a specific font, then it would have been much easier to find a matching template which can recognize digits efficiently. But this is not the case in hand written digit problem. For example, consider some sample images of’ ‘2’ taken from MNIST data set (Figure 1). These images don’t have many features in common as if we try to overlay one on another or try to skew them, they will not fit. This makes this problem a challenging one and demands for a machine learning algorithm.
 
Figure 1 – Sample images from MNIST dataset

In this report, we will implement neural network backpropagation algorithm for solving this problem. But our focus will be on improving the algorithm’s performance by using various neural networks improvement techniques.

      Ⅱ. NEURAL NETWORKS

Neural networks are adaptable systems which try to learn patterns in data and then try to mimic these patterns when presented with unseen data. These networks are made of interconnected processing units which receive input from external sources or connected units and evaluate an output which is then propagated to next unit. The decisions are based on discriminant functions or final output values of these networks.

In figure a representation of 3 Layer neural network is shown for hand written digit recognition problem. For our problem each image is represented by 28*28 matrix so total inputs fed into the network will be 784. Since each Image can be classified into one of the digits ( 0 -9 ) so there are 10 outputs. The number of neurons in hidden layer( K) and total number of hidden layers can change.

 
                                     Figure 2
a1,a2…..ak are activation neurons.
Back Propagation algorithm: 
[2]Back Propagation algorithm works in 3 steps. I have presented the algorithm for three layer case, subscript i represents input layer, j hidden layer and k output layer. wji are parameters from input to hidden layer, wkj are parameters from hidden to output layer. netj denotes the net input at layer j. bj, bk are the bias terms added.
Forward phase : Here we feed input xi into the network and compute  netj .
 
For output to hidden layer yk is evaluated using equation:
                 
Step2: Backward phase: In backward phase we propagate the error δ in backward direction for all layers except first layer using equations:
 
Step3: Updating weights : This is the last step where we update weights. The magnitude with which we update them depends on learning rate α. The equations are given as:
 
Ⅲ. Setting hyper-parameters

Before we begin setting our parameters we need to clean data so noise in data doesn’t affects our performance. MNIST data set makes our task easier. The images in this data set are already normalized in size and centered which allow us to focus only on machine learning.

 Training, Validation and Testing set: 

MNIST dataset has two sets of data. First data set 
contains 60000 images of which we will use 50000 for training and 10000 for validation. The second part will be used to test our classification results. We will select our best hyper-parameters by performing experiments on validation set, not on test set. The reason behind this is: if we try to choose our parameters based on testing set then we are overfitting our hyper-parameters parameters to testing set.

Network Structure: 

Determining Network topology is the important part of training neural nets as it governs the decision boundaries. There is no set rule which can give perfect results. It depends on the data we are dealing with (Linearly or Non- Linearly separable). However, we can use a rule of thumb to decide number of hidden layers and units in our network and then use cross validation and convergence rate as a parameter to test our heuristic selection. While doing cross validation, we must also keep an eye on overfitting and MSE of training set.

As per rule of thumb:
Choose number of hidden units such that total number of weights in the network are n/10 where n is the number of training points. 
When it comes to choosing number of Hidden layers then at most 2 hidden layer will suffice. Considering more layers will increase time complexity. We will start with one hidden layer and will proceed further according to classification performance.
Weights: 

Initial values of weights also govern the accuracy and convergence rate of our training algorithm. Our objective is that weights get updated uniformly. We don’t want one set of parameters to get trained before another. So, we will initialize weights in a certain range given as: 
  
-1÷√d   <  Wij  < 1÷√d

If Wij maps from input to hidden layer then d is the number of features or inputs to a hidden unit.

If Wij maps from hidden to output layer then d is the number of hidden units

Transfer Function: 

A transfer function governs the learning of algorithm as it’s derivative plays major part in back propagation algorithm. We have two choices: either to use sigmoid function or ReLU function. It is observed and we will also justify empirically that ReLU function makes the learning faster. Sigmoid function and ReLU functions are given as:
Sigmoid:
                     
Where t is the input to neuron.

 ReLU:      
                
Where x is the input to neuron.

The advantage of using RELU is that it decreases the probability of vanishing gradient. If we are using cross entropy as our cost function then there is no problem of vanishing gradient descent but for MSE we have to take care.  When x > 0 then gradient has a constant value whereas in sigmoid function gradient value gets increasingly smaller when absolute value of inputs is increased i.e
            G′(x)=G(x)(1−G(x))
G′(x) approaches zero when a is infinitely large. Whereas, the gradient value is constant in ReLU, which makes training faster.

The other benefit of using RELU is the sparsity it brings in representation when a≤0. Sigmoid on the other hand bring dense representations and it is observed that sparse representations are better than dense representation.

Cost Function:

 For choosing cost functions I have two options: Cross entropy or Mean square error. There equations are:

Cross entropy:
 
Where y=y1,y2 represents desired values at the output neurons, aL1,aL2,… are the actual output values. n is number of training data samples. The summation is done over all x inputs.

Mean Square Error(MSE) (Ѳ – Parameters, m -training samples, x- input, y - output): 
 
Cross entropy works well if we are using sigmoid function as an activator. Because it prevents vanishing gradient descent problem caused due to sigmoid. But since here we are using ReLU for its benefit over sigmoid, we will prefer using MSE as cost function.

Learning Rate: 

We will keep µ=0.1 initially and will increase and decrease it in case of slow convergence or divergence.

Gradient Descent: 

If data has redundant samples then it is better to use stochastic gradient descent in which we choose random samples from training data and update parameters after processing each sample. Also, It is faster compared to batch gradient descent in which we take a batch of training data and update parameters once whole batch is processed. In our case since training data comprises of 50000 samples so stochastic gradient descent will be preferred. We will justify this empirically by comparing performance of both batch and stochastic gradient descent in our problem. We will also implement a special case of SGD known as SGD momentum which helps to train our algorithm faster.

Epochs:

 If we are using batch gradient descent, then increasing the number of epochs will increase our computational time. Our objective should be to determine optimal epochs which gives faster convergence and better cross validation accuracy. Optimal epoch depends on batch size, learning rate and other hyper-parameters. Besides that, We will also implement early stopping in our algorithm to determine optimal epochs.

Ⅳ. TRAINING NEURAL NETWORK

I started training my algorithm after setting the hyper parameters as mentioned above with a batch size of 50000, but it was taking too long to train the algorithm. To solve this problem, I implemented the following three techniques and analyzed each one of them:

1. Reducing batch size(n) from 50000 to 5000. To implement it, we must also decrease number of parameters or hidden elements as we can have at most n(batch size) parameters . If we keep parameters more than n then it may cause overfitting problem.

But, since we are taking all 784 features of images then we can have at most 6-7 hidden elements in hidden layer. This reduction in hidden elements is not feasible as after implementing it, my recognition accuracy decreased to 30.22%.
This option is feasible only if we can reduce the batch size while maintaining appropriate number of hidden elements. The other two alternatives can solve this problem.

2. Down sampling: The image given is of size 28*28 . To reduce the number of features, I down sampled the image using max pooling.  For example for transforming a 28*28 image to 14*14. I divided the image into 2*2 submatrices and among each  matrix took the pixel with maximum intensity out of 4. I could have also used Gaussian blur to perform this operation but it would have increased computational time. After doing down sampling and feeding images into training algorithm, I obtained the following results with a batch size of 10000, hidden elements 80 and step size 0.1 and 1000 epochs :

  
                   Table 1
This is saving time but not giving good accuracy 

3. PCA (Principal Component Analysis):

PCA can reduce the dimensionality of images and provide us with eigen vectors along the direction of maximum variance. 

I applied PCA and obtained the eigen vectors. Their contribution to variance is shown in figure. 

 
                                  Figure 5



To determine the value of K or the size of input which I should choose to train my algorithm, I will consider different values of K at which variance is greater than 85 %.  At each value of K, I will test my algorithm’s performance with same set of parameters (Hidden elements = 10 , step size = 0.1 , Epochs = 700 ) both in terms of accuracy and computational time. The results I obtained are:



Principal Components	Cross validation Accuracy	Computation time (Seconds)
65	90.15	227.8065982
75	89.58	436.7237671
85	90.23	657.9708143
95	90.47	874.6496286
150	90.12	1106.175345
200	90.66	1358.916753
350	90.07	1661.201786
400	90.69	1969.04026

I will take k = 65 as I see a good trade-off between accuracy and computational time at this value.
 
With and without PCA: 

PCA has reduced the computational time while keeping the accuracy same as when we were using original image features. This can be seen from the following results(table 2) obtained at same hyper -parameters set before: 

 	Time (Seconds)	Accuracy (%)
Without PCA	1895.806597	90.3
With PCA	227.8066	90.15
                                     Table 2

Experimenting with Hyperparameters:

Dimensionality reduction has brought reduction in computational time but didn’t improved the accuracy. To improve cross validation accuracy, I will experiment now with other parameters keeping K fixed.

1.  Network Structure: 

Case 1: Single Hidden Layer: 

Optimal Number of Hidden Elements: 

After doing dimensionality reduction we have reduced the size of input from 784 to 65 which will allow us to increase hidden elements. To improve accuracy further we can increase the number of hidden element. The cross-validation accuracy percentage I obtained on increasing hidden elements is shown in the figure  below:

 
                                         Figure 6
Observation : As we can see from figure 6 the accuracy is increasing initially with increase in neurons . But after a certain stage it becomes static. As we are increasing hidden elements we are increasing our parameters which in turn is causing problem of overfitting due to which classification performance is not improving. I choose 80 hidden elements as I found a good tradeoff between accuracy(94.06%) and computational time (350 seconds) at this number.

Case 2 : Two Hidden Layers 

To check for improvement in computational time and accuracy I experimented with 2 hidden layers. Keeping all parameters fixed I compared the results with one hidden layer structure and found network structure with one layer is more effective in saving computational time. For this reason I didn’t proceeded with 2 layers network.

Generalization improvement techniques

A good performance on training data doesn’t mean that the algorithm will perform good on testing data. The reason behind this is overfitting. To prevent this, we can use some techniques:

Regularization:  
Increasing the size of input is one alternative to avoid overfitting as it is harder for a network to adapt to larger input size but this will be computationally expensive. So instead we will try a widely used technique known as regularization. We will add an extra term to cost function  

         λ is known as regularization parameter which acts as a compromise between preferring smaller weights or minimizing cost function.

After bringing regularization the results obtained are:
                   
                                   Table 2
Accuracy increased slightly by 1%.

Early Stopping : In order to avoid overfitting we need to stop training when validation error becomes minimum as this is the point till which generalization is good. 
In figure 7, I have located the iteration where validation error becomes static(200 epoch as epochs are scaled by 4). After implementing early stopping in my training algorithm I was able to increase accuracy from 94.06% to 95.13% and computation time from 350 to 212 seconds. This is a good improvement.


 
                                          Figure 7
b)  Drop Out: When number of parameters in a neural network are high then overfitting problem may occur. To overcome this problem, we can use drop out. It is a generalization method in which we randomly drop units and all connections with these units (weights) with certain probability p.  
During test time we will keep all activation neurons, however we will scale the parameters by a probability value p.
I implemented drop out in each layer. The results which I obtained when I set the probability of dropping units 0.95 are :

 
                           Table 3

The results are not as we expected. There is not much improvement in classification performance. The reason behind this is that our network structure is not large. Even the parameters we are using are not too many. Dropout works very well for highly complicated network topologies where there are more parameters and more then one hidden layer.

After implementing generalization techniques we have come to a point where we have determined our network structure and hyper-parameters.

Further Improvements: 

Gradient Descent: 

Till now we were using mini batch gradient descent. As discussed before stochastic gradient descent(SGD) may give better performance for our problem. To empirically verify whether SGD will improve the performance in our case, I implemented it and obtained the I took one sample each time at random and updated parameters after processing each sample (batch size =1). I was able to get accuracy at 50000 epochs ( one hidden layer , 80 hidden elements , 65 principal components , 0.1 step size )

The best performance was obtained at 24.03 seconds but we can do better . Instead of taking one sample at a time I took 50 samples randomly at a time and in 50000 epochs I was able to get accuracy of 97.3% on validation set but the time increased to 57.93 seconds . To decrease this time I will use SGD momentum where we can increase the convergence rate. The comparision of both SGD and SGD momentum can be seen in the following figure. 

Since choosing SGD momentum as our gradient descent method is giving best results on validation data, I will consider it instead of mini -batch gradient descent to perform classification on testing data.

 
                                      Figure 8 
Ⅳ. RESULTS
After selecting best hyper-parameters for neural network and implementing all generalization techniques. I test my algorithm on testing data and obtained the following results(figure 9): In figure epochs are scaled by 20000.

 
                                               figure 9
The recognition performance has a classification accuracy of 97.52% at 40000 epochs and computational time is 29.37 seconds.

Confusion Matrix : 

 
                                                 TABLE 4

Does our results Make sense: 

From Confusion matrix and table  it can be seen that results make sense. If we see which digits the classifier has incorrectly classified, then these are the ones which can be misinterpreted with others like 9 with 4, 2 with 7. 

 
                                                Table 5 
Ⅴ. CONCLUSION

In this Project, We presented neural network backpropagation algorithm to classify handwritten digits efficiently. Our main objective was to obtain optimal performance in terms of accuracy and computational time. We started with dimensionality reduction through PCA and down sampling and then proceeded with training network and selecting best hyper-parameters. After this we implemented various generalization techniques to improve our algorithm’s performance on validation and testing data set. In the end I was able to achieve an accuracy of 97.52%.


REFERENCES 

[1] Z. Dan, and C. Xu, “The Recognition of Handwritten Digit Based on BP Neural Network and the Impleementation on Android,” Third International Conference on Intelligent System Design and Engineering Applications, pp. 1498-1501
[2]. Karlik and A. Olgac, "Performance Analysis of Various Activation Functions in Generalized MLP Architectures of Neural Networks," International Journal of Artificial Intelligence And Expert Systems (IJAE), Volume (1): Issue (4).

[3] Vineet Singh, and Sunil Pranit Lal, “Recognizing Handwritten Digits and Characters”

[4] F. Lauer, C. Suen, and G. Bloch, “A trainable feature extractor for handwritten digit recognition,” Pattern Recognition, vol. 40, no. 6, pp.1816–1824, 2007.

[5] Christopher M. Bishop. Neural Networks for Pattern Recognition. Clarendon Press,
Oxford, 1995.
