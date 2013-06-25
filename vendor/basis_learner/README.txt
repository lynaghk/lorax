A README file for the code implementing the Basis Learner algorithm.

This code is given under the BSD 2-clause license (see License.txt).
Copyright (c) 2013 Ohad Shamir. All Rights Reserved.

The code is a Matlab implementation of the Basis Learner algorithm, as described in the associated paper. The code was developed on Matlab R2010a, and tested on some later versions as well. It should also work on GNU Octave, although this was not tested extensively. The code is intended for research purposes - use at your own risk.



Example Usage
=============

After extracting all files into the current directory, type the following in a Matlab console. The test classification error using all layers and best lambda should be 0.056 (or 5.6% error).

load('data');      % Get X,Y data matrices, consisting of 500 examples.
widths = [10 10];  % construct network with 2 layers of width 10, plus an output layer.
trainend = 250;    % Use first 250 examples in X,Y as a training set.
lambdaRange = 10.^(-3:1);
F = BuildNetwork(X,Y,widths,trainend);
Results = BuildOutputLayers(F,Y,trainend,widths,lambdaRange);



Code Description
================

The top two M-files are BuildNetwork.m, which builds the network given data, and BuildOutputLayers.m, which trains an output layer. Note that the current implementation focuses on binary and multiclass classification only. We now describe the usage of each of these Matlab functions.

F = BuildNetwork(X,Y,widths,trainend)
-------------------------------------

- X is an m*d data matrix, each row specifying a d-dimensional instance.

- Y is the associated m*1 label Y: either binary (taking one of two distinct values) or multiclass (i.e. of the form {0,1,2,..} or {1,2,3,...}).

- trainend specifies the training set: X(1:trainend,:) and Y(1:trainend) are assumed to be the training set, which the algorithm uses to build the network. X(trainend+1:end,:) and Y(trainend+1:end) are considered test set instances.

- The vector 'widths' specifies the required width of each level in the network, except the output layer. Thus, widths(i) specifies the level of the i-th layer, and length(widths) is the depth of the network (again, except the output layer).

- The function returns a matrix F, so that F(i,:) specifies the output of each node in our network when fed with the instance X(i,:). For i>trainend, F(i,:) represent the output of the nodes on a test set instance, which was not used in constructing the network. 

The function also has some internal parameters which can be modified (choice of how the first layer is built, batch size and tolerance parameter).


Results = BuildOutputLayers(F,Y,trainend,widths,lambdaRange)
------------------------------------------------------------

This function takes as input the matrix F computed by BuildNetwork, as well as Y,trainend,widths as before, and uses the training set component to build the network's output layer. For each i=1:length(widths) and for each j=1:length(lambdaRange), it trains a classifier using layers 1,2,..,i and regularization parameter lambdaRange(j), using hinge-loss or multiclass hinge-loss, and reports the train and test errors in Results(i,j,1) and Results(i,j,2) respectively. The results are also reported textually on standard output. 

Description of Auxiliary Files:
-------------------------------
- SGD_hinge.m: Implements stochastic gradient descent to optimize regularized hinge-loss with respect to data. Internal parameter num_epochs governs how many passes over the data to perform.
- MC_SGD.m: Implements stochastic gradient descent to optimize regularized multiclass hinge-loss with respect to data. Internal parameter num_epochs governs how many passes over the data to perform.
- orthonormalize.m: Computes the first layer of the network, using exact SVD. 
- orthonormalize_approx.m: Computes the first layer of the network, using approximate SVD. This function is not invoked unless the inernal BuildMethodFirstLayer parameter in BuildNetwork.m is set to 'approx'. 
- data.mat: small example dataset (see usage above).