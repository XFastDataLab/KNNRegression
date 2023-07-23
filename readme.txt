kNNRegression

Introduce
      Differential privacy is an effective way for private data protection, but makes it quite difficult for the task of classifying private
data, due to that it destroys the authenticity of the data. And the situation is even worse in distributed context. In this paper, we propose
a schema named DistributedPrivateKNN for classifying private data in distributed context, which includes training model and testing
model. In training model, kNN Regression (kNNReg) is invented for training private data, which is able to roughly restore the
neighborhood structure of raw private data after adding Laplacian noise. In testing model, a heuristic usage named ‘PP+LP’ is
designed for data owners to classify local unlabeled data, by combining local information with the public regressed noisy data obtained
by kNNReg, without leaking data privacy. The experimental results show that our algorithm is encouraging, not only it protects the data
privacy, but also has satisfactory performance in four aspects: the noisy data scale, neighborhood structure, noise directions and
classification accuracy.

Environment
    Internet Core i7-8700 CPU @3.20 GHz, 16G RAM. The Code is written in Matlab and run on window 10 (64).

Branch introduce
    1. test : Run code
    2. trainPartialDP :  Model training code
    3. genLaplasNoise : Generate Laplace noise
    4. addLaplacianNoise : Add Laplace noise
    5. drawshapes : Drawing code
    6. findKNN : Finding K-neighbors
    7. get_error_label :  Get error data information
    8. getAcc : Calculate similarity, i.e., accuracy
    9. knnClassify : The poll categorizes test_data by kNN and compares it against the given answer label.

Guideline
    1. Data sets Divide test data sets and training data sets.
    2. The parameter information of each data set can be modified in the test code.
    3. By running test, covariance, nearest neighbor similarity, accuracy of each node and overall accuracy can be obtained. 
       2D data can display original data picture, Laplacian noise picture and data picture after regression.
   