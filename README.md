# Kernel_based_QbESTD
This is the implementation of Kernel-based Query by Eample-Spoken Term Detection using CNN.

Refer to the following link for the TIMIT dataset.

https://catalog.ldc.upenn.edu/LDC93S1

Steps to use the code
1) Use git_Extract39DMFCC.py to obtain the 39-dimensional MFCC features from the audio files. Please provide an appropriate path to the audio files. It will compute the 39-dimensional MFCC for each audio file and save it as filename.mfcc.
2) To build a Gaussian Mixture Model, Use the buildAndSaveGMM function in git_ExtractGaussianPosteriorgrams.py with the number of GMM components and the link to mfcc files as arguments. It will create a GMM model using the reference audio files. Save the GMM model to calculate Gaussian posteriorgrams.
3) Use CalculatePosteriorgrams in git_ExtractGaussianPosteriorgrams.py function to calculate the Gaussian posteriorgram.
4) Use git_serialMatchingMatrixgeneration.py or git_parallelMatchingMatrixComputation.py to compute the matching matrix between the query and the reference utterance posterior grams and save it as an image.
5) After generating the images, use the git_Cnn_Training.py to train the CNN model.
   
