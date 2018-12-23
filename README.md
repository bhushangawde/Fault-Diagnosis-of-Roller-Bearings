# Fault-Diagnosis-of-Roller-Bearings

# Data

The Data folder contains the dataset which was used for the project. The original three files named are:
1. original data at 500rpm.csv
2. original data at 900rpm.csv
3. original data at 1300rpm.csv

It also contains denoised data for the file "original data at 1300rpm.csv". Savitzky-Golay filter was used for denoising the data.


# Code

This folder contains all the code files written for the project. Python programming language was used for written code.


# Plots

This folder contains the output plots for getting a better understanding of the methodologies used in the project.


# Methodology

We tried three ways of denoising the dataset. At first, we denoised the dataset using Running Mean Filter. But, it caused the datapoints to be shifted by some patch size. Then, we tried Wavelet based soft thresholding which gave a decent level of denoising. Lastly we used Savitzky Golay Filter for denoising the data which gave the best results (can be observed from the plots).

Then we divided the dataset into classes with size of 1024 values. Then we extracted the features like mean, kutosis, skewness, crest, clearance, standard deviation, shape factor etc. 

After extracting features, we selected the essential features using Principal Components Analysis.

Later, we identified the type of fault in roller bearings using k-means and Dendrogram Clustering with an accuracy of 80%.

