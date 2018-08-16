# Markov-Chain-Transition-operator
Code for the paper 'Spectral thresholding for the estimation of Markov chain transition operators'

The file estimatorp contains all necessary files to compute the estimator \tilde{p} for a given data-vector in one dimension. To do this run the program estimatoroph(d,J,l) on your file. This computes the estimator for the action of the transition operator P on the approximation space V_J. 'd' should be a data-vector containing the values of the observed, real valued chain (X_0, X_1, ..., X_n). J specifies the number of trigonometric basis functions which should be used in the computation of the estimator. Finally, l is the hard-threshold level at which the eigenvalues of the estimated matrix should be thresholded.
Having obtained the output of estimatoroph, a (J+1)\times (J+1)-matrix you can then run the program estimatortd(p,x,y) to obtain the corresponding estimate for p(x,y). 

The file diffusiongen contains the necessary files to simulate discrete samples from a Cox-Ingersoll-Ross process and from a Radial Ornstein-Uhlenbeck process. These are then saved in a .txt file, "Diffusiondata.txt". 
