import numpy as np


def PCA(X,num_dim=None):
    X_pca = X # placeholder 3000x784
    
    # finding the projection matrix that maximize the variance (Hint: for eigen computation, use numpy.eigh instead of numpy.eig)
    X_mean = np.mean(X,axis=0) #1x784
    X_shifted = X - X_mean #3000x784
    cov = np.cov(X_shifted.T) #covariance matrix for the data #784x784
    
    
    eigenobject = (np.linalg.eigh(cov)) #an object that contains the eigenvectors and eigenvalues
    eigenvectors = np.flip(eigenobject[1].T,axis=0)  #784 x 784
    eigenvalues = np.flip(eigenobject[0],axis=0)   #784 x 1   
    
    
    sum_eig = np.sum(eigenvalues) #sum of all eigenvalues
    
    # select the reduced dimensions that keep >90% of the variance
    if num_dim is None:
        sum_sofar = 0 #sum of k eigenvalues k<n
        frac = 0.0
        for i,eig in enumerate(eigenvalues):
            if frac >= 0.9:
                break
        
            sum_sofar+=eig
            frac = sum_sofar/sum_eig    
        
        X_pca = eigenvectors[0:i,:]@X_shifted.T
        num_dim=i
        
    else:
        X_pca = eigenvectors[0:num_dim,:]@X_shifted.T
        
        
    # project the high-dimensional data to low-dimensional one'''
    

    return X_pca.T, num_dim
