import numpy as np

class GaussianDiscriminant:

    def __init__(self,k=2,d=8,priors=None): # k is number of classes, d is number of features
        # k and d are needed to initialize mean and covariance matrices
        self.mean = np.zeros((k,d)) # mean
        self.S = np.zeros((k,d,d)) # class-dependent covariance
        if priors is not None:
            self.p = priors
        else:
            self.p = [1.0/k for i in range(k)] # assume equal priors if not given
        self.k = k
        self.d = d

    def fit(self,Xtrain,ytrain):
        # compute the mean for each class
        
        self.mean[0,:] = np.array(np.mean(Xtrain[np.where(ytrain == 1)],axis=0)) #same for both cases
        self.mean[1,:] =  np.array(np.mean(Xtrain[np.where(ytrain == 2)],axis=0))
        # compute the class-dependent covariance
        self.S[0,:,:] = np.cov(Xtrain[np.where(ytrain == 1)].T,ddof=0) #different for the other case 
        self.S[1,:,:] = np.cov(Xtrain[np.where(ytrain == 2)].T,ddof=0)
        
        
        pass


    def predict(self,Xtest):
        # predict function to get prediction for test set
        
        wi = np.zeros((self.k,self.d)) #2x8
        Wi = np.zeros((self.k,self.d,self.d)) #2x8x8
        wi0 = np.zeros(self.k) #2
        
        
        wi[0,:] = np.matmul(np.linalg.inv(self.S[0,:,:]),self.mean[0])
        wi[1,:] = np.matmul(np.linalg.inv(self.S[1,:,:]),self.mean[1])
        
        Wi[0,:,:]= -0.5*np.linalg.inv(self.S[0,:,:])
        Wi[1,:,:]= -0.5*np.linalg.inv(self.S[1,:,:])
        
        wi0[0] = -0.5*np.matmul(self.mean[0].T,np.matmul(np.linalg.inv(self.S[0,:,:]),self.mean[0])) - 0.5*np.log(np.linalg.det(self.S[0,:,:])) + np.log(self.p[0]) #P(C1) = 0.1
        wi0[1] = -0.5*np.matmul(self.mean[1].T,np.matmul(np.linalg.inv(self.S[1,:,:]),self.mean[1])) - 0.5*np.log(np.linalg.det(self.S[1,:,:])) + np.log(self.p[1]) #P(C2) = 0.9
        
        g = np.zeros(2)
        predicted_class = np.ones(Xtest.shape[0]) # placeholder  
        
        for i in np.arange(Xtest.shape[0]): # for each test set example
            for c in np.arange(self.k): # calculate discriminant function value for each class
                g[c] = np.dot(Xtest[i,:].T,np.dot(Wi[c,:,:],Xtest[i,:])) + np.dot(wi[c,:].T,Xtest[i,:]) + wi0[c]
                pass
            # determine the predicted class based on the discriminant function values
            if g[0] > g[1]:
                predicted_class[i] = 1
            else:
                predicted_class[i] = 2

        return predicted_class

    def params(self):
        return self.mean[0],self.mean[1],self.S[0,:,:],self.S[1,:,:]


class GaussianDiscriminant_Ind:

    def __init__(self,k=2,d=8,priors=None): # k is number of classes, d is number of features
        # k and d are needed to initialize mean and covariance matrices
        self.mean = np.zeros((k,d)) # mean
        self.S = np.zeros((d,d)) # class-independent covariance
        if priors is not None:
            self.p = priors
        else:
            self.p = [1.0/k for i in range(k)] # assume equal priors if not given
        self.k = k
        self.d = d

    def fit(self,Xtrain,ytrain):

        # compute the mean for each class
        self.mean[0] = np.mean(Xtrain[np.where(ytrain == 1)],axis=0) #same for both cases
        self.mean[1] = np.mean(Xtrain[np.where(ytrain == 2)],axis=0)
        # compute the class-independent covariance
        self.S = np.cov(Xtrain.T,ddof=0)
        pass

    def predict(self,Xtest):
        # predict function to get prediction for test set
        predicted_class = np.ones(Xtest.shape[0]) # placeholder
        wi = np.zeros((self.k,self.d)) #2x8
        wi0 = np.zeros(self.k) #2
        g = np.zeros(self.k)
        
        wi[0,:] = np.matmul(np.linalg.inv(self.S),self.mean[0])
        wi[1,:] = np.matmul(np.linalg.inv(self.S),self.mean[1])
        
        wi0[0] = -0.5*np.matmul(self.mean[0].T,np.matmul(np.linalg.inv(self.S),self.mean[0])) + np.log(self.p[0]) #P(C1) = 0.1
        wi0[1] = -0.5*np.matmul(self.mean[1].T,np.matmul(np.linalg.inv(self.S),self.mean[1])) + np.log(self.p[1]) #P(C2) = 0.9
        
        for i in np.arange(Xtest.shape[0]): # for each test set example
            for c in np.arange(self.k): # calculate discriminant function value for each class
                g[c] =   np.dot(wi[c,:].T,Xtest[i,:]) + wi0[c]
                pass

            # determine the predicted class based on the discriminant function values
            if g[0] > g[1]:
                predicted_class[i] = 1
            else:
                predicted_class[i] = 2
        return predicted_class

    def params(self):
        return self.mean[0], self.mean[1], self.S
