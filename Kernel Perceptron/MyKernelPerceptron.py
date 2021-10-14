import numpy as np

sigma_pool = [0.002,0.7,1.2] #sigma for kernel trick

class KernelPerceptron:
    def __init__(self,train_x, train_y, sigma_idx):
        self.sigma = sigma_pool[sigma_idx] # sigma value for RBF kernel
        self.train_x = train_x # kernel perceptron makes predictions based on training data
        self.train_y = train_y
        self.alpha = np.zeros([len(train_x),]).astype('float32') # parameters to be optimized

    def RBF_kernel(self,x):
        dist = np.linalg.norm(self.train_x - x, axis=1)      
        return dist
        

    def fit(self,train_x,train_y):
        max_iter = 1000
        
        # training the model
        for iter in range(max_iter):
            kernel = []
            #y = np.zeros(len(self.train_y))
            #y = []
            error_count = 0 # use a counter to record number of misclassification
            
            # loop through all samples and update the parameter accordingly
            for i in range(len(self.train_x)):
                dist = self.RBF_kernel(self.train_x[i])                
                K = np.exp(-(dist ** 2) / (2 * self.sigma))
                kernel.append(K)
            kernel = np.array(kernel)
            y = np.sign(kernel @ ( self.train_y * self.alpha ))
            # for j in range(len(y)):
            #     y[j] = np.sign(np.sum(self.alpha[j] * self.train_y[j] * kernel[j]))
                
            # stop training if parameters do not change any more
            for k in range(len(y)):
                if y[k] != self.train_y[k]:
                    #print(y[k] != self.train_y[k])
                    self.alpha[k] += 1
                    error_count += 1
                
            if error_count == 0:
                break
        
    def predict(self,test_x):
        # generate predictions for given data
        pred = np.ones([len(test_x)]).astype('float32') # placeholder
        kernel = []
        for i in range(len(test_x)):
                dist = self.RBF_kernel(test_x[i])
                K = np.exp(-(dist ** 2) / (2 * self.sigma))
                kernel.append(K)
        
        kernel = np.array(kernel)
        
        pred = np.sign(kernel @ (self.train_y * self.alpha))
        # for j in range(len(test_x)):
        #     pred[j] = np.sign(np.sum(self.alpha[j] * self.train_y[j] * kernel[j]))
        #     if pred[j] 
        
        
        
        return pred

    def param(self,):
        return self.alpha
