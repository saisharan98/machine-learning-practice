import numpy as np

#MLP with 2-3 hidden layers from scratch
def process_data(data,mean=None,std=None):
    # normalize the data to have zero mean and unit variance (add 1e-15 to std to avoid divide-by-zero)
    if mean is not None:
        # mean and std is precomputed with the training data
        data = (data - mean)/(std + 1e-15)
        data = np.nan_to_num(data)
        return data
    else:
        # compute the mean and std based on the training data
        mean , std = np.mean(data,axis=0), np.std(data, axis=0) # placeholder
        data = (data - mean)/(std + 1e-15)
        data = np.nan_to_num(data)
        
        return data, mean, std

def process_label(label):
    # convert the labels into one-hot vector for training
    one_hot = np.zeros([len(label),10])
    rows, cols = one_hot.shape[0], one_hot.shape[1]
    for i in range(rows):
        for j in range(cols):
            one_hot[i,label[i]] = 1
    return one_hot

def sigmoid(x):
    # implement the sigmoid activation function for hidden layer
    f_x = 1.0/(1.0 + np.exp(-x)) # placeholder  
    return f_x

def softmax(x):
    # implement the softmax activation function for output layer
    f_x = np.exp(x)/np.sum(np.exp(x)) # placeholder
    return f_x

class MLP:
    def __init__(self,num_hid):
        # initialize the weights
        self.weight_1 = np.random.random([64,num_hid]) #w 64 x num_hid
        self.bias_1 = np.random.random([1,num_hid]) #w0 1 x num_hid
        self.weight_2 = np.random.random([num_hid,10]) #v num_hid x 10 
        self.bias_2 = np.random.random([1,10]) #v0 1 x 10
        

    def fit(self,train_x,train_y, valid_x, valid_y): #train_x shape: 1000x64
        
        lr = 5e-3     # learning rate
        count = 0     # counter for recording the number of epochs without improvement
        best_valid_acc = 0
    
        epochs = 100
        
        while count<=epochs:
            # training with all samples (full-batch gradient descents)
            # implementing the forward pass
        
            z_h = self.get_hidden(train_x) #1000 x num_hid     
            predicted_y = self.predict(train_x) # y probabilites   
            predicted_label = process_label(predicted_y) #labels
            
            # implementing the backward pass (backpropagation)
            # computing the gradients w.r.t. different parameters    
            
            diff_label = train_y - predicted_label  #1000 x 10 (r^t - y^t)    
            
            ##----update for v_h-----##
            weight2_update = lr * np.dot(z_h.T,diff_label) # num_hid x 10
            bias2_update = lr * np.sum(diff_label,axis=0)
            
            ##----update for w_h----##
            inner_product = (np.dot(diff_label, self.weight_2.T))  #1000 x 10 x (num_hid x 10).T = 1000 x num_hid.
            dz = (z_h * (1 - z_h)) #z_h: 1000 x num_hid 
            coeffs = (inner_product * dz) # 1000 x num_hid
            weight1_update = lr * np.dot(train_x.T,coeffs) #64 x num_hid
            bias1_update = lr * np.sum(coeffs, axis=0) 

            #update the parameters based on sum of gradients for all training samples
            self.weight_1 += weight1_update
            self.weight_2 += weight2_update
            
            self.bias_1 += bias1_update
            self.bias_2 += bias2_update

            # evaluate on validation data
            predictions = self.predict(valid_x)
            valid_acc = np.count_nonzero(predictions.reshape(-1)==valid_y.reshape(-1))/len(valid_x)
            
            # compare the current validation accuracy with the best one
            if valid_acc>best_valid_acc:
                best_valid_acc = valid_acc
                count = 0
            else:
                count += 1
        return best_valid_acc, predicted_label
    

    def predict(self,x):
        # generate the predicted probability of different classes
        z_h =  self.get_hidden(x)
        # converting class probability to predicted labels              
        y = softmax(np.dot(z_h,self.weight_2) + self.bias_2)    #1000 x 10                             
        y_max = np.argmax(y, axis=1)
                                     
        return y_max

    def get_hidden(self,x):
        # extract the intermediate features computed at the hidden layers (after applying activation function)    
        z = sigmoid(np.dot(x, self.weight_1) + self.bias_1)
        
        return z

    def params(self):
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2
