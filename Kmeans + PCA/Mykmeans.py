import numpy as np


class Kmeans:
    def __init__(self,k=6): # k is number of clusters
        self.num_cluster = k
        self.center = None
        self.error_history = []

    def run_kmeans(self, X, y):
        # initialize the centers of clutsers as a set of pre-selected samples
        init_idx = [1, 200, 1000, 1001, 1500, 2000] #intial indices
        self.center = X[init_idx] #Initial centers 6x784
        num_iter = 0 # number of iterations for convergence
        '''EACH CENTER WILL BE  A 1x784 or 1xnum_dim VECTOR'''
       
        # initialize cluster assignment
        prev_cluster_assignment = np.zeros([len(X),]).astype('int')
        cluster_assignment = np.zeros([len(X),]).astype('int') 
        is_converged = False
        
        dist = np.zeros((len(X),len(self.center)))
        
        # iteratively update the centers of clusters till convergence
        while not is_converged:
           

            # iterate through the samples and compute their cluster assignment (E step)
            for i in range(len(X)):         
                for j in range(len(self.center)):
                # use euclidean distance to measure the distance between sample and cluster centers
                    dist[i,j] = np.linalg.norm(X[i] - self.center[j]) #minimum distance to the center
                    #print("dist,i,j = ",dist[i,j],i,j)
                    
                

            # determine the cluster assignment by selecting the cluster whose center is closest to the sample
            
            cluster_assignment = np.argmin(dist,axis=1)
            
            
            # update the centers based on cluster assignment (M step)
            
            for i in range(len(self.center)):
                self.center[i] = np.mean(X[np.where(cluster_assignment == i)],axis=0)
            
            
            

            # computing the reconstruction error for the current iteration
            cur_error = self.compute_error(X, cluster_assignment)
            self.error_history.append(cur_error)

            # reach convergence if the assignment does not change anymore
            is_converged = True if (cluster_assignment==prev_cluster_assignment).sum() == len(X) else False
            prev_cluster_assignment = np.copy(cluster_assignment)
            num_iter += 1

        # construct the contingency matrix
        
        contingency_matrix = np.zeros([self.num_cluster,3])
        for i in range(len(X)):
            if y[i] == 0:
                contingency_matrix[cluster_assignment[i],0] +=1
            elif y[i] == 8:
                contingency_matrix[cluster_assignment[i],1] +=1
            elif y[i] == 9:
                contingency_matrix[cluster_assignment[i],2] +=1

        return num_iter, self.error_history, contingency_matrix

    def compute_error(self,X,cluster_assignment):
        # compute the reconstruction error for given cluster assignment and centers
        error = 0 # placeholder
        for i in range(len(X)):
                error+=np.sum((((X[i] - self.center[cluster_assignment[i]]))**2))
        return error

    def params(self):
        return self.center
