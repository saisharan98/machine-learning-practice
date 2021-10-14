import numpy as np

class Tree_node:
    def __init__(self,):
        self.feature = None # index of the selected feature (for non-leaf node)
        self.class_label = None # class label (for leaf node)
        self.left_child = None # left child node
        self.right_child = None # right child node
#Functions are in the style of sklearn 
class Decision_tree:
    def __init__(self,min_entropy):
        self.min_entropy = min_entropy
        self.root = None

    def fit(self,train_x,train_y):
        # construct the decision-tree with recursion
        self.root = generate_tree(train_x,train_y,self.min_entropy)

    def predict(self,test_x):
        
        prediction = np.zeros([len(test_x),]).astype('int') 
        for i in range(len(test_x)):                                            # iterate through all samples
            cur_node = self.root                                                # start at root node
            while cur_node.right_child != None and cur_node.left_child != None: # check if the current node is a leaf or not
                if test_x[i,cur_node.feature] == 0:                             # check if the data point of the selected feature is 0
                    cur_node = cur_node.left_child                              # if yes, traverse to the left child 
                else:
                    cur_node = cur_node.right_child                             # if no, traverse to the right child 

            prediction[i] = cur_node.class_label                                # add the cur_node's (which will now be a leaf node) class label

        return prediction

def generate_tree(data,label,min_entropy):
    
    cur_node = Tree_node()                                                      # initialize the current tree node
    left_data = []
    right_data = []
    left_label = []
    right_label = []
    
    node_entropy = compute_node_entropy(label) # compute the node entropy of labels

    
    if node_entropy < min_entropy:                                              # base case
        # determine the class label for leaf node:-
        freq = np.bincount(label)                                               # find the frequencies of each class label
        maximum = np.argmax(freq)                                               # find the most occurring class label
        cur_node.class_label = maximum                                          # assign the most occurring class label as the node's label
        return cur_node

    # select the feature that will best split the current non-leaf node
    selected_feature = select_feature(data,label)                               # choose the feature that has the lowest split entropy
    cur_node.feature = selected_feature                                         # assign the that feature to the node

    # split the data based on the selected feature and start the next level of recursion
    for i in range(len(data)):
        if data[i,selected_feature] == 0:                                       # checking if every sample of the selected feature is 0 or not
            left_data.append(data[i])                                           # if 0, add those samples and labels to an array
            left_label.append(label[i])
        else:
            right_data.append(data[i])
            right_label.append(label[i])
            
    cur_node.left_child = generate_tree(np.array(left_data), np.array(left_label), min_entropy)      # recursively generate tree using left or right data
    cur_node.right_child = generate_tree(np.array(right_data), np.array(right_label), min_entropy)
    return cur_node

def select_feature(data,label):
    # iterate through all features and compute their corresponding entropy
    w_m0 = 0 #only 2 classes 0 or 1 doesn't matter what you choose
    cur_entropy = []
    for i in range(len(data[0])):
        left_y = []                                                             # make empty lists to split based on selected features
        right_y = []
        for j in range(len(data[:,i])):
        # compute the entropy of splitting based on the selected features
            if data[j,i] != w_m0:
                right_y.append(label[j])
            else:
                left_y.append(label[j])
        cur_entropy.append(compute_split_entropy(left_y,right_y)) 

        # select the feature with minimum entropy
    best_feat = np.argmin(cur_entropy)
    return best_feat

def compute_split_entropy(left_y,right_y):
    # compute the entropy of a potential split, left_y and right_y are labels for the two splits
    left_p = len(left_y)/(len(left_y) + len(right_y))                           # proportion of elements in the left and right arrays
    right_p = len(right_y)/(len(left_y) + len(right_y))
    node_entropy_left = left_p * compute_node_entropy(left_y)                   # individual node entropies 
    node_entropy_right = right_p * compute_node_entropy(right_y)
    
    split_entropy = node_entropy_left + node_entropy_right 

    return split_entropy

def compute_node_entropy(label):
    # compute the entropy of a tree node (add 1e-15 inside the log2 when computing the entropy to prevent numerical issue)
    label = np.array(label)
    classes = np.unique(label)  
    node_entropy = 0
    for i in classes:
        p_i = len(label[np.where(label==i)])/len(label)                         #probabilties of the ith class
        entropy_i = -p_i * np.log2(p_i + 1e-15)                                 #entropy of the ith class
        node_entropy += entropy_i                                               #total entropy

    return node_entropy
