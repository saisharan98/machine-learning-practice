from matplotlib import pyplot as plt

# different color for different digits
color_mapping = {0:'red',1:'green',2:'blue',3:'yellow',4:'magenta',5:'orangered',
                6:'cyan',7:'purple',8:'gold',9:'pink'}

def plot2d(data,label,title,split='train'):
    # 2d scatter plot of the hidden feature
    fig = plt.figure().add_subplot()
    fig.set_title(title)
    fig.set_ylabel("Hidden Feature 1")
    fig.set_xlabel("Hidden Feature 2")
    for i in color_mapping:
        for j in range(len(label)):
            if (i==label[j]):
                fig.scatter(data[j,0], data[j,1], c=color_mapping[i], marker='.')
    title = "{}.png".format(title)
    fig.figure.savefig(title)
    pass

def plot3d(data,label,title,split='train'):
    # 3d scatter plot of the hidden features
    fig = plt.figure().add_subplot(projection = "3d")
    fig.set_title(title)
    fig.set_xlabel("Hidden Feature 0")
    fig.set_ylabel("Hidden Feature 1")
    fig.set_zlabel("Hidden Feature 2")
    for i in color_mapping:
        for j in range(len(label)):
            if (i==label[j]):
                fig.scatter(data[j,0], data[j,1], data[j,2], c=color_mapping[i],marker='.')
    title = "{}.png".format(title)       
    fig.figure.savefig(title)
    pass
