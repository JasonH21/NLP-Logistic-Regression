import numpy as np
import argparse


def sigmoid(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)

def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, dtype='float')
    return dataset


def train(
    theta : np.ndarray, # shape (D,) where D is feature dim
    X : np.ndarray,     # shape (N, D) where N is num of examples
    y : np.ndarray,     # shape (N,)
    num_epoch : int, 
    learning_rate : float
) -> None:
    np1 = np.array([0])
    np2 = np.array([1])
    truetheta = np.insert(theta,0,np1,axis=0)
    for i in range(num_epoch):
        for index in range(0,len(y)):
            currRow = X[index]
            realCurrRow = np.insert(currRow,0,np2,axis=0)
            grad = np.zeros(len(realCurrRow))
            yval = y[index]
            dot = np.dot(realCurrRow,truetheta)
            for j in range(len(realCurrRow)):
                grad[j] = -1*(yval-sigmoid(dot))*realCurrRow[j]
            truetheta = truetheta-learning_rate*grad
    return truetheta
                
                

def predict(
    theta : np.ndarray,
    X : np.ndarray
) -> np.ndarray:
    ans = np.zeros(len(X))
    np2 = np.array([1])
    for r in range(len(X)):
        
        newRow = np.insert(X[r],0,np2,axis=0)
        dot = np.dot(newRow,theta)
        sig = sigmoid(dot)
        if(sig>0.5):
            ans[r]=1
        else:
            ans[r]=0
    return ans

def compute_error(
    y_pred : np.ndarray, 
    y : np.ndarray
) -> float:
    numwrong = 0
    for i in range(len(y_pred)):
        if(y_pred[i]!=y[i]):
            numwrong+=1
    return numwrong/len(y)
     
# def getX(filein):
#     ans = []
#     with open(filein) as f:
#         for line in f:
#             l=line.split('\t')
#             ans.append(l)
#     return ans

def output(array,file):
    f= open(file,"w+")
    for val in array:
        f.write(str(int(val)))
        f.write("\n")
    f.close();
def outputMetrics(trainErr,testErr,file):
    a = "{:.6f}".format(trainErr)
    b = "{:.6f}".format(testErr)
    f= open(file,"w+")
    f.write("error(train): ")
    f.write(str(a))
    f.write("\n")
    f.write("error(test): ")
    f.write(str(b))
    
def main(trainin,valin,testin,trainout,testout,metricsout,numEpoch,lr):
    a = load_tsv_dataset(trainin)
    X1 = a[:,1:]
    y1 = a[:,0]
    theta = np.zeros(len(X1[0]))
    newTheta = train(theta,X1,y1,int(numEpoch),float(lr))
    p1= predict(newTheta,X1)
    output(p1,trainout)

    b = load_tsv_dataset(testin)
    X2 = b[:,1:]
    y2 = b[:,0]
    p2 = predict(newTheta,X2)
    output(p2,testout)

    trainErr = compute_error(p1,y1)
    testErr = compute_error(p2,y2)
    outputMetrics(trainErr,testErr,metricsout)
    
    


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=str, 
                        help='number of epochs of gradient descent to run')
    parser.add_argument("learning_rate", type=str, 
                        help='learning rate for gradient descent')
    args = parser.parse_args()
    main(args.train_input,args.validation_input,args.test_input,args.train_out,args.test_out,args.metrics_out,args.num_epoch,args.learning_rate)