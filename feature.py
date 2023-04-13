import csv
import numpy as np
import argparse

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


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
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file) as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map
def trim(inputRow,glove):
    ans = []
    words = inputRow.split()
    for word in words:
        temp = glove.get(word)
        if(temp is None):
            continue
        else:
            ans.append(word)
    return ans
def sumSentence(row,glove):
    length = len(row)
    ans = np.zeros(300)
    for i in range(0,len(row)):
        temp = glove.get(row[i])
        ans = np.add(ans,temp)
    np4 = ans/(length)
    return np4

    
def getLabels(input):
    ans = np.zeros(len(input))
    for i in range(len(input)):
        currLabel = input[i][0]
        ans[i] = currLabel
    return ans

def output(array,file):
    f= open(file,"w+")
    for row in array:
        for index in range(len(row)):
            l = "{:.6f}".format(row[index])
            f.write(str(l))
            if(index!=len(row)-1):
                f.write("\t")
        f.write("\n")
    
    f.close();
def main(dicty, trainin):
    glove = load_feature_dictionary(dicty)
    #train data
    a = load_tsv_dataset(trainin)
    ans = np.zeros(shape=(len(a),300))
    for index in range(len(a)):
        temp = trim(a[index][1],glove)
        noLabels = sumSentence(temp,glove)
        ans[index] = noLabels
    labels = getLabels(a)
    finalans = np.insert(ans,0,labels,axis=1)
 
    trainout = args.train_out
    output(finalans,trainout)

    #validation data
    b = load_tsv_dataset(args.validation_input)
    ans2 = np.zeros(shape=(len(b),300))
    for index2 in range(len(b)):
        temp = trim(b[index2][1],glove)
        noLabels = sumSentence(temp,glove)
        ans2[index2] = noLabels
    labels = getLabels(b)
    finalans2 = np.insert(ans2,0,labels,axis=1)
    output(finalans2,args.validation_out)
    
    #test data
    c = load_tsv_dataset(args.test_input)
    ans3 = np.zeros(shape=(len(c),300))
    for index3 in range(len(c)):
        temp = trim(c[index3][1],glove)
        noLabels = sumSentence(temp,glove)
        ans3[index3] = noLabels
    labels = getLabels(c)
    finalans3 = np.insert(ans3,0,labels,axis=1)
    output(finalans3,args.test_out)

    
    

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str, 
                        help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("validation_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the test data should be written')
    args = parser.parse_args()
    main(args.feature_dictionary_in,args.train_input)
