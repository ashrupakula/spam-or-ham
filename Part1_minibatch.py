import pandas as pd
import numpy as np
from nltk.stem.porter import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

HAM = 0
SPAM = 1
eta = 0.1
THRESHOLD = 0.5
BATCH_SIZE = 128

def pre_process():
    df = pd.read_csv("Assignment_4_data.txt", delimiter="\t", header = None)
    stemmer = PorterStemmer()
    df[2] = df[1]
    df[3] = df[1]
    df[4] = df[1]
    delimiters = [',','.',':',';','-','!','?','(',')','[',']','{','}','"',"'",'\\','/','|','\t']
    stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    for i in range(len(df)):
        print("Pre-processing Part 1: "+str(i) +'/'+str(len(df)-1),end='\r')
        for de in delimiters:
            df.iloc[i,1] = df.iloc[i,1].replace(de, ' ') #Breaking sentences into tokens
        df.iloc[i,2] = df.iloc[i,1].lower().split()
        df.iloc[i,3] = []
        for j in df.iloc[i,2]:
            if(j not in stopwords): #Remove stopwords
                df.iloc[i,3].append(j)
        df.iloc[i,4] = []
        for j in df.iloc[i,3]: #Porterstemming
            df.iloc[i,4].append(stemmer.stem(j))
    print('')


    uniquewordstemp = []
    uniquewordsfreq = []
    for i in range(len(df)): #Finding unique words and their frequencies
        for word in df.iloc[i,4]:
            if(word not in uniquewordstemp):
                uniquewordstemp.append(word)
                uniquewordsfreq.append(1)
            else:
                uniquewordsfreq[uniquewordstemp.index(word)]+=1
    uniquewords = []
    for i in range(500): #Choosing top 500 unique words
        ind = np.argmax(uniquewordsfreq)
        uniquewordsfreq[ind] = -1
        uniquewords.append(uniquewordstemp[ind])

    onehotvectorlist = []
    y_values = []
    for i in range(len(df)): #Converting data into 0's and 1's
        print("Pre-processing Part 2: "+str(i) +'/'+str(len(df)-1),end='\r')
        if(df.iloc[i,0].lower()=='ham'):
            y_values.append(HAM)
        elif(df.iloc[i,0].lower()=='spam'):
            y_values.append(SPAM)
        else:
            print("Gaddafi")
            y_values.append(-1)
        onehotvector = [0]*len(uniquewords)
        for j in df.iloc[i,4]:
            if(j in uniquewords):
                k = uniquewords.index(j)
                onehotvector[k] = 1
        onehotvectorlist.append(onehotvector)
    print('')

    df[5] = pd.Series(onehotvectorlist)
    df[6] = pd.Series(y_values)
    data_X = np.array(onehotvectorlist)
    data_Y = np.array(y_values)
    train_X, test_X, train_y, test_y = train_test_split(data_X, data_Y, test_size = 0.2, random_state = 0)
    print("Preprocessing done")
    return train_X, test_X, train_y, test_y

def random_initialise(d): #Weight_initialiser
    w = np.array([[[0]*d[l+1] for i in range(d[l])] for l in range(len(d)-1)])
    for l in range(len(d)-1):
        for i in range(d[l]):
            for k in range(d[l+1]):
                w[l][i][k] = np.random.random()/4 - 0.125         
    return w

def ReLU(a):
    if(a>0):
        return a
    return 0

def sigmoid(x):
    return 1/(1+np.exp(-x))


def forward_pass(x0, d, w): #Forward pass
    x = np.array([[0]*d[i] for i in range(len(d))])
    x[0][0] = (1)
    for i in range(1, d[0]):
        x[0][i] = x0[i-1]
    for l in range(1, len(d)-1):
        x[l][0] = 1
        for j in range(d[l]-1):
            x[l][j] = (ReLU(np.sum([w[l-1][i][j]*x[l-1][i] for i in range(d[l-1])])))
    l = len(d)-1
    for j in range(d[l]):
        x[l][j] = sigmoid(np.sum([w[l-1][i][j]*x[l-1][i] for i in range(d[l-1])]))
    return x

def back_propogation(d, x, w, y): #Backpropogation
    delta = np.array([[0]*d[i] for i in range(len(d))])
    lend = len(d)
    for i in range(d[-1]):
        delta[-1][i] = -y*(1-x[-1][i])+(1-y)*(x[-1][i])
    for l in range(1, len(d)-1):
        for i in range(1, d[lend-l-1]):
            if(x[lend-l-1][i] <= 0):
                delta[lend-1-l][i] = 0
            else:
                delta[lend-l-1][i] = np.dot(w[lend-l-1][i],delta[lend-l])
    return delta

def compute_error_point(x, y):
    if(y==1):
        return -(y*np.log(x[-1][0])+np.finfo(float).eps)
    if(y==0):
        return -((1-y)*np.log(1-x[-1][0]))
    return -(y*np.log(x[-1][0])+np.finfo(float).eps)-((1-y)*np.log(1-x[-1][0]))

def update_weights(w, grad, d, lent):
    w = [[[w[l][i][j] - eta*grad[l][i][j] for j in range(d[l+1])] for i in range(d[l])] for l in range(len(d)-1)]
    return w
        
def train(w, X, y, d, epoch): #Training
    ite = 0
    while(ite<len(X)):
        grad = np.array([[[0]*d[l+1] for i in range(d[l])] for l in range(len(d)-1)])
        X_batch = X[ite:ite+BATCH_SIZE, :] #Data Loader
        y_batch = y[ite:ite+BATCH_SIZE]
        for i in range(len(X_batch)):
            print('Epoch '+str(epoch)+':'+str(ite+i)+str('/')+str(len(X)), end = '\r')
            x = forward_pass(X_batch[i,:], d, w)
            delta = back_propogation(d, x, w, y_batch[i])
            grad = [[[grad[l1][i1][j1]+x[l1][i1]*delta[l1+1][j1] for j1 in range(d[l1+1])] for i1 in range(d[l1])] for l1 in range(len(d)-1)]
        w = update_weights(w, grad, d, len(X_batch))
        ite+=len(X_batch)
    print('Epoch '+str(epoch)+':'+str(ite)+str('/')+str(len(X)))
    return w

def train2(w, X, y, d, epoch):
    print('Epoch '+str(epoch)+':'+str(epoch)+str('/')+str(len(X)), end = '\r')
    x = forward_pass(X[epoch,:], d, w)
    delta = back_propogation(d, x, w, y[epoch])
    w = update_weights(w, x, delta, d)
    print('')
    return w
        
def compute_error(X, y, w, d):
    error = 0
    for i in range(len(X)):
        print("Error computing: "+str(i) +'/'+str(len(X)-1), end = '\r')
        x = forward_pass(X[i,:], d, w)
        #print(x[2][0])
        temperror = compute_error_point(x, y[i])
        if((np.isnan(temperror))|(np.isinf(temperror))):
            temperror = 10000
        error += temperror
    error/=len(X)
    print('')
    return error



train_X, test_X, train_y, test_y = pre_process()
d = [501, 101, 1]
w = random_initialise(d)
train_err = []
test_err = []
epoch = 1
epsilon = 100
while(epsilon>0.005):
    #print('Epoch '+str(epoch)+':', end = '')
    w = train(w, train_X, train_y, d, epoch)
    train_err.append(compute_error(train_X, train_y, w, d))
    print("Training Error: " + str(train_err[-1]))
    test_err.append(compute_error(test_X, test_y, w, d))
    print("Test Error: " + str(test_err[-1]))
    if(epoch>=3):
        epsilon = np.abs(train_err[-1]- train_err[-2])
        if(train_err[-1]>train_err[-2]):
            break
    epoch+=1

plt.plot(np.arange(1, len(train_err)+1, 1), train_err, label = "Training Error")
plt.plot(np.arange(1, len(test_err)+1, 1), test_err, label = "Testing Error")
plt.title("Error vs. No. of Epochs")
plt.xlabel("Epochs")
plt.ylabel("Average Error")
plt.legend()
plt.savefig("Part1_minibatch_graph.png", bbox_inches = 'tight')

print("Training Error: "+str(train_err))
with open('training_error_p1_minibatch.txt', 'w') as f:
    for item in train_err:
        f.write(str(item)+'\n')
print("Test Error: "+str(test_err))
with open('test_error_p1_minibatch.txt', 'w') as f:
    for item in test_err:
        f.write(str(item)+'\n')

print("Threshold: "+str(THRESHOLD))
count = 0
for i in range(len(train_X)):
    x = forward_pass(train_X[i,:], d, w)
    if(x[-1][0]>THRESHOLD):
        yhat = SPAM
    else:
        yhat = HAM
    if(yhat == train_y[i]):
        count+=1
print("Train Accuracy: "+str(count/len(train_X)))

#Test
count = 0
for i in range(len(test_X)):
    x = forward_pass(test_X[i,:], d, w)
    if(x[-1][0]>THRESHOLD):
        yhat = SPAM
    else:
        yhat = HAM
    if(yhat == test_y[i]):
        count+=1
print("Test Accuracy: "+str(count/len(test_X)))