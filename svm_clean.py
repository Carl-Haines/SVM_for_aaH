import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import datetime
from sklearn import svm
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix


threshold = 40.1  # Storm threshold
class_weight = 1  # SVM hyperparameter
gamma = 0.01  # SVM hyperparameter
C_ = 0.1  # SVM hyperparameter
training_period = 8  # number of 3-hourly data-points
lead_time = 3  # Lead time of forecast in hours


def read_data(file_path):
    '''
    reads in aaH data
    :param file_path: str
    :return: dataframe of dates and data. Dates are in UNIX time
    '''
    try:
        my_data = pd.read_table(file_path, header=None, delim_whitespace=True)  # import data as pandas dataframe, separate columns at whitespace
    except FileNotFoundError:
        print("Your input file does not exist.")
        raise FileNotFoundError('Your input file does not exist.')

    my_data.columns = ['year', 'month', 'day', 'hour', '5', 'data', '7', '8', '9', '10', '11']  # name columns
    my_data['datetime'] = pd.to_datetime(my_data[['year', 'month', 'day', 'hour']])   # make colunm with timestamp
    my_data = my_data.drop(['year', 'month', 'day', 'hour', '5', '7', '8', '9', '10', '11'], axis=1) # keep only relevent colunms
    my_data['datetime'] = my_data['datetime'].astype(np.int64) // 10 ** 9 # Convert datetime into Unix time
    return my_data


def confusion_table(true, pred):
    """
    computes the number of TP, TN, FP, FN events given the arrays with observations and predictions
    and returns the true skill score

    Args:
    true: np array with observations (1 for scintillation, 0 for nonscintillation)
    pred: np array with predictions (1 for scintillation, 0 for nonscintillation)

    Returns: true negative, false positive, true positive, false negative
    """
    Nobs = len(pred)
    TN = 0.
    TP = 0.
    FP = 0.
    FN = 0.
    for i in range(Nobs):
        if (pred[i] == 0 and true[i] == 0):
            TN += 1
        elif (pred[i] == 1 and true[i] == 0):
            FP += 1
        elif (pred[i] == 1 and true[i] == 1):
            TP += 1
        elif (pred[i] == 0 and true[i] == 1):
            FN += 1
        else:
            print("Error! Observation could not be classified.")
    return TN, FP, TP, FN


# Function to plot the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Contingency table',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Contingency Table")
    else:
        print('Contingency Table')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14, weight='bold')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=20)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,fontsize=14)#, weight='bold')
    plt.yticks(tick_marks, classes,fontsize=14)#, weight='bold')

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontsize=22, weight='bold')

    plt.tight_layout()
    plt.ylabel('True label', fontsize=14, weight='bold')
    plt.xlabel('Predicted label', fontsize=14, weight='bold')

aaH = read_data('./data/aaH_data.txt')

predicted_label = 'data'
predicted_column = aaH[predicted_label].values

# transform data
X = np.full((len(predicted_column),training_period+1), np.NaN)
for i in range(training_period+1):
    X[:, i] = np.roll(aaH['data'].values, i + int((lead_time/3)-1))
X = X[(training_period-1):, :]
y = X[:, 0]
X = X[:,1:]

# SVM training
y[y < threshold] = 0
y[y >= threshold] = 1

# Separate data into training and testing data by alternating years

idx_train =[]
idx_test = []
for i in range(0,150,2):
    print(i)
    idx_train = np.append(idx_train, i*365*8 + np.arange(0,365*8))
    idx_test = np.append(idx_test, (i+1) * 365 * 8 + np.arange(0, 365 * 8))

X_train = X[idx_train. astype(int)]
X_test = X[idx_test. astype(int)]
y_train = y[idx_train. astype(int)]
y_test = y[idx_test. astype(int)]

# Create input data scaler based only on training set
scaler_X = RobustScaler()
scaler_X = scaler_X.fit(X_train)

X_train_scaled = scaler_X.transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Create the SVM model
clf = svm.SVC(kernel='rbf', C=0.1, gamma=0.01, class_weight={1:class_weight},  probability=True)
clf.fit(X_train_scaled, y_train)

# Test and evaluate
pred = clf.predict(X_test_scaled)

TN,FP,TP,FN = confusion_table(y_test,pred)
confusion_matrix_svm = confusion_matrix(y_test, pred)
confusion_matrix_class_names = ['no storm','storm']

fig1 = plt.figure()
plot_confusion_matrix(confusion_matrix_svm, classes=confusion_matrix_class_names, normalize=False,
                              title='SVM contingency table: Class weight ' + str(class_weight))
fig1.tight_layout()

fig = plt.figure()
plot_confusion_matrix(confusion_matrix_svm, classes=confusion_matrix_class_names, normalize=True,
                          title='SVM normalised contingency table: Class weight ' +str(class_weight))
fig.tight_layout()
plt.show()