# import packages
import pandas as pd
import numpy as np
import csv
import os.path
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
#import Assignments.3.Bayesian-learning-and-boosting.lab3 


def preprocessing(filename, ev=False):
    # replace missing values with nan
    missing_value_types = ["ooh", "?", "nan", ""]
    df = pd.read_csv(filename, na_values=missing_value_types)
    print(df.head(3))
    # print datatypes for each column, object = string
    #print(df.dtypes)

    # find rows with missing values
    index_missing = np.where(df.isnull()==True)[0]
    index = np.unique(index_missing)
    # loop through them and remove them
    for i in index:
        df = df.drop([i], axis=0)

    # find unique values in string columns  
    if ev == False: 
        char2y = {u:i for i, u in enumerate(df['y'])}
        char2x5 = {u:i for i, u in enumerate(df['x5'])}
        char2x6 = {u:i for i, u in enumerate(df['x6'])}
        print(char2y)
        print(char2x5)
        print(char2x6)

    # cast categorical columns to pandas categorical and create a one-hot encoding
    df['x5'] = pd.Categorical(df['x5'])
    dfDummiesX5 = pd.get_dummies(df['x5'], prefix='x5')
    df['x6'] = pd.Categorical(df['x6'])
    dfDummiesX6 = pd.get_dummies(df['x6'], prefix='x6')
    print(dfDummiesX5.head(3), dfDummiesX6.head(3))
    # concatenate to original dataframe
    df = pd.concat([df, dfDummiesX5, dfDummiesX6], axis=1)
    print(df.head(3))
    # drop x5 and x6
    df = df.drop('x5', axis=1)
    df = df.drop('x6', axis=1)
    print(df.head(3))
    print(df.shape)
    return df

def split_data(df, split, from_end, random_seed):
    # Create random_seed
    np.random.seed(random_seed)
    df.iloc[np.random.permutation(len(df))]
    # Make y labels into numbers 
    df.y = pd.Categorical(df.y)
    # Add to end of dataframe
    df['y_code'] = df.y.cat.codes

    # Split evenly over classes
    # 183 is smallest number of same class
    start = int(183*split)
    i_bob_s = (np.where(df['y'] == "Bob")[0])[0:start]
    i_bob_e = (np.where(df['y'] == "Bob")[0])[start:183]
    i_atsuto_s = (np.where(df['y'] == "Atsuto")[0])[0:start]
    i_atsuto_e = (np.where(df['y'] == "Atsuto")[0])[start:183]
    i_jorg_s = (np.where(df['y'] == "Jörg")[0])[0:start]
    i_jorg_e = (np.where(df['y'] == "Jörg")[0])[start:183]

    # Make pandas train and test dataset to numpy arrays
    x_train_b = df.iloc[i_bob_s, 2:from_end]
    x_train_a = df.iloc[i_atsuto_s, 2:from_end]
    x_train_j = df.iloc[i_jorg_s, 2:from_end]
    x_train_p = x_train_b.append(x_train_a).append(x_train_j)
    x_train = x_train_p.to_numpy()

    y_train_b = df.iloc[i_bob_s, [-1]]
    y_train_a = df.iloc[i_atsuto_s, [-1]]
    y_train_j = df.iloc[i_jorg_s, [-1]]
    y_train_p = y_train_b.append(y_train_a).append(y_train_j)
    y_train = y_train_p.to_numpy().ravel()

    x_test_b = df.iloc[i_bob_e, 2:from_end]
    x_test_a = df.iloc[i_atsuto_e, 2:from_end]
    x_test_j = df.iloc[i_jorg_e, 2:from_end]
    x_test_p = x_test_b.append(x_test_a).append(x_test_j)
    x_test = x_test_p.to_numpy()

    y_test_b = df.iloc[i_bob_e, [-1]]
    y_test_a = df.iloc[i_atsuto_e, [-1]]
    y_test_j = df.iloc[i_jorg_e, [-1]]
    y_test_p = y_test_b.append(y_test_a).append(y_test_j)
    y_test = y_test_p.to_numpy().ravel()

    # shuffle training dataset
    rng_state = np.random.get_state()
    np.random.shuffle(x_train)
    np.random.set_state(rng_state)
    np.random.shuffle(y_train)
    np.random.set_state(rng_state)
    np.random.shuffle(x_test)
    np.random.set_state(rng_state)
    np.random.shuffle(y_test)
    print(y_test.shape, y_train.shape, x_test.shape, x_train.shape)
    print(x_train_p)
    return x_train, y_train, x_test, y_test

def evaluation_dataset(df):
    x_val = df.iloc[:, 1:]
    return x_val

def find_best_split(df):
    for split in range(60, 95):
        split /= 100
        x_train, y_train, x_test, y_test = split_data(df, split, -1, 100)
        # Classify Gaussian Naive Bayes
        print("Split:", split)
        gnb = GaussianNB()
        y_pred = gnb.fit(x_train, y_train).predict(x_test)
        acc = 1 - (y_test != y_pred).sum()/x_test.shape[0]
        print("Bayes: Number of mislabeled points out of a total %d points : %d"  % (x_test.shape[0], (y_test != y_pred).sum()))
        print("Accuracy: ", acc)

        # Classify Decision Tree
        clf = DecisionTreeClassifier()
        y_pred_dt = clf.fit(x_train, y_train).predict(x_test)
        acc = 1 - (y_test != y_pred_dt).sum()/x_test.shape[0]
        print("DT: Number of mislabeled points out of a total %d points : %d"  % (x_test.shape[0], (y_test != y_pred_dt).sum()))
        print("Accuracy: ", acc)

        print("=========================================")

def plot_boost(bdt_real, bdt_discrete):
    real_test_errors = []
    discrete_test_errors = []

    for real_test_predict, discrete_train_predict in zip(
            bdt_real.staged_predict(X_test), bdt_discrete.staged_predict(X_test)):
        real_test_errors.append(
            1. - accuracy_score(real_test_predict, y_test))
        discrete_test_errors.append(
            1. - accuracy_score(discrete_train_predict, y_test))

    n_trees_discrete = len(bdt_discrete)
    n_trees_real = len(bdt_real)

    # Boosting might terminate early, but the following arrays are always
    # n_estimators long. We crop them to the actual number of trees here:
    discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]
    real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]
    discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_trees_discrete]

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.plot(range(1, n_trees_discrete + 1),
            discrete_test_errors, c='black', label='SAMME')
    plt.plot(range(1, n_trees_real + 1),
            real_test_errors, c='black',
            linestyle='dashed', label='SAMME.R')
    plt.legend()
    plt.ylim(0.18, 0.62)
    plt.ylabel('Test Error')
    plt.xlabel('Number of Trees')

    plt.subplot(132)
    plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_errors,
            "b", label='SAMME', alpha=.5)
    plt.plot(range(1, n_trees_real + 1), real_estimator_errors,
            "r", label='SAMME.R', alpha=.5)
    plt.legend()
    plt.ylabel('Error')
    plt.xlabel('Number of Trees')
    plt.ylim((.2,
            max(real_estimator_errors.max(),
                discrete_estimator_errors.max()) * 1.2))
    plt.xlim((-20, len(bdt_discrete) + 20))

    plt.subplot(133)
    plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_weights,
            "b", label='SAMME')
    plt.legend()
    plt.ylabel('Weight')
    plt.xlabel('Number of Trees')
    plt.ylim((0, discrete_estimator_weights.max() * 1.2))
    plt.xlim((-20, n_trees_discrete + 20))

    # prevent overlapping y-axis labels
    plt.subplots_adjust(wspace=0.25)
    plt.show()

def find_best_nr_estimators():
    real_error = []
    discrete_error = []
    estimator = []
    for n_est in range(10, 100, 1):
        bdt_real = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=n_est, learning_rate=1)
        bdt_discrete = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=n_est, learning_rate=1.5, algorithm="SAMME")

        bdt_real.fit(X_train, y_train)
        bdt_discrete.fit(X_train, y_train)
        real_error.append(bdt_real.score(X_test, y_test))
        discrete_error.append(bdt_discrete.score(X_test, y_test))
        estimator.append(n_est)
    
    plt.plot(estimator, real_error, '-g', discrete_error, '-b')
    plt.legend(('real_error', 'discrete_error'),
            loc='upper right')
    plt.title('Boosted Decisiton Tree')
    plt.show()

print("=======================Preprocessing training data======================= ")
filename = 'TrainOnMe.csv'
df = preprocessing(filename)
print("=======================Preprocessing evaluation data======================= ")
df_ev = preprocessing('EvaluateOnMe.csv', ev=True)
print("=======================Creating evaluation data======================= \n")
X_val = evaluation_dataset(df_ev)
print("=======================Splitting training data======================= ")
# Which columns to use, -1, -6, -8
X_train, y_train, X_test, y_test = split_data(df, 0.85, -1, 100)
print("=======================Decision Tree Classifier======================= ")
clf = DecisionTreeClassifier(max_depth=4, splitter='best', criterion='gini')
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)

bdt_real = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=65, learning_rate=1)
bdt_real.fit(X_train, y_train)
acc_real = bdt_real.score(X_test, y_test)

rf = RandomForestClassifier(max_depth=4, random_state=0, n_estimators=150)
rf.fit(X_train, y_train)
acc_rf = rf.score(X_test, y_test)

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
acc_gnb = 1 - (y_test != y_pred).sum()/X_test.shape[0]

gnb_boost = AdaBoostClassifier(GaussianNB(), n_estimators=10, learning_rate=1)
gnb_boost.fit(X_train, y_train)
acc_gnb_boost = gnb_boost.score(X_test, y_test)

print("NB non boosted", acc_gnb,"\nNB boosted", acc_gnb_boost)
print("DT non boosted", acc, "\nDT boosted", acc_real)
print("random forest:", acc_rf)
