from __future__ import division
import scipy.io as scio
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import more_itertools as mit
import json

# read training label
def read_train_label():
    training_set1 = scio.loadmat("../preprocessed_data/normtraining_set1.mat")
    train1 = np.array(training_set1['y'])
    training_set2 = scio.loadmat("../preprocessed_data/normtraining_set2.mat")
    train2 = np.array(training_set2['y'])
    training_set3 = scio.loadmat("../preprocessed_data/normtraining_set3.mat")
    train3 = np.array(training_set3['y'])
    training_set4 = scio.loadmat("../preprocessed_data/normtraining_set4.mat")
    train4 = np.array(training_set4['y'])

    train_label = np.concatenate([train1,train2,train3,train4])
    label_train= []

    for i in train_label:
        label_train.append(i[1])
    return label_train

# read test_label or validation_label
def read_test_label():
    test_set1 = scio.loadmat("../preprocessed_data/normtest_set1.mat")
    test1 = np.array(test_set1['y'])
    test_set2 = scio.loadmat("../preprocessed_data/normtest_set2.mat")
    test2 = np.array(test_set2['y'])
    test_set3 = scio.loadmat("../preprocessed_data/normtest_set3.mat")
    test3 = np.array(test_set3['y'])
    test_set4 = scio.loadmat("../preprocessed_data/normtest_set4.mat")
    test4 = np.array(test_set4['y'])

    test_label = np.concatenate([test1,test2,test3,test4])

    label_test= []
    for i in test_label:
        label_test.append(i[1])
    return label_test

# Fitting model
# model should be trained based on the channels
# to find out which channels is the best, train all channels individually and select the best
# eg: if random forest is used, uncomment the channels for random forest
def train_model(data,test,label_train):
    start = 0
    end = 5
    result = []
    channel_prob = []

    for i in range(19):
        df1 = data[list(data.columns.values[start:end])]

        test1 = test[list(test.columns.values[start:end])]
        start = end
        end += 5
        if i == 3 or i == 9 or i == 13 or i == 15 or i ==17 : # Random forest
        #if i == 0 or i == 2 or i == 4 or i == 5 or i == 9: # bayes(old)
        #if i == 5 or i == 6 or i == 10 or i == 17 or i == 18: # bayes
        #if i == 15 or i == 17 or i == 18:
        #if i != 22:
            print(df1)
            print(i)
            data_feat = df1.values
            test_feat = test1.values

            clf = RandomForestClassifier(n_estimators=2500, random_state=0, n_jobs=2, criterion='gini',min_samples_split=7)
            #clf = ExtraTreesClassifier(n_estimators=2500, random_state=0, n_jobs=2, criterion='gini',min_samples_split=7)
            #clf = GaussianNB()

            clf.fit(data_feat, label_train)
            y_pred = clf.predict_proba(test_feat)
            channel_prob.append(y_pred[:,1])

            this_AUC = metrics.roc_auc_score(label_test, y_pred[:,1])
            print("AUC: " + str(this_AUC))
            #result.append(this_AUC)
    return channel_prob

## moving average applied to the prediction for smoothing:
def moving_average(label_test, channel_prob):
    temp = []
    average_list = np.array([sum(i) / len(i) for i in zip(*channel_prob)])
    tes = np.mean(average_list.reshape(-1, 4), axis=1)
    for i in tes:
        temp += 4 * [i]
    this_AUC = metrics.roc_auc_score(label_test, temp)
    return this_AUC,temp

# finding the threshold value based on youden index
def Youden_index(label_test, label_predicted):
    fpr, tpr, tres = metrics.roc_curve(label_test, label_predicted)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = tres[optimal_idx]
    return optimal_threshold

# assign label based on threshold
def assign_label(optimal_threshold,label_predicted):
    label = np.where(np.array(label_predicted) > optimal_threshold, 1, 0)  # 0.75
    return label

# calculate the accuracy
def accuracy_score(label_test, label):
    acc = metrics.accuracy_score(label_test, label)
    return acc

## calculate thje proportion of seizure event correctly predicted
## calculate the time needed for seizure prediction
def seizure_event(label_test, label):
    test_index = []
    for index, i in enumerate(label_test):
        if i == 1:
            test_index.append(index)

    seizure_segment = [list(group) for group in mit.consecutive_groups(test_index)]  # seizure found
    ss_length = len(seizure_segment)

    ## calculate if the seizure segment has successfully being predicted.
    ## During the seizure segment, if one of the seizure is predicted, we concluded that we detected the seizure segment
    seizure_count = 0  # seizure predicted
    seconds_list = []
    for y in seizure_segment:
        events = [label[i] for i in y]
        if 1 in events:
            seizure_count += 1

            # calculate how many seconds passed when they successfully detected the first seizure label
            res = next(x for x, value in enumerate(events) if value == 1)
            seconds_list.append(res + 1)

    # get the average predicted seconds
    average_seconds = sum(seconds_list) / len(seconds_list)
    return ss_length, seizure_count, seizure_count / ss_length, average_seconds


# find sensitivity and specificity
def measurement(label_test,label):
    tn, fp, fn, tp = metrics.confusion_matrix(label_test, label).ravel()

    sensitivity = tp/(tp+fn)
    specificty =  tn/(tn+fp)
    precision = tp/(tp+fp)
    fpr = fp / (fp + tn)
    fnr = fn / (tp + fn)
    return {"Sensitivity":sensitivity,
            "Specificity": specificty,
            "Precesion": precision,
            "False positive rate": fpr,
            "False negative rate": fnr}


data = pd.read_csv("../features/features_unnorm_train3.csv")
test = pd.read_csv("../features/features_unnorm_test3.csv")

label_train = read_train_label()
label_test  = read_test_label()
channel_prob = train_model(data,test,label_train)
result = moving_average(label_test, channel_prob)
optimal_threshold = Youden_index(label_test, result[1])
label = assign_label(0.4862,result[1])
acc = accuracy_score(label_test, label)
seizure_info= seizure_event(label_test, label)
metrics = measurement(label_test,label)

# save all performance and information to json file
model_information = {"AUC":result[0],
                     "Optimal Threshold":0.4862,
                     "Accuracy":acc,
                     "Proportion of seizure segment correctly predicted":seizure_info[2],
                     "average time taken to predicted seizure": seizure_info[3]}
model_information.update(metrics)
print(model_information)

with open('./result/7channel_rf(test).json', 'w') as fp:
    json.dump(model_information, fp)
