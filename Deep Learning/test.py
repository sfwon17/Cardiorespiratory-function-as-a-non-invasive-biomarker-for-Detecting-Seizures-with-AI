import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.io as scio
from utils import auc, f1
from sklearn import metrics
import more_itertools as mit
import json

test_set1 = scio.loadmat("../../preprocessed_data/normtest_set1.mat")
test1 = np.array(test_set1['y'])
test_set2 = scio.loadmat("../../preprocessed_data/normtest_set2.mat")
test2 = np.array(test_set2['y'])
test_set3 = scio.loadmat("../../preprocessed_data/normtest_set3.mat")
test3 = np.array(test_set3['y'])
test_set4 = scio.loadmat("../../preprocessed_data/normtest_set4.mat")
test4 = np.array(test_set4['y'])

test_label = np.concatenate([test1,test2,test3,test4])

label_test = []
for i in test_label:
    label_test.append(i[1])

best_model1 = './7/20200131best_model(channel).117-0.8010.h5'
best_model2 = './15/20200131best_model(channel).17-0.8308.h5'
best_model3 = './17/20200131best_model(channel).116-0.8007.h5'
best_model4 = './18/20200131best_model(channel).02-0.8047.h5'

test = pd.read_csv("../../features/features_unnorm_test3.csv")

def predict(model,test_set):
    test_feat = test_set.values
    test = test_feat.reshape((test_feat.shape[0], 1, test_feat.shape[1]))
    model = tf.keras.models.load_model(model, custom_objects={"auc": auc, "f1": f1})
    prediction = model.predict_proba(test)
    return prediction

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

channel_num = 0
start = 0
end = 5

result = []
channel_prob = []
temp =[]
for i in range(19):
    test1 = test[list(test.columns.values[start:end])]
    start = end
    end += 5
    if i == 7:
        print(i)
        print(test1)
        best_model1 = best_model1
        prediction = predict(best_model1,test1)
        channel_prob.append(prediction[:,1])
    if i == 15:
        print(i)
        print(test1)
        best_model1 = best_model2
        prediction = predict(best_model1,test1)
        channel_prob.append(prediction[:,1])
    if i == 17:
        print(i)
        print(test1)
        best_model1 = best_model3
        prediction = predict(best_model1,test1)

        channel_prob.append(prediction[:,1])

    if i == 18:
        print(i)
        print(test1)
        best_model1 = best_model4
        prediction = predict(best_model1,test1)
        channel_prob.append(prediction[:,1])

result = moving_average(label_test, channel_prob)
optimal_threshold = Youden_index(label_test, result[1])
label = assign_label(optimal_threshold,result[1])
acc = accuracy_score(label_test, label)
seizure_info= seizure_event(label_test, label)
metrics = measurement(label_test,label)

model_information = {"AUC":result[0],
                     "Optimal Threshold":optimal_threshold,
                     "Accuracy":acc,
                     "Proportion of seizure segment correctly predicted":seizure_info[2],
                     "average time taken to predicted seizure": seizure_info[3]}
model_information.update(metrics)
print(model_information)

with open('../result/lstm.json', 'w') as fp:
    json.dump(model_information, fp)
