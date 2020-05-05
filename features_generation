import scipy.signal as spsig
import pandas as pd
import scipy.stats as spstat
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

## Reading data patients batch fro training, validation, test
training_set1 = scio.loadmat("./normtraining_set1.mat")
train1 = np.array(training_set1['x_eeg'])
training_set2 = scio.loadmat("./normtraining_set2.mat")
train2 = np.array(training_set2['x_eeg'])
training_set3 = scio.loadmat("./normtraining_set3.mat")
train3 = np.array(training_set3['x_eeg'])
training_set4 = scio.loadmat("./normtraining_set4.mat")
train4 = np.array(training_set4['x_eeg'])


bands=[0.1,4,8,12,30,70]
test_data = np.concatenate([test1,test2,test3,test4])
print(test_data.shape)

#test_set1 = scio.loadmat("./normvalidation_set1.mat")
#val1 = np.array(test_set1['x_eeg'])
#test_set2 = scio.loadmat("./normvalidation_set2.mat")
#val2 = np.array(test_set2['x_eeg'])
#test_set3 = scio.loadmat("./normvalidation_set3.mat")
#val3 = np.array(test_set3['x_eeg'])
#test_set4 = scio.loadmat("./normvalidation_set4.mat")
#val4 = np.array(test_set4['x_eeg'])

#val_data = np.concatenate([val1,val2,val3,val4])
#print(val_data.shape)


## Extracting features
def feature_gen(dataset):
    
    data = []
    # for each segment
    for j in range(dataset.shape[0]):
        print(j)
        featureList = []
        output = []
        # for each channel
        for k in range(19):
            test = dataset[j][:,k]

            # average amplitude
            if j >= 1:
                prev = dataset[j - 1][:, k]
                rav = (np.sqrt((test ** 2).mean())) / (np.sqrt((prev ** 2).mean()))
                featureList.append('rav%i' % (k))
                output.append(rav)
            else:
                featureList.append('rav%i' % (k))
                output.append(0)

            test = preprocessing.scale(test,  with_std=True, copy = False)

            #filtered = lowess(test,list(range(1, 1001)), frac=0.5)
            #test = filtered[:,1]


            # mean
            #featureList.append('mean%i' % (k))
            #output.append(test.mean())

            # sigma
            #featureList.append('sigma%i'%(k))
            #output.append(test.std())

            # kurt
            #featureList.append('kurt%i' % (k))
            #output.append(spstat.kurtosis(test))

            # skew

            #featureList.append('skew%i' % (k))
            #output.append(spstat.skew(test))

            # zero
            #featureList.append('zero%i' % (k))
            #output.append(((test[:-1] * test[1:]) < 0).sum())

            # rms
            #featureList.append('RMS%i'%(k))
            #output.append(np.sqrt((test ** 2).mean()))

            # psd
            f, psd = spsig.welch(test, fs=80)

            # max energy
            featureList.append('MaxF%i' % (k))
            output.append(psd.argmax())

            # sum enegry
            featureList.append('SumEnergy%i' % (k))
            output.append(psd.sum())

            diff = np.diff(test, n=1)
            diff2 = np.diff(test, n=2)

            # Hjorth parameter
            featureList.append('Mobility%i' % (k))
            output.append(np.std(diff) / test.std())

            featureList.append('Complexity%i' % (k))
            output.append(np.std(diff2) * np.std(test) / (np.std(test) ** 2.))

            # bands
            #psd /= psd.sum()
            #for c in range(1, len(bands)):

                # start = time.time()
             #   featureList.append('BandEnergy%i%i' % (k, c))
             #   output.append(psd[(f > bands[c - 1]) & (f < bands[c])].sum())

            # bands entropy

            #featureList.append('entropy%i'%(k))
            #output.append(-1.0*np.sum(psd[f>bands[0]]*np.log10(psd[f>bands[0]])))


        data.append(pd.DataFrame({'features': output}, index=featureList).T)
    datatable = pd.concat(data,ignore_index=True)
    print("done")
    return datatable


# export data to csv file
train_features = feature_gen(train_data)
train_features.to_csv("features_unnorm_train3.csv", index=False)
test_features = feature_gen(test_data)#test_features.to_csv("features_unnorm_test3.csv", index=False)
val_features = feature_gen(val_data)
val_features.to_csv("features_unnorm_val3.csv", index=False)
