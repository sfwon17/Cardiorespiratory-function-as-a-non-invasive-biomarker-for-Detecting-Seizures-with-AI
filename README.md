# Cardiorespiratory function as a non-invasive biomarker for detecting seizures with artificial intelligence: Utilising data from long-term patient recordings from an epilepsy monitoring unit

The codes and algorithms are based on: 
1. https://github.com/PatrickYu1994/Epilepsy  
2. https://dl.acm.org/doi/fullHtml/10.1145/3373017.3373055

Refer to the list above for more information about the dataset and pre-processing steps. Contact wongsh@deakin.edu.au or zongyuan.ge@monash.edu if you have any questions related to the model. If you have questions regarding the data set as it is not publicly available, please contact shobi.sivathamboo@monash.edu

## Requirements
- Python 3
- Tensorflow
- Keras 

## Overview
The task is to create a seizure detection algorithm that can detect seizure and non-seizure segment using EEG and cardiorespiratory data such as ECG, EMG, THO and Air-flow. The goal was to improve the result produced in the initial stage. Additional steps such as feature engineering, extraction and selection, pre-processing and metrics evaluation were taken.

## Explanation
Run pre-processing.py and features_generation.py in that order and will produce 3 files which are training, validation and testing set. Please look at the subfolders for different models used for the prediction. Dataset were run on batches because of memory limitation and will not affect the final result of a model significantly. The final models especially traditional models might not be the best or ideal model in this scenario, but it will hopefully provide additional insight of how extracting features helps in improving the performance of the prediction. 

## Features
Features that were considered:
1. Statistical properties such as kurtosis and skewness
2. Amplitude-based features
3. Frequency-based features 
4. Time-Frequency based features
5. Wavelett-based features
6. PSD
7. Other features such as zero-crossing, entropy, Hjorth parameters

## Metrics
1. AUC 
2. Optimal Threshold (Youden Index)
3. Accuracy
4. Proportion of seizure correctly predicted
5. Average time taken to predict seizure
6. Sensitivty 
7. Specificity
8. Precision
9. False positive rate
10. False negative rate

## Conclusion
Despite their reduced complexity, simpler models achieved better performance compared to more complex CNN models when using extracted features. This demonstrates that simpler models, when effectively utilizing extracted features, can outperform more complex architectures, suggesting a potential shift towards efficiency and interpretability in seizure detection models.
