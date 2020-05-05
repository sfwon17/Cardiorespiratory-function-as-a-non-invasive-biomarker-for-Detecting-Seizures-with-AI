# Cardiorespiratory function as a non-invasive biomarker for detecting seizures with artificial intelligence: Utilising data from long-term patient recordings from an epilepsy monitoring unit

The codes and algorithms are based on:
1. https://github.com/PatrickYu1994/Epilepsy  
2. https://dl.acm.org/doi/fullHtml/10.1145/3373017.3373055

Refer to the list above for more information about dataset and pre-processing steps. Contact wongsh@deakin.edu.au or zongyuan.ge@monash.edu if you have any questions. If you have questions with the data set as it is not publicly available, please contact shobi.sivathamboo@monash.edu

## Requirements
- Python 3
- Tensorflow
- Keras 

## Overview
The task is to create a seizure detection algorithm that can detect seizure and non-seizure segment using EEG and cardiorespiratory data such as ECG, EMG, THO and Air-flow. The goal was to improve the result produced in the initial stage. Additional steps such as feature engineeri,extraction and selection, pre-processing and metrics evaluation were taken.

## Features
Features that were considered:
1. Kurtosis
2. Skewness
3. Zero Crossing
4. PSD
5. Hjorth parameters
6. Entropy 
7. Amplitude 

## Metrics
1. AUC 
2. Optimal Threshold (Youden Index)
3. ACC
4. Proportion of seizure correctly predicted
5. Average time taken to predict seizure
6. Sensitivty 
7. Specificity
8. Precision
9. False positive rate
10. False negative rate
