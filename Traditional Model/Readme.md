## Requirements
- sklearn
- scipy
- more_itertools

## Explanation 
This proof of concept algorithm is used to show that simple models with a combination of proper channels selection will improve the performance of prediction compared to original models. Models used includes Random Forest, Extra Trees and Gaussian Bayes. Prediction of each of the channels were then combined and smoothed out to produce a final prediction. Models with 3 and 7 channels were tested in the algorithm to examine the effect of the number of channels on final result. 

## Instruction 
To run validation and testing set, simply change the directory path of the dataset. Alternatively, create a new script by exporting the trained model and run it on the testing set. 



