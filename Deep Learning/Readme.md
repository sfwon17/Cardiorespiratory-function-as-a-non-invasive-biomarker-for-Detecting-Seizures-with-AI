## Requirements 
- keras
- tensorflow

## Explanation 
LSTM were used similarly to the traditional models where top performing channels were selected to produce a prediction. Different hyperparameters were tested. 

## Instruction
Run train.py and find out the top performing channels. Replace the directory path to those of the top performing channels in test.py. Replace the "numbering" to the number of top performing channels before running on test.py  

## Notes 
Since LSTM will only works when there are sequences in dataset. Please make sure dataset used are in sequence when running the models. Sequence dataset will likely produce a better prediction. 
