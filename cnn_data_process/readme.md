# cnn_data_process folder 
## file explanation
1. **cnn_data_process.py**: This file is used to process the data for the CNN model.
It reads the data from the csv file and processes it to be used in the CNN model. The
processor includes the following steps:
    - Read the data from the csv file
    - Normalize the data
    - trim the data
    - stack the data to matrix
    - slice the data to get time window data
2. **data_spare&label.py**: This file sparate the dataset to train, val, test set and attach 
label to samples. the processor includes the following steps:
    - Read the shots info from the csv file
    - Split the data into training, validation and testing data
    - Remove other tags and left the stack data
    - cut the data to get the same length
    - label the data
3. **ip_concate_new_file.py**: This file is used to concatenate the data from old train, val and 
test dataset into a label all file.

## data file (created by processing) explanation

