# LightGBM model for DBC
## aims
1. define a metrix to describe shots' DBC
2. find the best strategy to use a dataset with enough DBC value to train a model
performs well on the test set(which is fixed).
3. the test set's DBC should be set more wisely to make this study more meaningful.
4. model evaluation on the whole test set, then evaluate the model on each category
   test set. it's expect that high DBC shots should be predicted with highest accuracy 
like 100%. then the shots with lower DBC value than the high can be predicted with
accuracy like 90%. then we judge a model's performance on a machine or a category 
with the accuracy, the disruption rate and the DBC value.
5. the DBC distribution of train set and test set is different, but the disruption
physics of train set and test set is similar. This should be added by disruption 
consequence.
 
## approaches
### shots
1. calculate the DBC in ip range with bt and p, the DBC value can be uncontinuous,
so we need to
2. train shots chosen:
   1. all the zero DBC shots
   2. the high DBC value shots should be chosen manually according to the disruption
   consequence.
3. choose the test shots through 2 ways:  
    ~~1. random choose: take ip_1-4 as dataset, sklearn train_test_split to get the test set
   and train set randomly by file:dataset_test_compare_shots.py.~~  
    ~~2. choose the shots with the highest DBC value~~
    1. test set should contain many category except the zero DBC shots
   2. train set and test set should have different DBC distribution, but the disruption
   physical meaning should be the same.
   3. the performance of the model should be evaluated by different category shots.
### data processing
### model train and eval
### analysis disruption consequence

## files in this folder
## training log
1. 2024-11-27:  
 model name: lgbm_model.txt  
eval result folder: case1_11_27
2. 2024-12-02:
all shots are processed as train set in file repo folder name: label_train_all
3. 2024-12-03: 
   1.  
       folder name: case_12_03_1,   
       model name: lgbm_model_undis_0.5.txt  
       test set: ip_5 apparte 50% shots from wrong shots  
       train set: ip_1-4 and 50% shots apparted from wrong shots  
        auc:0.89
   
   2. 
       folder name: case_12_03,  
       model name: lgbm_model.txt  
       test set: ip_5 apparte 50% shots from wrong shots  
       train set: ip_1-4 
        auc:0.87
4. 2024-12-04:  
   1.  
       folder name: case_12_04_1,  
       model name: lgbm_model_123.txt  
       test set: ip_4 
       train set: ip_1-3  
        auc:0.86
   2. 
       folder name: case_12_04_2,  
       model name: lgbm_model_random.txt  
       test set: randm_test_shots
       train set: randm_train_shots  
        auc:0.93
   3. 
       folder name: case_12_04_3,  
       model name: lgbm_model_123_4.txt  
       test set: ip_5 apparte 50% shots from wrong shots  
       train set: randm_train_shots
        auc:0.86
   4. 
       folder name: case_12_04_4,  
       model name: lgbm_model_123_5.txt  
       test set: ip_5 apparte 50% shots from wrong shots  
       train set: ip_1-3 
        auc:0.81
   5. 
        folder name: case2_12_24_1
        model name: lgbm_model_12345.txt
        test set: c1-5, 20% of c1,c2,c3,c4, all c5
        train set: 80% of c1,c2,c3,c4
        auc: 0.92
   6.  
       folder name: case_12_04_4,  
       model name: lgbm_model_added_1.txt  
       test set: ip_5 apparte 50% shots from wrong shots  
       train set: ip_1-3 
        auc:0.81

 
