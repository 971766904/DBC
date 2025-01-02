# data process and dataset shots change 
## files explain
1. split_test_shots.py    
   This file is to split the c1-4 shots in to train and test set. The test set 
    should contain all these categories
2. choose_shots_from5_2_train.py  
    This file is to choose part of c5 shots added to train set. Save the shots 
    number as npy file
3. cate_file.py  
   This file is to help sort the dataset to vary category by display the scatter
    figs or other kind of figs
4. disruption_conse_show.py  
    This file is to show the shot's disruption consequence, then we'll sort these 
    shots according to the disruption consequence to train set. This will make the 
    train set and  the test set have similar disruption consequence distribution or
    similar disruption physics information