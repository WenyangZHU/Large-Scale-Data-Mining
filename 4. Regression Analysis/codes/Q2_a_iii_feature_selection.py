
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression


# load data from file
data = pd.read_csv('network_backup_dataset.csv')
data.columns = ['week', 'day_of_week_orig', 'start_time','work_flow_orig','file_name_orig','size','duration']

def replace_str_with_int(data, column, insert_pos, truncate_pos=0, map_day=None):
    new_col = []
    for item in data[column]:
        if map_day:
            new_col.append(map_day[item])
        else:
            new_col.append(int(item[truncate_pos:]))
    
    data.insert(insert_pos, column[:len(column) - 5], new_col)
    data.drop(column, 1, inplace = True)       

# 1 encode day of week
map_day = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
replace_str_with_int(data, 'day_of_week_orig', 2, 0, map_day)

# 2 encode work flow
replace_str_with_int(data, 'work_flow_orig', 3, 10)

# 3 encode file name
replace_str_with_int(data, 'file_name_orig', 4, 5)

# extract input and output
input_arr = []
for row in range(len(data)):
    input_arr.append(data.loc[row, 'week':'file_name'].values)

output_arr = data.loc[:, 'size'].values

# standardize input, output
scaler = StandardScaler()
input_arr = scaler.fit_transform(input_arr)
output_arr = scaler.fit_transform(output_arr.reshape(-1, 1))

methods = ['f_regression', 'mutual_info_regression']
def select_features(method, method_str):
    select = SelectKBest(method, k=3).fit(input_arr, output_arr)
    input_arr_new = select.transform(input_arr)
    print ('use '+method_str+':')
    print (input_arr_new.shape)
    selected_features = select.get_support(True)
    print (selected_features)
    return input_arr_new

# 1 Use f_regression to select three most important variables
input_arr_new_f_regression = select_features(f_regression, methods[0])
# 2 Use mutual_info_regression to select three most important variables
input_arr_new_mutual_info_regression = select_features(mutual_info_regression, methods[1])

# train and fit new linear regression models
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def linear_regression_model(input_arr, method_str):
    lr = linear_model.LinearRegression()
    kf = KFold(n_splits=10, shuffle=False)
    best_param = None
    best_rmse = float("inf") 
    fold = [i for i in range(10)]
    train_rmses = []
    test_rmses = []

    for train_index, test_index in kf.split(input_arr):
        train_in = [input_arr[i] for i in train_index]
        train_out = [output_arr[i] for i in train_index]
        test_in = [input_arr[i] for i in test_index]
        test_out = [output_arr[i] for i in test_index]
    
        lr.fit(train_in, train_out)
    
        test_pre = lr.predict(test_in)
        train_pre = lr.predict(train_in)
    
        train_rmse = rmse(train_pre, train_out)
        test_rmse = rmse(test_pre, test_out)
    
        if ((train_rmse + test_rmse) < best_rmse):
            best_param = lr.get_params()

        train_rmses.append(train_rmse)
        test_rmses.append(test_rmse)

    df = pd.DataFrame({
        'fold' : fold,
        'train_rmse' : train_rmses,
        'test_rmse' : test_rmses
    }, columns = ['fold', 'train_rmse', 'test_rmse'])
    # RMSE
    print ('after '+method_str+':')
    display(df)

    lr.set_params(**best_param)
    all_pre = lr.predict(input_arr)
    residuals = np.subtract(output_arr, all_pre)

    # plot fitted values vs true values
    plt.figure(figsize=(15,9))
    plt.scatter(output_arr, all_pre, color='deeppink', edgecolors='k')
    plt.plot([output_arr.min(), output_arr.max()], [output_arr.min(), output_arr.max()], 'k--', lw=4)
    plt.ylabel('Fitted Backup Size (GB)', fontsize = 18)
    plt.xlabel('True Backup Size (GB)', fontsize = 18)
    plt.title('Fitted Values vs True Values (standardized + '+method_str+')', fontsize = 23)
    plt.show()

    # plot residuals vs fitted values
    plt.figure(figsize=(15,9))
    plt.scatter(all_pre, residuals, color='deeppink', edgecolors='k')
    plt.plot([all_pre.min(), all_pre.max()], [0, 0], 'k--', lw=4)
    plt.ylabel('Residuals', fontsize = 18)
    plt.xlabel('Fitted Backup Size (GB)', fontsize = 18)
    plt.title('Residuals vs Fitted Values (standardized '+method_str+' )', fontsize = 23)
    plt.show()
    
# 1 after using f_regression 
linear_regression_model(input_arr_new_f_regression, methods[0])
# 2 after using mutual_info_regression
linear_regression_model(input_arr_new_mutual_info_regression, methods[1])

