import glob
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from utility import date_processing, calculate_variance, calculate_inverse_weight_var, combine_factor_list
from vq_tools.utils.folder_utils import create_folder
from sklearn.linear_model import LinearRegression
from utility import all_stock_list, industry_dic, industry_list
from config import LOADING_FACTOR_FOLDER, FACTOR_RETURN, STOCK_FACTOR_FOLDER, MKT_CHECK, INDUSTRY_NORM_CHECK
from utility import all_stock_list, industry_dic, industry_list

OUTPUT_FACTOR_RETURN = FACTOR_RETURN
OUTPUT_STOCK_FACTOR_FOLDER = STOCK_FACTOR_FOLDER
INPUT_FOLDER = f'{LOADING_FACTOR_FOLDER}/*.csv'

data_factor_folder = sorted(glob.glob(INPUT_FOLDER))  # sort datetime
date_list = [os.path.splitext(os.path.basename(sub_path))[0] for sub_path in data_factor_folder]
start_date = '2019-01-03'
start_index = date_list.index(f'{start_date}')
print(start_index)

if MKT_CHECK:
    all_factor = combine_factor_list + industry_list + ['index']
else:
    all_factor = combine_factor_list + industry_list

date_index = date_list[start_index:]
factor_return_df = pd.DataFrame(index=date_index, columns=all_factor)


sub_factor_dic = {}
for factor in all_factor:
    sub_factor_dic[factor] = pd.DataFrame(index=date_index, columns=all_stock_list)
    sub_factor_dic[factor].index.name = 'Date'

for i in range(start_index, len(data_factor_folder)):
    loading_factor_minus_0 = pd.read_csv(data_factor_folder[i], index_col=0)

    loading_factor = loading_factor_minus_0
    current_date = os.path.splitext(os.path.basename(data_factor_folder[i]))[0]

    # save for later use of analysis
    for factor in all_factor:
        for stock in loading_factor_minus_0.index:
            sub_factor_dic[factor][stock][current_date] = loading_factor_minus_0[factor][stock]

    loading_factor.dropna(inplace=True)
    loading_factor.replace([np.inf, -np.inf], 0, inplace=True)
    try:
        print(current_date)
        # cross-sectional
        stock_list = loading_factor_minus_0.index
        weight_WLS = calculate_inverse_weight_var(stock_list, current_date)

        y = loading_factor['pctchange'].values.reshape(-1, 1)
        # X = loading_factor[all_factor].replace([np.inf, -np.inf], 0, inplace=True).values
        X = loading_factor[all_factor].values

        mod_wls = sm.WLS(y, X, weights=weight_WLS)
        res_wls = mod_wls.fit()
        factor_return_df.loc[current_date] = res_wls.params

        # regressor = LinearRegression(fit_intercept=False)
        # regressor.fit(X, y)
        # factor_return_df.loc[current_date] = regressor.coef_

    except:
        print(f'not {current_date}')
        pass


# export factor return
factor_return_df.dropna(inplace=True)
factor_return_df.index.name = 'Date'
factor_return_df.to_csv(OUTPUT_FACTOR_RETURN)

# export factor of all stock everyday
create_folder(OUTPUT_STOCK_FACTOR_FOLDER)
for factor in all_factor:
    sub_factor_dic[factor].to_csv(f"{OUTPUT_STOCK_FACTOR_FOLDER}/{factor}.csv")

print(factor_return_df)
