import json
import git
import subprocess
from rework_backtrader.utils.directory_controller import remove_if_existed
from config import FACTOR_RETURN, STOCK_FACTOR_FOLDER, DATA_FOLDER, MKT_CHECK
from utility import all_stock_list, industry_list, combine_factor_list, date_processing
import pandas as pd
import matplotlib.pyplot as plt


INPUT_FACTOR_RETURN = FACTOR_RETURN
INPUT_STOCK_FACTOR_FOLDER = STOCK_FACTOR_FOLDER
daily_file_path = f"{DATA_FOLDER}/live_data/daily"


def analysis(input_file, book_size):
    INPUT_ALPHA_WEIGHT = input_file
    # factor return
    # INPUT_STOCK_FACTOR_FOLDER = input_file
    if MKT_CHECK:
        all_factor_list = combine_factor_list + industry_list + ['index']
    else:
        all_factor_list = combine_factor_list + industry_list
    factor_return_df = pd.read_csv(INPUT_FACTOR_RETURN, index_col=0, parse_dates=[0])
    base_index = factor_return_df.index
    # weight alpha
    weight_alpha = pd.read_csv(INPUT_ALPHA_WEIGHT, index_col=0, parse_dates=[0])
    weight_alpha = weight_alpha.shift(1).dropna(how='all')
    weight_alpha = weight_alpha.merge(pd.DataFrame(index=base_index),
                                        how='right', left_index=True, right_index=True)
    weight_alpha = weight_alpha.fillna(method='ffill').fillna(0)
    weight_alpha = weight_alpha/book_size
    alpha_stock_list = weight_alpha.columns
    # notify missing stock
    missing_stock = []
    for stock in alpha_stock_list:
        if stock not in all_stock_list:
            print('missing', stock)
    # get each factor of each stock
    sub_factor_dic = {}
    for factor in all_factor_list:
        sub_factor_dic[factor] = pd.read_csv(f"{INPUT_STOCK_FACTOR_FOLDER}/{factor}.csv", index_col=0, parse_dates=[0])
        sub_factor_dic[factor] = sub_factor_dic[factor].merge(pd.DataFrame(index=base_index), how='right', on='Date')
        sub_factor_dic[factor].fillna(0, inplace=True)
    # alpha decompose: processing factor
    decompose_df = pd.DataFrame(index=base_index, columns=all_factor_list)
    for factor in all_factor_list:
        decompose_df[factor] = 0
        for stock in alpha_stock_list:
            if stock not in ['VN30', 'VN30F1M', 'VNINDEX']:
                decompose_df[factor] += weight_alpha[stock]*sub_factor_dic[factor][stock]
        decompose_df[factor] = decompose_df[factor]*factor_return_df[factor]
    # print(decompose_df)

    # alpha decompose: processing alpha return
    decompose_df['pctchange'] = 0
    for stock in alpha_stock_list:
        if stock not in ['VN30', 'VN30F1M', 'VNINDEX']:
            daily_df = pd.read_csv(f'{daily_file_path}/{stock}.csv', parse_dates=[0], index_col=0)
            daily_df.index = daily_df.index.tz_localize(None)
            # daily_df = date_processing(daily_df)
            daily_df['return'] = (daily_df['Close'] - daily_df['Close'].shift(1))/daily_df['Close'].shift(1)
            daily_df = daily_df.merge(pd.DataFrame(index=base_index), how='right', right_index=True, left_index=True)
            daily_df['return'].fillna(0, inplace=True)
            decompose_df['pctchange'] += daily_df['return']*weight_alpha[stock]

    # alpha decompose: processing alpha residual
    decompose_df['residual'] = decompose_df['pctchange'] - decompose_df[all_factor_list].sum(axis=1)
    decompose_df.cumsum().plot()
    decompose_df[combine_factor_list + industry_list + ['residual']].cumsum().plot()
    decompose_df[combine_factor_list].cumsum().plot()
    decompose_df[industry_list + ['residual']].cumsum().plot()
    if MKT_CHECK:
        decompose_df[['beta', 'leverage', 'profitability', 'size', 'liquidity', 'value', 'volatility', 'momentum',
                  'growth', 'earning_yield', 'ST_reversal', 'LT_reversal', 'index']].cumsum().plot()
    else:
        decompose_df[['beta', 'leverage', 'profitability', 'size', 'liquidity', 'value', 'volatility', 'momentum',
                      'growth', 'earning_yield', 'ST_reversal', 'LT_reversal']].cumsum().plot()
    plt.show()
    return decompose_df.cumsum().reset_index().to_dict('records')


if __name__ == '__main__':
    app = analysis('/home/vu/Downloads/pos_daily_after_optimize_mosek.csv', 1)
    print(app)