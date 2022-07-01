import pandas as pd
import glob
import os
from factor_processing import Factor
from vq_tools.utils.folder_utils import create_folder
import numpy as np
from utility import all_stock_list, industry_dic, industry_list, \
    stock_to_industry_dic, \
    descriptor_list, combine_factor_list, industry_beta, descriptor_weighted_cap_mean
from config import LOADING_FACTOR_FOLDER, DATA_FOLDER, MKT_CHECK, INDUSTRY_NORM_CHECK


# OUTPUT FOLDER
OUTPUT_LOADING_FACTOR_FOLDER = LOADING_FACTOR_FOLDER
# INPUT FOLDER
mkt_price_cap_path = f'{DATA_FOLDER}/VIETSTOCK_STS/sts_price'
financial_statement_path = f'{DATA_FOLDER}/FDMT/'
daily_path = f'{DATA_FOLDER}/live_data/daily/'

# extract longest datetime length
base_stock = 'HSG'
base_df = pd.read_csv(f'{mkt_price_cap_path}/{base_stock}.csv', index_col=0)
index_df = pd.DataFrame(index=base_df.index.astype(str))
length = len(index_df)

if INDUSTRY_NORM_CHECK:
    stock_descriptor_dic = {}
    stock_factor_dic = {}
    # scale by industry
    for industry in industry_dic.keys():
        print(industry)
        # save data of each descriptor of each industry (temp_dic_1)
        temp_dic_1 = {}
        for descriptor in descriptor_list:
            temp_dic_1[descriptor] = {}
        temp_dic_1['industry_beta'] = {}
        mkt_cap_dic = {}
        # save mean and std of each descriptor of each industry (tem_dic_2)
        temp_dic_2 = {}
        industry_return = np.zeros(length)
        sum_mkt_cap = np.zeros(length)
        for stock in industry_dic[industry]:
            print(stock)
            df_fs = pd.read_csv(f'{financial_statement_path}/{stock}.csv', index_col=0, parse_dates=[0])
            df_mkt_cap = pd.read_csv(f'{mkt_price_cap_path}/{stock}.csv', index_col=0, parse_dates=[0])
            df_daily = pd.read_csv(f'{daily_path}/{stock}.csv')
            df_vnindex = pd.read_csv(f'{daily_path}/VNINDEX.csv')
            factor = Factor(df_mkt_cap, df_fs, df_daily, df_vnindex)
            descriptor_df = pd.DataFrame(index=factor.join_df.index.astype(str))
            # beta
            descriptor_df['beta'], descriptor_df['HSIGMA'], _ = factor.beta(T=252)
            # reversal
            descriptor_df['LTRS'] = factor.LTRS()
            descriptor_df['STRS'] = factor.STRS()
            # momentum
            descriptor_df['RS'] = factor.RS()
            descriptor_df['HALPHA'] = factor.HALPHA()
            # volatility
            descriptor_df['DASTD'] = factor.DASTD()
            descriptor_df['CMRA'] = factor.CMRA()
            # leverage
            descriptor_df['MLEV'] = factor.MLEV()
            descriptor_df['BLEV'] = factor.BLEV()
            descriptor_df['DTOA'] = factor.DTOA()
            # profitability
            descriptor_df['ATO'] = factor.ATO()
            descriptor_df['GM'] = factor.GM()
            descriptor_df['ROA'] = factor.ROA()
            descriptor_df['ROE'] = factor.ROE()
            # size
            descriptor_df['LNCAP'] = factor.LNCAP()
            # value
            descriptor_df['BTOP'] = factor.BTOP()
            # liquidity
            descriptor_df['STOM'] = factor.STOM()
            descriptor_df['STOQ'] = factor.STOQ(descriptor_df['STOM'])
            descriptor_df['STOA'] = factor.STOA(descriptor_df['STOM'])
            # growth
            descriptor_df['SGRO'] = factor.SGRO()
            descriptor_df['EGRO'] = factor.ERGO()
            # earning yield
            descriptor_df['ETOP'] = factor.ETOP()
            descriptor_df['pctchange'] = np.array(factor.join_df['pctchange'])
            descriptor_df['MarketCap'] = np.array(factor.join_df['MarketCap'])  # save for later use
            descriptor_df = index_df.merge(descriptor_df, how='left', on='Date')
            # save to main dictionary
            stock_descriptor_dic[stock] = descriptor_df
            mkt_cap_dic[stock] = descriptor_df['MarketCap'].fillna(0)*1e6  # save for later use

            # save to temp dic 1 to calculate mean and variance
            for descriptor in descriptor_list:
                temp_dic_1[descriptor][stock] = descriptor_df[descriptor]

            industry_return += np.array(descriptor_df['MarketCap'].fillna(0)*1e6) * np.array(descriptor_df['pctchange'].fillna(0))
            sum_mkt_cap += np.array(descriptor_df['MarketCap'].fillna(0)*1e6)

        industry_return = industry_return/sum_mkt_cap

        # calculate industry_beta
        for stock in industry_dic[industry]:
            descriptor_df = stock_descriptor_dic[stock]
            descriptor_df['industry_beta'] = industry_beta(descriptor_df['pctchange'].ffill(), industry_return, length,
                                                           T=int(252), intercept=True)
            stock_descriptor_dic[stock] = descriptor_df
            temp_dic_1['industry_beta'][stock] = descriptor_df['industry_beta']

        # calculate mean and variance of each descriptor
        mkt_cap_df = pd.DataFrame(mkt_cap_dic)
        new_descriptor_list = descriptor_list + ['industry_beta']
        for descriptor in new_descriptor_list:
            single_descriptor_df = pd.DataFrame(temp_dic_1[descriptor])
            stats_df = pd.DataFrame(index=single_descriptor_df.index)
            stats_df['mean'] = descriptor_weighted_cap_mean(single_descriptor_df, mkt_cap_df, sum_mkt_cap)
            # single_descriptor_df['mean'] = single_descriptor_df.mean(axis=1, skipna=True)
            stats_df['std'] = single_descriptor_df.std(axis=1, skipna=True)
            if np.sum(stats_df['std']) == 0:
                print(descriptor, industry)
            temp_dic_2[descriptor] = (stats_df['mean'], stats_df['std'])

        # normalize descriptor by industry
        for stock in industry_dic[industry]:
            # normalize descriptor
            current_stock_descriptor_df = stock_descriptor_dic[stock]
            for descriptor in new_descriptor_list:
                m, std = temp_dic_2[descriptor]
                if industry == 'Ngân hàng' and descriptor in ['MLEV', 'BLEV']:
                    current_stock_descriptor_df[descriptor] = 0
                else:
                    current_stock_descriptor_df[descriptor] = (current_stock_descriptor_df[descriptor] - m) / std
            # stock_descriptor_dic[stock] = current_stock_df

            # combine descriptor to factor
            current_stock_factor_df = pd.DataFrame(index=current_stock_descriptor_df.index)
            current_stock_factor_df['pctchange'] = current_stock_descriptor_df['pctchange']
            current_stock_factor_df['beta'] = 0.5 * current_stock_descriptor_df['beta'] + \
                                              0.5 * current_stock_descriptor_df['industry_beta']
            current_stock_factor_df['momentum'] = 0.75 * current_stock_descriptor_df['RS'] + \
                                                  0.25 * current_stock_descriptor_df['HALPHA']
            current_stock_factor_df['ST_reversal'] = current_stock_descriptor_df['STRS']
            current_stock_factor_df['LT_reversal'] = current_stock_descriptor_df['LTRS']
            current_stock_factor_df['volatility'] = 0.50 * current_stock_descriptor_df['DASTD'] + \
                                                    0.3 * current_stock_descriptor_df['CMRA'] + \
                                                    0.2 * current_stock_descriptor_df['HSIGMA']
            current_stock_factor_df['leverage'] = 0.25 * current_stock_descriptor_df['MLEV'] + \
                                                  0.5 * current_stock_descriptor_df['DTOA'] + \
                                                  0.25 * current_stock_descriptor_df['BLEV']
            current_stock_factor_df['profitability'] = 0.1 * current_stock_descriptor_df['ATO'] + \
                                                       0.5 * current_stock_descriptor_df['GM'] + \
                                                       0.4 * current_stock_descriptor_df['ROA']
            current_stock_factor_df['size'] = current_stock_descriptor_df['LNCAP']
            current_stock_factor_df['value'] = current_stock_descriptor_df['BTOP']
            current_stock_factor_df['liquidity'] = 0.4 * current_stock_descriptor_df['STOM'] + \
                                                   0.3 * current_stock_descriptor_df['STOQ'] + \
                                                   0.3 * current_stock_descriptor_df['STOA']
            current_stock_factor_df['growth'] = 0.67 * current_stock_descriptor_df['SGRO'] + \
                                                0.33 * current_stock_descriptor_df['EGRO']
            current_stock_factor_df['earning_yield'] = current_stock_descriptor_df['ETOP']
            stock_factor_dic[stock] = current_stock_factor_df

    final_df = {}
    for date in index_df.index:
        final_df[date] = pd.DataFrame(columns=['pctchange'] + combine_factor_list + industry_list,
                                      index=all_stock_list)

    new_factor_list = ['pctchange'] + combine_factor_list

    for date in final_df.keys():
        for factor in new_factor_list:
            for stock in all_stock_list:
                final_df[date][factor][stock] = stock_factor_dic[stock][factor][date]

        # fill dummy for industry variable:
        for stock in all_stock_list:
            for industry in industry_list:
                if stock_to_industry_dic[stock] == industry:
                    final_df[date][industry][stock] = 1
                else:
                    final_df[date][industry][stock] = 0
            if MKT_CHECK:
                final_df[date]['index'][stock] = 1

    create_folder(OUTPUT_LOADING_FACTOR_FOLDER)
    for date in final_df.keys():
        final_df[date].to_csv(f"{OUTPUT_LOADING_FACTOR_FOLDER}/{date}.csv")
else:
    stock_descriptor_dic = {}
    stock_factor_dic = {}
    # scale by industry
    # save data of each descriptor of each industry (temp_dic_1)
    temp_dic_1 = {}
    for descriptor in descriptor_list:
        temp_dic_1[descriptor] = {}

    mkt_cap_dic = {}
    # save mean and std of each descriptor of each industry (tem_dic_2)
    temp_dic_2 = {}
    sum_mkt_cap = np.zeros(length)
    for stock in all_stock_list:
        print(stock)
        df_fs = pd.read_csv(f'{financial_statement_path}/{stock}.csv', index_col=0, parse_dates=[0])
        df_mkt_cap = pd.read_csv(f'{mkt_price_cap_path}/{stock}.csv', index_col=0, parse_dates=[0])
        df_daily = pd.read_csv(f'{daily_path}/{stock}.csv')
        df_vnindex = pd.read_csv(f'{daily_path}/VNINDEX.csv')
        factor = Factor(df_mkt_cap, df_fs, df_daily, df_vnindex)
        descriptor_df = pd.DataFrame(index=factor.join_df.index.astype(str))
        # beta
        descriptor_df['beta'], descriptor_df['HSIGMA'], _ = factor.beta(T=252)
        # reversal
        descriptor_df['LTRS'] = factor.LTRS()
        descriptor_df['STRS'] = factor.STRS()
        # momentum
        descriptor_df['RS'] = factor.RS()
        descriptor_df['HALPHA'] = factor.HALPHA()
        # volatility
        descriptor_df['DASTD'] = factor.DASTD()
        descriptor_df['CMRA'] = factor.CMRA()
        # leverage
        descriptor_df['MLEV'] = factor.MLEV()
        descriptor_df['BLEV'] = factor.BLEV()
        descriptor_df['DTOA'] = factor.DTOA()
        # profitability
        descriptor_df['ATO'] = factor.ATO()
        descriptor_df['GM'] = factor.GM()
        descriptor_df['ROA'] = factor.ROA()
        descriptor_df['ROE'] = factor.ROE()
        # size
        descriptor_df['LNCAP'] = factor.LNCAP()
        # value
        descriptor_df['BTOP'] = factor.BTOP()
        # liquidity
        descriptor_df['STOM'] = factor.STOM()
        descriptor_df['STOQ'] = factor.STOQ(descriptor_df['STOM'])
        descriptor_df['STOA'] = factor.STOA(descriptor_df['STOM'])
        # growth
        descriptor_df['SGRO'] = factor.SGRO()
        descriptor_df['EGRO'] = factor.ERGO()
        # earning yield
        descriptor_df['ETOP'] = factor.ETOP()
        descriptor_df['pctchange'] = np.array(factor.join_df['pctchange'])
        descriptor_df['MarketCap'] = np.array(factor.join_df['MarketCap'])  # save for later use
        descriptor_df = index_df.merge(descriptor_df, how='left', on='Date')
        # save to main dictionary
        stock_descriptor_dic[stock] = descriptor_df
        mkt_cap_dic[stock] = descriptor_df['MarketCap'].fillna(0)*1e6  # save for later use
        sum_mkt_cap += np.array(descriptor_df['MarketCap'].fillna(0) * 1e6)
        # save to temp dic 1 to calculate mean and variance
        for descriptor in descriptor_list:
            temp_dic_1[descriptor][stock] = descriptor_df[descriptor]

    # calculate mean and variance of each descriptor
    mkt_cap_df = pd.DataFrame(mkt_cap_dic)
    for descriptor in descriptor_list:
        single_descriptor_df = pd.DataFrame(temp_dic_1[descriptor])
        stats_df = pd.DataFrame(index=single_descriptor_df.index)
        stats_df['mean'] = descriptor_weighted_cap_mean(single_descriptor_df, mkt_cap_df, sum_mkt_cap)
        # single_descriptor_df['mean'] = single_descriptor_df.mean(axis=1, skipna=True)
        stats_df['std'] = single_descriptor_df.std(axis=1, skipna=True)
        if np.sum(stats_df['std']) == 0:
            print(descriptor)
        temp_dic_2[descriptor] = (stats_df['mean'], stats_df['std'])

    # normalize descriptor by industry
    for stock in all_stock_list:
        # normalize descriptor
        current_stock_descriptor_df = stock_descriptor_dic[stock]
        for descriptor in descriptor_list:
            m, std = temp_dic_2[descriptor]
            if stock_to_industry_dic[stock] == 'Ngân hàng' and descriptor in ['MLEV', 'BLEV']:
                current_stock_descriptor_df[descriptor] = 0
            else:
                current_stock_descriptor_df[descriptor] = (current_stock_descriptor_df[descriptor] - m) / std
        # stock_descriptor_dic[stock] = current_stock_df

        # combine descriptor to factor
        current_stock_factor_df = pd.DataFrame(index=current_stock_descriptor_df.index)
        current_stock_factor_df['pctchange'] = current_stock_descriptor_df['pctchange']
        current_stock_factor_df['beta'] = current_stock_descriptor_df['beta']
        current_stock_factor_df['momentum'] = 0.75 * current_stock_descriptor_df['RS'] + \
                                              0.25 * current_stock_descriptor_df['HALPHA']
        current_stock_factor_df['ST_reversal'] = current_stock_descriptor_df['STRS']
        current_stock_factor_df['LT_reversal'] = current_stock_descriptor_df['LTRS']
        current_stock_factor_df['volatility'] = 0.50 * current_stock_descriptor_df['DASTD'] + \
                                                0.3 * current_stock_descriptor_df['CMRA'] + \
                                                0.2 * current_stock_descriptor_df['HSIGMA']
        current_stock_factor_df['leverage'] = 0.25 * current_stock_descriptor_df['MLEV'] + \
                                              0.5 * current_stock_descriptor_df['DTOA'] + \
                                              0.25 * current_stock_descriptor_df['BLEV']
        current_stock_factor_df['profitability'] = 0.1 * current_stock_descriptor_df['ATO'] + \
                                                   0.5 * current_stock_descriptor_df['GM'] + \
                                                   0.4 * current_stock_descriptor_df['ROA']
        current_stock_factor_df['size'] = current_stock_descriptor_df['LNCAP']
        current_stock_factor_df['value'] = current_stock_descriptor_df['BTOP']
        current_stock_factor_df['liquidity'] = 0.4 * current_stock_descriptor_df['STOM'] + \
                                               0.3 * current_stock_descriptor_df['STOQ'] + \
                                               0.3 * current_stock_descriptor_df['STOA']
        current_stock_factor_df['growth'] = 0.67 * current_stock_descriptor_df['SGRO'] + \
                                            0.33 * current_stock_descriptor_df['EGRO']
        current_stock_factor_df['earning_yield'] = current_stock_descriptor_df['ETOP']
        stock_factor_dic[stock] = current_stock_factor_df

    final_df = {}
    for date in index_df.index:
        final_df[date] = pd.DataFrame(columns=['pctchange'] + combine_factor_list + industry_list,
                                      index=all_stock_list)

    new_factor_list = ['pctchange'] + combine_factor_list

    for date in final_df.keys():
        for factor in new_factor_list:
            for stock in all_stock_list:
                final_df[date][factor][stock] = stock_factor_dic[stock][factor][date]

        # fill dummy for industry variable:
        for stock in all_stock_list:
            for industry in industry_list:
                if stock_to_industry_dic[stock] == industry:
                    final_df[date][industry][stock] = 1
                else:
                    final_df[date][industry][stock] = 0
            if MKT_CHECK:
                final_df[date]['index'][stock] = 1

    create_folder(OUTPUT_LOADING_FACTOR_FOLDER)
    for date in final_df.keys():
        final_df[date].to_csv(f"{OUTPUT_LOADING_FACTOR_FOLDER}/{date}.csv")
