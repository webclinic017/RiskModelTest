import numpy as np
import pandas as pd
from config import DATA_FOLDER
from sklearn.linear_model import LinearRegression

industry_dic = {
    'Bán lẻ': ['CTR', 'FRT', 'PNJ', 'MWG', 'DST', 'DGW', 'PET', 'VRE', 'FPT', 'FOX', 'HAX'],
    'Bất động sản': ['SZC', 'LHG', 'ITC', 'NDN', 'BII', 'VIC', 'AMD', 'PDR', 'ASM', 'KDH', 'OGC', 'TDC',
                     'NVL', 'NTL', 'PVL', 'SJS', 'AAV', 'FIR', 'CRE', 'DXG', 'VPI', 'LDG', 'IDC', 'VHM',
                     'TDH', 'NLG', 'TIP', 'CCL', 'ITA', 'HAR', 'KOS', 'NBB', 'SCR', 'KBC', 'HDC', 'HQC',
                     'FLC', 'QCG', 'HDG', 'BCM', 'CEO', 'HAG', 'AGG'],
    'Dịch vụ Tài chính': ['AGR', 'FIT', 'BVH', 'BVS', 'TVB', 'FTS', 'CTS', 'SHS', 'VCI', 'ART', 'APS',
                          'VIG', 'IDJ', 'BCG', 'VND', 'VDS', 'APG', 'VIX', 'TVC', 'MIG', 'BSI', 'PVI',
                          'HCM', 'HHS', 'MSN', 'MBS', 'DRH', 'SSI', 'PGI', 'BIC', 'TCH', 'BMI', 'ORS'],
    'Hàng & Dịch vụ Công nghiệp': ['PAN', 'MHC', 'GMD', 'KLF', 'DL1', 'SCS', 'HAH', 'SAM', 'VTO', 'TIG',
                                   'AAA', 'HII', 'ASG', 'REE', 'VSC', 'ITD', 'DAH', 'PVT', 'VOS', 'HVN',
                                   'VJC', 'GEX', 'SGP'],
    'Hàng cá nhân & Gia dụng': ['TNG', 'MSH', 'SHI', 'GIL', 'TCM', 'PTB', 'ADS', 'TTF', 'HVH', 'VGT'],
    'Hóa chất': ['LAS', 'DPR', 'PHR', 'DCL', 'DPM', 'DGC', 'BFC', 'GVR', 'AMV', 'JVC', 'HAI', 'DCM', 'CSV',
                 'PLP', 'DHG', 'DMC', 'DRC', 'LTG'],
    'Ngân hàng': ['NVB', 'VIB', 'BID', 'STB', 'TCB', 'SSB', 'MSB', 'SHB', 'HDB', 'MBB', 'VCB', 'OCB', 'ACB',
                  'VPB', 'CTG', 'TPB', 'LPB', 'EIB'],
    'Thực phẩm và đồ uống': ['FMC', 'VHC', 'SBT', 'VNM', 'LSS', 'TNA', 'TSC', 'TAR', 'NAF', 'BAF', 'IDI',
                             'DBC', 'CMX', 'SAB', 'KDC', 'SJF', 'HNG', 'ANV', 'QNS'],
    'Tiện ích': ['BWE', 'TDM', 'PVD', 'BSR', 'PLX', 'VSH', 'NT2', 'PVS', 'PPC', 'PVC', 'OIL', 'GAS', 'POW',
                 'PLC', 'GEG', 'PGV', 'SGT', 'ELC', 'VGI'],
    'Tài nguyên Cơ bản': ['HPG', 'DHC', 'VPG', 'KSB', 'TNI', 'NKG', 'VGS', 'POM', 'SMC', 'HSG', 'TLH',
                          'HAP'],
    'Xây dựng và Vật liệu': ['CTI', 'LCG', 'LIG', 'DIG', 'PC1', 'PHC', 'DTD', 'DLG', 'TCD', 'S99', 'DPG',
                             'CII', 'IJC', 'MST', 'TV2', 'BMP', 'VCG', 'ROS', 'EVG', 'VC3', 'BCC', 'TTB',
                             'VCS', 'D2D', 'NTP', 'SCI', 'L14', 'HT1', 'TGG', 'MBG', 'FCN', 'HTN', 'HUT',
                             'HBC', 'VGC', 'HPX', 'CTD', 'GKM']
}

descriptor_list = [
    'beta',  # beta
    'RS', 'HALPHA',  # momentum
    'STRS',
    'LTRS',
    'DASTD', 'CMRA', 'HSIGMA',  # volatility
    'DTOA', 'BLEV', 'MLEV',  # 'leverage',
    'ATO', 'ROE', 'ROA', 'GM',  # profitability
    'LNCAP',  # size
    'STOM', 'STOQ', 'STOA',  # liquidity
    'BTOP',  # value
    'SGRO', 'EGRO',  # growth
    'ETOP'  # earning yield
]

combine_factor_dic = {
    'beta': ['beta'],
    'momentum': ['RS', 'HALPHA'],
    'ST_reversal': ['STRS'],
    'LT_reversal': ['LTRS'],
    'volatility': ['DASTD', 'CMRA', 'HSIGMA'],
    'leverage': ['DTOA', 'BLEV', 'MLEV'],
    'profitability': ['ATO', 'ROE', 'ROA', 'GM'],
    'size': ['LNCAP'],
    'liquidity': ['STOM', 'STOQ', 'STOA'],
    'value': ['BTOP'],
    'growth': ['SGRO', 'EGRO'],
    'earning_yield': ['ETOP']
}

industry_list = list(industry_dic.keys())

stock_to_industry_dic = {}
for industry in industry_dic:
    for stock in industry_dic[industry]:
        stock_to_industry_dic[stock] = industry

all_stock_list = []
for value in industry_dic.values():
    all_stock_list += value

combine_factor_list = list(combine_factor_dic.keys())


def calculate_variance(stock, to_date):
    file_path = f'{DATA_FOLDER}/live_data/daily/'
    return_df_factor = pd.read_csv(f'{file_path}{stock}.csv', index_col=['Date'], parse_dates=['Date'])[
        ['Close']].pct_change().dropna()
    return_df_factor = return_df_factor[:to_date]
    return_df_factor = return_df_factor.iloc[-252:, :]
    try:
        variance = np.var(return_df_factor['Close'])
        return variance
    except:
        return -1


def calculate_inverse_weight_var(stock_list, to_date):
    variance_list = []
    for stock in stock_list:
        my_var = calculate_variance(stock, to_date)
        if my_var > 0:
            variance_list.append(1 / my_var)
        else:
            pass
    return variance_list


def date_processing(dataframe):
    dataframe['Date'] = [i.split(' ')[0][:] for i in dataframe['Date']]
    dataframe['Date'] = dataframe['Date'].apply(pd.Timestamp)  # very important: convert string to date
    dataframe.set_index(['Date'], inplace=True)
    dataframe.index = dataframe.index.astype(str)
    return dataframe


def industry_beta(stock_return, industry_return, length, T=int(252), intercept=True):
    industry_beta_vec = np.zeros(length)
    industry_beta_vec[:] = np.nan
    for t in range(T, length+1):
        try:
            y = stock_return[(t - T):t]
            X = industry_return[(t - T):t].reshape(-1, 1)
            reg = LinearRegression(fit_intercept=intercept).fit(X, y)
            industry_beta_vec[t-1] = reg.coef_
        except:
            pass
    return industry_beta_vec


def descriptor_weighted_cap_mean(descriptor_df, mkcap_df, sum_mkt_cap_vec):
    length = len(descriptor_df)
    weighted_cap_mean = np.zeros(length)
    stocks = descriptor_df.columns
    for stock in stocks:
        weighted_cap_mean += descriptor_df[stock].fillna(0) * mkcap_df[stock].fillna(0)
    weighted_cap_mean = weighted_cap_mean/sum_mkt_cap_vec
    return weighted_cap_mean
