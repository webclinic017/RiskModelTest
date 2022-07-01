import copy
import csv
import operator
import numpy as np
import pandas as pd
import glob, sys, os
from utility import date_processing
from sklearn.linear_model import LinearRegression
import math

class Factor:
    def __init__(self, mkt_cap_df, fs_df, daily_df, vnindex_df):
        self.mkt_cap_df = mkt_cap_df
        self.fs_df = fs_df
        self.join_df = self.mkt_cap_df.merge(self.fs_df, how='left', on='Date')
        self.join_df.index = self.join_df.index.astype(str)

        self.daily_df = date_processing(daily_df)
        self.vnindex_df = date_processing(vnindex_df)
        self.vnindex_df['rm'] = (self.vnindex_df['Close'] - self.vnindex_df['Close'].shift(1))/self.vnindex_df['Close'].shift(1)

        self.join_df = self.join_df.merge(self.daily_df, how='left', on='Date')
        self.join_df = self.join_df.merge(self.vnindex_df['rm'], how='left', on='Date')
        self.join_df['pctchange'] = (self.join_df['Close'] - self.join_df['Close'].shift(1))/self.join_df['Close'].shift(1)
        self.join_df.ffill(axis=0, inplace=True)
        self.join_df.bfill(axis=0, inplace=True)
        self.join_df.fillna(0, inplace=True) # only apply for first row since pctchange = 0
        self.date = list(self.join_df.index)
        self.length = len(self.join_df)

    def halflife(self, half_life, length):
        t = np.arange(length)
        w = 2 ** (t / half_life) / sum(2 ** (t / half_life)) # Half line hien h la theo ngay
        return w

    def beta(self, T=int(252), intercept=True):
        beta = np.zeros(self.length)
        beta[:] = np.nan
        residual_vol = np.zeros(self.length)
        residual_vol[:] = np.nan
        alpha = np.zeros(self.length)
        alpha[:] = np.nan
        w = self.halflife(half_life=126, length=T)
        for t in range(T, self.length+1):
            y = (self.join_df.pctchange[(t - T):t]).values
            X = self.join_df.rm[(t - T):t].values.reshape(-1, 1)
            reg = LinearRegression(fit_intercept=intercept).fit(X, y)
            beta[t-1] = reg.coef_
            residual = y - reg.predict(X)
            residual_vol[t-1] = np.std(residual*w)
            alpha[t-1] = reg.intercept_
        return beta, residual_vol, alpha

    ################################################################### REVERSAL
    # Long term reversal (1 year)
    # def LTRS(self, T=int(252), L=int(21/2), halflife=int(126/2)):
    #     ltrs = np.zeros(self.length)
    #     for t in range(T, self.length + 1):
    #         rt = - self.join_df.Close[t - T - 1] / self.join_df.Close[t - 5]
    #         # note: today index is t-1
    #         ltrs[t - 1] = rt
    #     return ltrs

    def LTRS(self, T=252, half_life=int(252/4), L=int(252/4), window=11):
        w = self.halflife(half_life, length=T)
        rs = np.zeros(self.length)
        rs[:] = np.nan
        rstr = np.zeros(self.length)
        rstr[:] = np.nan
        for t in range(T, self.length+1):
            rt = self.join_df.pctchange[(t-T): t]
            rs[t-1] = - sum(rt * w)
        for t in range(T + L + window, self.length + 1):
            rstr[t-1] = np.nanmean(rs[(t-L-window): (t-L)])
        return rstr




    # Short term reversal (1 month)
    # def STRS(self,  T=21, halflife=10, L=5):
    #     strs = np.zeros(self.length)
    #     w = self.halflife(half_life=halflife, length=T)
    #     for t in range(T + L, self.length + 1):
    #         rt = self.join_df.pctchange[(t - T - L): (t - L)]
    #         strs[t - 1] = sum(np.log(1 + rt) * w)
    #     return strs


    # core
    # def STRS(self, T=21):
    #     strs = np.zeros(self.length)
    #     for t in range(T, self.length+1):
    #         rt = - self.join_df.Close[t-T-1] / self.join_df.Close[t-5]
    #         # note: today index is t-1
    #         strs[t-1] = rt
    #     return strs

    def STRS(self, T=63, half_life=10, L=1, window=3):
        w = self.halflife(half_life, length=T)
        rs = np.zeros(self.length)
        rs[:] = np.nan
        rstr = np.zeros(self.length)
        rstr[:] = np.nan
        for t in range(T, self.length+1):
            rt = self.join_df.pctchange[(t-T): t]
            rs[t-1] = - sum(rt * w)
        for t in range(T + L + window, self.length + 1):
            rstr[t-1] = np.nanmean(rs[(t-L-window): (t-L)])
        return rstr

    # 1 day reversal
    def D1RS(self):
        return np.array(np.log(1+self.join_df.pctchange))

    ################################################################### MOMENTUM
    # Relative Strength (momentum)
    def RS(self, T=252, half_life=int(252/4), L=3):
        w = self.halflife(half_life, length=T)
        rs = np.zeros(self.length)
        rs[:] = np.nan
        rstr = np.zeros(self.length)
        rstr[:] = np.nan
        for t in range(T, self.length+1):
            rt = self.join_df.pctchange[(t-T): t]
            rs[t-1] = sum(rt * w)
        for t in range(T + L + L, self.length + 1):
            rstr[t-1] = np.nanmean(rs[(t-L-L): (t-L)])
        return rstr

    def HALPHA(self, T=int(252), L=11):
        halpha = np.zeros(self.length)
        halpha[:] = np.nan
        _, _, alpha = self.beta(T)
        for t in range(T + L + L, self.length + 1):
            halpha[t-1] = np.nanmean(alpha[(t-L-L): (t-L)])
        return halpha

    ################################################################### VOLATILITY
    # Daily Standard Deviation
    # def DASTD(self, half_life=15, T=65):
    #     w = self.halflife(half_life, length=T)
    #     dastd = np.zeros(self.length)
    #     for t in range(T, self.length+1):
    #         rt = self.join_df.pctchange[(t-T):t]
    #         dastd[t-1] = np.sqrt(sum(w * (rt**2))*T)
    #     return dastd

    def DASTD(self, T=252):
        # w = self.halflife(half_life, length=T)
        dastd = np.zeros(self.length)
        dastd[:] = np.nan
        w = self.halflife(half_life=42, length=T)
        for t in range(T, self.length+1):
            rt = self.join_df.pctchange[(t-T):t]*w
            dastd[t-1] = np.nanstd(rt)
        return dastd

    # Cumulative Range
    def CMRA(self, period=12):
        cmra = np.zeros(self.length)
        for k in np.arange(period * 21, self.length+1):
            r = []
            for i in range(period):
                r_month = np.prod(self.join_df.pctchange[k - (i + 1) * 21:k - i * 21] + 1) - 1
                r.append(np.log(1 + r_month))
            Z = np.cumsum(r)
            cmra[k-1] = max(Z) - min(Z)
        return cmra

    ################################################################### LEVERAGE
    # Market Leverage
    def MLEV(self):
        LD = np.abs(self.join_df['no dai han 330'])
        ME = self.join_df['MarketCap']*1e6
        ME.replace(to_replace=0, method='ffill', inplace=True)
        PE = self.join_df['b. co phieu uu dai 1b']
        mlev = (ME+PE+LD)/ME
        return np.array(mlev)

    # Book Leverage
    def BLEV(self):
        LD = np.abs(self.join_df['no dai han 330'])
        PE = self.join_df['b. co phieu uu dai 1b']
        BE = np.abs(self.join_df['tong no phai tra va von chu so huu']) - np.abs(self.join_df['tong no phai tra'])
        if sum(BE) == 0:
            BE = np.abs(self.join_df['von chu so huu (400=410420) 400'])
        blev = (BE + PE + LD) / BE
        return np.array(blev)

    # Debt to Asset ratio
    def DTOA(self):
        TD = np.abs(self.join_df['tong no phai tra'])
        TA = np.abs(self.join_df['tong no phai tra va von chu so huu'])
        if sum(TA) == 0:
            TD = np.abs(self.join_df['no phai tra (300=310340) 300'])
            TA = np.abs(TD) + np.abs(self.join_df['von chu so huu (400=410420) 400'])
        dtoa = TD/TA
        return np.array(dtoa)

    ################################################################### PROFITABILITY
    # Asset Turnover
    def ATO(self):
        # for bank
        sales = self.join_df['thu nhap lai thuan'] # ngan hang
        if sum(sales) == 0:
            sales = self.join_df['doanh thu thuan ve ban hang va cung cap dich vu 10'] # san xuat
            if sum(sales) == 0:
                sales = self.join_df['cong doanh thu hoat dong (01>11) 20'] # chung khoan

        TA = np.abs(self.join_df['tong no phai tra va von chu so huu'])
        if sum(TA) == 0:
            TD = np.abs(self.join_df['no phai tra (300=310340) 300'])
            TA = np.abs(TD) + np.abs(self.join_df['von chu so huu (400=410420) 400'])

        ATO = sales/TA
        return np.array(ATO)

    # Gross Margin
    def GM(self):
        # san xuat
        sales = self.join_df['doanh thu thuan ve ban hang va cung cap dich vu 10']
        COGS = abs(self.join_df['gia von hang ban 11'])
        gross_profit = sales - COGS

        if sum(sales) == 0:
            # for bank
            gross_profit = self.join_df['thu nhap lai thuan']
            sales = self.join_df['thu nhap lai va cac khoan thu nhap tuong tu 24']

            if sum(sales) == 0:
                # for security
                sales = self.join_df['cong doanh thu hoat dong (01>11) 20']
                COGS = abs(self.join_df['cong chi phi hoat dong (21>33) 40'])
                gross_profit = sales - COGS

        gm = gross_profit/sales
        return np.array(gm)

    # Return on Asset
    def ROA(self):
        earning = self.join_df['loi nhuan sau thue (xixii)']
        TA = np.abs(self.join_df['tong no phai tra va von chu so huu'])
        if sum(TA) == 0:
            # for bank
            TD = np.abs(self.join_df['no phai tra (300=310340) 300'])
            TA = np.abs(TD) + np.abs(self.join_df['von chu so huu (400=410420) 400'])
        roa = earning/TA
        return np.array(roa)

    def ROE(self):
        earning = self.join_df['loi nhuan sau thue (xixii)']
        BE = np.abs(self.join_df['tong no phai tra va von chu so huu']) - np.abs(self.join_df['tong no phai tra'])
        if sum(BE) == 0:
            # for bank
            BE = np.abs(self.join_df['von chu so huu (400=410420) 400'])
        roa = earning / BE
        return np.array(roa)

    ################################################################### SIZE
    # Log of MarketCap
    def LNCAP(self):
        mktcap = self.join_df['MarketCap']*1e6
        mktcap.replace(to_replace=0, method='ffill', inplace=True)
        return np.array(np.log(mktcap))

    ################################################################### VALUE
    # Book to Price Ratio
    def BTOP(self):
        BE = self.join_df['von chu so huu (400=410420) 400']
        if sum(BE) == 0:
            # for bank
            BE = np.abs(self.join_df['tong no phai tra va von chu so huu']) - np.abs(self.join_df['tong no phai tra'])
        ME = self.join_df['MarketCap']*1e6
        ME.replace(to_replace=0, method='ffill', inplace=True)
        btop = BE/ME
        return np.array(btop)

    # Sale to Price Ratio
    def STOP(self):
        sales = self.join_df['doanh thu thuan ve ban hang va cung cap dich vu 10']
        if sum(sales) == 0:
            # for bank
            sales = self.join_df['thu nhap lai thuan'] # sua thanh thu nhap lai thuan (1-2)
            if sum(sales) == 0:
                # for security
                sales = self.join_df['cong doanh thu hoat dong (01>11) 20']
        ME = self.join_df['MarketCap']*1e6
        ME.replace(to_replace=0, method='ffill', inplace=True)
        stop = sales/ME
        return np.array(stop)

    ################################################################### LIQUIDITY
    # Monthly share turnover
    def STOM(self, T=21):
        stom = np.zeros(self.length)
        stom[:] = np.nan
        mkt_cap = self.join_df['MarketCap']*1e6
        mkt_cap.replace(to_replace=0, method='ffill', inplace=True)
        num_share = mkt_cap/self.join_df['ClosePrice']
        for t in range(T, self.length+1):
            turnover = np.nansum(self.join_df.TotalVol[(t-T):t]/num_share[(t-T):t])
            if math.isnan(turnover):
                stom[t-1] = 0
            else:
                stom[t-1] = np.log(turnover)
        return stom

    # Quarterly share turnover
    def STOQ(self, stom, t=21, T=3):
        stoq = np.zeros(self.length)
        stoq[:] = np.nan
        for k in np.arange(T*t, self.length+1):
            idx = (k-1) - np.arange(T)*t
            stoq[k-1] = np.log(np.nanmean(np.exp(stom.iloc[idx])))
        return stoq

    # Annual share turnover
    def STOA(self, stom, t=21, T=12):
        stoa = np.zeros(self.length)
        stoa[:] = np.nan
        for k in np.arange(T*t, self.length):
            idx = k-1 - np.arange(T)*t
            stoa[k-1] = np.log(np.nanmean(np.exp(stom.iloc[idx])))
        return stoa

    ################################################################### GROWTH
    # Sales growth (trailing 2 years)
    def SGRO(self, T=int(252*2)):
        sgro = np.zeros(self.length)
        num_share = self.join_df['MarketCap'] * 1e6 / self.join_df['ClosePrice']
        num_share.replace(to_replace=0, method='ffill', inplace=True)
        sales = self.join_df['doanh thu thuan ve ban hang va cung cap dich vu 10']
        if sum(sales) == 0:
            # for bank
            sales = self.join_df['thu nhap lai thuan']  # sua thanh thu nhap lai thuan (1-2)
            if sum(sales) == 0:
                # for security
                sales = self.join_df['cong doanh thu hoat dong (01>11) 20']
        self.join_df['sales_per_share'] = sales/num_share
        for t in range(T, self.length + 1):
            copy_vec = copy.deepcopy(self.join_df['sales_per_share'][(t-T):t])
            y = copy_vec.unique()
            X = np.arange(len(y)).reshape(-1, 1)
            regressor = LinearRegression(fit_intercept=False)
            regressor.fit(X, y)
            sgro[t-1] = regressor.coef_/np.nanmean(y)
        return sgro

    # Earning growth (trailing 2 years)
    def ERGO(self, T=int(252*2)):
        ergo = np.zeros(self.length)
        earning = self.join_df['loi nhuan sau thue (xixii)']
        num_share = self.join_df['MarketCap'] * 1e6 / self.join_df['ClosePrice']
        num_share.replace(to_replace=0, method='ffill', inplace=True)
        self.join_df['earning_per_share'] = earning/num_share
        for t in range(T, self.length + 1):
            copy_vec = copy.deepcopy(self.join_df['earning_per_share'][(t - T):t])
            y = copy_vec.unique()
            X = np.arange(len(y)).reshape(-1, 1)
            regressor = LinearRegression(fit_intercept=False)
            regressor.fit(X, y)
            ergo[t-1] = regressor.coef_/np.nanmean(y)
        return ergo

    ################################################################### EARNING YIELD
    def ETOP(self, T=252):
        earning = self.join_df['loi nhuan sau thue (xixii)']
        mkt_cap = self.join_df['MarketCap'] * 1e6
        mkt_cap.replace(to_replace=0, method='ffill', inplace=True)
        return earning/mkt_cap
