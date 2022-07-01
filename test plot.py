import pandas as pd
import matplotlib.pyplot as plt
from utility import combine_factor_list
import numpy as np
from utility import industry_list
INPUT_FACTOR_RETURN = '/home/vu/PycharmProjects/RiskModelTest/RiskBarraModel_ind_and_mkt/factor_return_df.csv'
import seaborn as sns

factor_return_df = pd.read_csv(INPUT_FACTOR_RETURN, index_col=0)
print(factor_return_df)
corr = factor_return_df.corr()

# plot the heatmap
sns.heatmap(corr,
            xticklabels=corr.columns,
            yticklabels=corr.columns, cmap="Blues", annot=True)
plt.show()

factor_return_df[['beta', 'leverage', 'profitability', 'size', 'value', 'volatility', 'momentum',
                  'growth', 'earning_yield', 'ST_reversal', 'LT_reversal','index']].cumsum().plot()
factor_return_df[industry_list].cumsum().plot()
plt.show()
