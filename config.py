import os
from rework_backtrader.utils.directory_controller import create_if_not_existed

DATA_FOLDER = '/data'
# default mkt = F and industry = T
MKT_CHECK = False
INDUSTRY_NORM_CHECK = False

if INDUSTRY_NORM_CHECK and MKT_CHECK:
    DATA_PARENT_FOLDER = f"{os.getenv('DATA_PARENT_FOLDER', os.getcwd())}"
    LOADING_FACTOR_FOLDER = os.path.join(DATA_PARENT_FOLDER, 'RiskBarraModel_ind_and_mkt/factor_loading_folder')
    create_if_not_existed(LOADING_FACTOR_FOLDER)
    # DATA_FOLDER = os.path.join(DATA_PARENT_FOLDER, 'data')
    FACTOR_RETURN = os.path.join(DATA_PARENT_FOLDER, 'RiskBarraModel_ind_and_mkt/factor_return_df.csv')
    STOCK_FACTOR_FOLDER = os.path.join(DATA_PARENT_FOLDER, 'RiskBarraModel_ind_and_mkt/sub_factor_folder')
    create_if_not_existed(STOCK_FACTOR_FOLDER)
elif not INDUSTRY_NORM_CHECK and MKT_CHECK:
    DATA_PARENT_FOLDER = f"{os.getenv('DATA_PARENT_FOLDER', os.getcwd())}"
    LOADING_FACTOR_FOLDER = os.path.join(DATA_PARENT_FOLDER, 'RiskBarraModel_full_universe_and_mkt/factor_loading_folder')
    create_if_not_existed(LOADING_FACTOR_FOLDER)
    # DATA_FOLDER = os.path.join(DATA_PARENT_FOLDER, 'data')
    FACTOR_RETURN = os.path.join(DATA_PARENT_FOLDER, 'RiskBarraModel_full_universe_and_mkt/factor_return_df.csv')
    STOCK_FACTOR_FOLDER = os.path.join(DATA_PARENT_FOLDER, 'RiskBarraModel_full_universe_and_mkt/sub_factor_folder')
    create_if_not_existed(STOCK_FACTOR_FOLDER)
elif INDUSTRY_NORM_CHECK and not MKT_CHECK:
    DATA_PARENT_FOLDER = f"{os.getenv('DATA_PARENT_FOLDER', os.getcwd())}"
    LOADING_FACTOR_FOLDER = os.path.join(DATA_PARENT_FOLDER, 'RiskBarraModel_ind_and_no_mkt/factor_loading_folder')
    create_if_not_existed(LOADING_FACTOR_FOLDER)
    # DATA_FOLDER = os.path.join(DATA_PARENT_FOLDER, 'data')
    FACTOR_RETURN = os.path.join(DATA_PARENT_FOLDER, 'RiskBarraModel_ind_and_no_mkt/factor_return_df.csv')
    STOCK_FACTOR_FOLDER = os.path.join(DATA_PARENT_FOLDER, 'RiskBarraModel_ind_and_no_mkt/sub_factor_folder')
    create_if_not_existed(STOCK_FACTOR_FOLDER)
elif not INDUSTRY_NORM_CHECK and not MKT_CHECK:
    DATA_PARENT_FOLDER = f"{os.getenv('DATA_PARENT_FOLDER', os.getcwd())}"
    LOADING_FACTOR_FOLDER = os.path.join(DATA_PARENT_FOLDER, 'RiskBarraModel_full_universe_and_no_mkt/factor_loading_folder')
    create_if_not_existed(LOADING_FACTOR_FOLDER)
    # DATA_FOLDER = os.path.join(DATA_PARENT_FOLDER, 'data')
    FACTOR_RETURN = os.path.join(DATA_PARENT_FOLDER, 'RiskBarraModel_full_universe_and_no_mkt/factor_return_df.csv')
    STOCK_FACTOR_FOLDER = os.path.join(DATA_PARENT_FOLDER, 'RiskBarraModel_full_universe_and_no_mkt/sub_factor_folder')
    create_if_not_existed(STOCK_FACTOR_FOLDER)

