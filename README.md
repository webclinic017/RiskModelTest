### I/ Overall:
factor_process.py:
containing a specific class to calculate all factor of each stock:
- the input of the Factor class is daily, sts price and FDMT files of each stock
- the output of the Factor is factors of each stock

utility.py:
Containing:
- Industry_dic, stock_to_industry_dic (map stock to its industry)
- all_list_stock (all stock in universe)
- industry_list
- function: calculate_variance and calculate_inverse_variance (for cross_sectional)
- date_processing: quickly process datetime format

config.py:
- 2 options: MKT_CHECK and INDUSTRY_NORM_CHECK (default first is False, second is True)

### II/ Process:
Please run the following files in consecutive order:
1. data_processing.py:
- please adjust option in the config file before running
- input: folders of daily, sts price and FDMT of all stocks in the universe
Note: the data would be automatically processed with the Factor class from the factor_process.py
- output: a folder containing the FACTOR LOADING MATRIX of each DAY (already normalized by industry)
(output_file_format: row_names: factors, col_names: stocks, file_names: date)

2. Barra_model_cross_sectional.py:
- input: a folder containing all FACTOR LOADING MATRIX of each DAY
- output1: a folder containing FACTORS OF ALL STOCKS every day
(output_file_format: row_names: stocks, col_names: date, file_names: factor_name)
- output2: a csv file containing the FACTOR RETURN MATRIX
(output_file_format: row_names: factor, col_names: date)

3. test_portfolio_analysis.py
- input1: a folder containing FACTORS OF ALL STOCKS every day
- input2: a csv file containing the FACTOR RETURN MATRIX
- input3: a csv file containing weight of each stock in alpha
- output: a plot that breaks down all factor return from industry factors and style factors
