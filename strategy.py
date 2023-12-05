# strategy.py
# run data_download.py before to download data from baostock
"""
this script uses the data downloaded from baostock to implement a mean reverting strategy on all stocks in the index and calculates the performance of the strategy
"""
from dependencies import *
import os
stocks_data_directory = 'stocks_data'

class Mean_reverting_strategy:
    def __init__(self, constant_notional=100):
        self.constant_notional = constant_notional
        self.strategy_result = None
        self.stock_codes = None
        self.back_test_result = None
        self.stats = None


    def get_stock_codes(self):
        # get stock codes from index.csv
        self.stock_codes = pd.read_csv(os.path.join(stocks_data_directory, "index.csv"))['code'].tolist()
        

    def get_feature(self, feature_name):
        # get feature from csv files
        data = pd.read_csv(os.path.join(stocks_data_directory, f"{self.stock_codes[0]}.csv"))
        date_time = pd.to_datetime(data['time'].astype(str), format="%Y%m%d%H%M%S%f")
        data['date_time'] = date_time

        # set index as date_time
        data.set_index('date_time', inplace=True)
        data = data[[feature_name]]
        data.rename(columns={feature_name: self.stock_codes[0]}, inplace=True)

        for code in self.stock_codes[1:]:
            temp = pd.read_csv(os.path.join(stocks_data_directory, f"{code}.csv"))
            date_time = pd.to_datetime(temp['time'].astype(str), format="%Y%m%d%H%M%S%f")
            temp['date_time'] = date_time
            # set index as date_time
            temp.set_index('date_time', inplace=True)
            temp = temp[[feature_name]]
            temp.rename(columns={feature_name: code}, inplace=True)
            #data = pd.concat([data, temp], axis=1)
            data = data.join(temp, how='outer')
        data = data.dropna(axis=0, how='all')    
        return data   
     
       
    def z_scores_signal(self, lookback_window=5,threshold=1.5,safety_threshold=2):
        # Calculate z-scores for each stock for each day and return signals accordingly
        data_close = self.get_feature('close')
        data_close = data_close.fillna(method='ffill')

        data_volume = self.get_feature('volume')
        data_volume = data_volume.fillna(method='ffill')

        rolling_mean_close = data_close.rolling(window=lookback_window).mean()
        rolling_std_close = data_close.rolling(window=lookback_window).std()
        z_scores_close = (data_close - rolling_mean_close) / rolling_std_close

        rolling_mean_volume = data_volume.rolling(window=lookback_window).mean()
        rolling_std_volume = data_volume.rolling(window=lookback_window).std()
        z_scores_volume = (data_volume - rolling_mean_volume) / rolling_std_volume


        signals = pd.DataFrame(0, index=data_close.index, columns=data_close.columns)
        signals[(z_scores_close < -threshold) & (z_scores_close>- safety_threshold)] = abs(z_scores_close)/abs(rolling_mean_volume) ## oversold // will mean revert up

        return signals
    
    
    def calculate_rsi_signal(self, window=14,threshold=30):
        # Calculate RSI for each stock for each day and return signals accordingly
        data = self.get_feature('close')
        # Calculate daily price changes
        daily_changes = data.diff()

        # Calculate gains and losses
        gains = np.where(daily_changes > 0, daily_changes, 0)
        losses = np.where(daily_changes < 0, -daily_changes, 0)

        # Calculate average gains and losses over the specified window
        avg_gains = pd.DataFrame(gains, index=data.index, columns=data.columns).rolling(window=window, min_periods=1).mean()
        avg_losses = pd.DataFrame(losses, index=data.index, columns=data.columns).rolling(window=window, min_periods=1).mean()

        # Calculate relative strength (RS)
        rs = avg_gains / avg_losses

        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        signals = pd.DataFrame(0, index=data.index, columns=data.columns)
        signals[rsi<threshold] = (100-rsi) ## oversold // will mean revert up
        return signals
    

    def mean_reverting_strategy(self, z_score_lookback_window=5,rsi_lookback_window=14,z_score_threshold=1.5,z_score_safety_threshold=2,rsi_threshold=30):
        # Calculate signals for each stock for each day by combining z-score and RSI signals
        z_score_signals = self.z_scores_signal(z_score_lookback_window,z_score_threshold,z_score_safety_threshold)
        rsi_oversold_condition = self.calculate_rsi_signal(rsi_lookback_window,rsi_threshold)

        # Convert to boolean
        z_score_signals_bool = z_score_signals.astype(bool)
        rsi_oversold_condition_bool = rsi_oversold_condition.astype(bool)

        # Combine signals
        combined_signals = pd.DataFrame(0, index=z_score_signals.index, columns=z_score_signals.columns)
        # Buy signal: Asset is oversold according to both Z-score and RSI
        combined_signals[z_score_signals_bool & rsi_oversold_condition_bool] = rsi_oversold_condition*z_score_signals
        combined_signals = combined_signals.fillna(0)
        # adjust portfolio allocation
        combined_signals=self.adjust_portfolio_allocation(combined_signals)
        combined_signals = combined_signals.fillna(0)
        self.strategy_result=combined_signals
        return combined_signals
    

    def adjust_portfolio_allocation(self, positions):
        # Ensure total holding position at each day across all stocks does not exceed 100
        row_sums = positions.sum(axis=1)
        positions = positions.mul((self.constant_notional / row_sums), axis=0)
        return positions

    def get_results(self,parameters):
        # Calculate performance of the strategy
        # initialize parameters
        start_date = pd.to_datetime(parameters['start_date'])
        end_date = pd.to_datetime(parameters['end_date'])
        z_score_lookback_window = parameters.get('z_score_lookback_window', 5)
        rsi_lookback_window = parameters.get('rsi_lookback_window', 14)
        z_score_threshold = parameters.get('z_score_threshold', 1.5)
        z_score_safety_threshold = parameters.get('z_score_safety_threshold', 2)
        rsi_threshold = parameters.get('rsi_threshold', 30)
        risk_free_rate = 0
        stats = {}
         
        # getting returns and positions for backtesting
        data_close = self.get_feature('close')
        data_close = data_close.loc[start_date:end_date]
        self.mean_reverting_strategy(z_score_lookback_window,rsi_lookback_window,z_score_threshold,z_score_safety_threshold,rsi_threshold)
        positions=self.strategy_result
        positions = positions.loc[start_date:end_date]
        result = pd.DataFrame()
        
        returns = data_close.pct_change()
        returns = positions.shift(1) * returns
        # log_returns = positions.shift(1) * np.log(returns)
        result['returns'] = returns.sum(axis=1)
        # result['log_returns'] = log_returns.sum(axis=1)
        result['cumulative_returns'] = (result['returns']).cumsum()
        result['peaks'] = result['cumulative_returns'].cummax()
        down_side_returns = result['returns'][result['returns'] < risk_free_rate]

        # Calculate performance statistics
        stats['total_return'] = result['cumulative_returns'].iloc[-1]
        stats['sharpe_ratio'] = (result['returns'].mean()*252 - risk_free_rate) / (result['returns'].std()*np.sqrt(252))
        stats['max_drawdown'] = (result['peaks'] - result['cumulative_returns'] ).max()
        stats['sortino_ratio'] = (result['returns'].mean()*252 - risk_free_rate) / (down_side_returns.std()*np.sqrt(252))
        self.back_test_result=result
        self.stats=stats

        return result,stats
    
    def index_holding_result(self,parameters):
        # Calculate performance of the index
        # initialize parameters
        start_date = pd.to_datetime(parameters['start_date'])
        end_date = pd.to_datetime(parameters['end_date'])
        risk_free_rate = 0
        stats = {}
         
        # getting returns and positions for backtesting
        data_close = self.get_feature('close')
        data_close = data_close.loc[start_date:end_date]
        result = pd.DataFrame()
        
        returns = data_close.pct_change()
        returns = returns*(self.constant_notional/len(self.stock_codes))
        result['returns'] = returns.sum(axis=1)
        result['cumulative_returns'] = (result['returns']).cumsum()
        result['peaks'] = result['cumulative_returns'].cummax()
        down_side_returns = result['returns'][result['returns'] < risk_free_rate]

        # Calculate performance statistics
        stats['total_return'] = result['cumulative_returns'].iloc[-1]
        stats['sharpe_ratio'] = (result['returns'].mean()*252 - risk_free_rate) / (result['returns'].std()*np.sqrt(252))
        stats['max_drawdown'] = (result['peaks'] - result['cumulative_returns'] ).max()
        stats['sortino_ratio'] = (result['returns'].mean()*252 - risk_free_rate) / (down_side_returns.std()*np.sqrt(252))
        self.back_test_result=result
        self.stats=stats

        return result,stats
        

if __name__ == "__main__":
    
    strategy = Mean_reverting_strategy()
    strategy.get_stock_codes()
    # initialize parameters

    # insample_start_date = '2022-04-01'
    # insample_end_date = '2022-06-30'
    # outsample_start_date = '2022-07-01'
    # outsample_end_date = '2022-07-31'

    parameters_insample = {}
    parameters_insample['z_score_lookback_window'] = 5
    parameters_insample['rsi_lookback_window'] = 14
    parameters_insample['z_score_threshold'] = 1.5
    parameters_insample['z_score_safety_threshold'] = 2
    parameters_insample['rsi_threshold'] = 30
    parameters_insample['start_date'] = '2022-04-01'
    parameters_insample['end_date'] = '2022-06-30'

    parameters_outsample = {}
    parameters_outsample['z_score_lookback_window'] = parameters_insample['z_score_lookback_window']
    parameters_outsample['rsi_lookback_window'] = parameters_insample['rsi_lookback_window']
    parameters_outsample['z_score_threshold'] = parameters_insample['z_score_threshold']
    parameters_outsample['z_score_safety_threshold'] = parameters_insample['z_score_safety_threshold']
    parameters_outsample['rsi_threshold'] = parameters_insample['rsi_threshold']
    parameters_outsample['start_date'] = '2022-07-01'
    parameters_outsample['end_date'] = '2022-07-31'

    
    returns_insample,stats_insample = strategy.get_results(parameters_insample)
    print("In-sample results: strategy")
    print(returns_insample)
    print(stats_insample)
    plt.figure()
    returns_insample['cumulative_returns'].plot()
    plt.title('Cumulative returns of strategy for In-sample period')
    plt.xlabel('time')
    plt.ylabel('cum_returns')
    # plt.savefig('strat_insample.png')
    # plt.show()

    index_returns_insample,index_stats_insample = strategy.index_holding_result(parameters_insample)
    print("In-sample results: index")
    print(index_returns_insample)
    print(index_stats_insample)
    # plt.figure()
    index_returns_insample['cumulative_returns'].plot()
    plt.title('Cumulative returns for In-sample period')
    plt.xlabel('time')
    plt.ylabel('cum_returns')
    plt.legend(['strategy','index'])
    plt.savefig('insample.png')
    # plt.show()


    returns_outsample,stats_outsample = strategy.get_results(parameters_outsample)
    print("Out-sample results: strategy")
    print(returns_outsample)
    print(stats_outsample)
    plt.figure()
    returns_outsample['cumulative_returns'].plot()
    plt.title('Cumulative returns of strategy for Out-sample period')
    plt.xlabel('time')
    plt.ylabel('cum_returns')
    # plt.savefig('strat_outsample.png')
    # plt.show()

    index_returns_outsample,index_stats_outsample = strategy.index_holding_result(parameters_outsample)
    print("Out-sample results: index")
    print(index_returns_outsample)
    print(index_stats_outsample)
    # plt.figure()
    index_returns_outsample['cumulative_returns'].plot()
    plt.title('Cumulative returns for Out-sample period')
    plt.xlabel('time')
    plt.ylabel('cum_returns')
    plt.legend(['strategy','index'])
    plt.savefig('outsample.png')
    # plt.show()


    # Save the plot to a file (e.g., a PNG file)
    # plt.savefig('plot.png')

    # Display the plot
    