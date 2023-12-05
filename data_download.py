# data_download.py
"""
this script downloads data from baostock and saves it to a csv file in the stocks_data directory    
"""
from dependencies import *
import os

def login():
    lg = bs.login()
    return lg

def logout():
    bs.logout()

class DataCollector:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.index_data = None

    def download_index_composition(self, date, output_directory):
        rs = bs.query_zz500_stocks(date)
        self.index_data = rs.get_data()
        file_path = os.path.join(output_directory, "index.csv")
        self.index_data.to_csv(file_path, index=False)


    def download_and_save_stock_data(self, code, output_directory):
        rs = bs.query_history_k_data_plus(
        code, "date,time,code,open,high,low,close,volume,amount",
        self.start_date, self.end_date, frequency="30", adjustflag="3")
        data = rs.get_data()

        if not data.empty:
            # Save data to CSV file
            file_path = os.path.join(output_directory, f"{code}.csv")
            data.to_csv(file_path, index=False)
            print(f"Data for {code} downloaded and saved.")
        else:
            print(f"No data for {code}.")


    def collect_stock_data(self, output_directory):
        i=0
        for code in self.index_data['code']:
            i+=1
            self.download_and_save_stock_data(code, output_directory)
            print(f"Progress: {i*100/len(self.index_data)} %")
            

if __name__ == "__main__":
    output_directory = 'stocks_data'
    os.makedirs(output_directory, exist_ok=True)

    login()
    try:
        collector = DataCollector(start_date='2022-04-01', end_date='2022-07-31')
        collector.download_index_composition('2021-01-01',output_directory)
        collector.collect_stock_data(output_directory)
        

    finally:
        logout()
