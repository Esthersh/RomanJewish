import pandas as pd
import json

"""
Reads json file results into a single csv file
"""

if __name__ == '__main__':
    res_file = "/home/esther/PycharmProjects/RomanJewish/results/mkw_gpt5_1.json"
    df = pd.read_json(res_file)
    df.to_csv(f"{res_file.replace('json', 'csv')}")




