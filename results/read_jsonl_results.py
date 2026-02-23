import pandas as pd
import json

"""
Reads json file results into a single csv file
"""

if __name__ == '__main__':
    res_file = "/home/esther/PycharmProjects/RomanJewish/results/annotated_results_5shot_gemini3_keywords_stable.json"
    df = pd.read_json(res_file)
    df.to_csv(f"{res_file.replace('json', 'csv')}")




