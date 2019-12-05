'''
    Author: Adam Jilling
    Rank each clustering algo for each dataset, 1 through 7 in all 4 categories
    [actual runtime, actual performance, predicted runtime, predicted performance]
'''

import rpy2.robjects as ro
import numpy as np
import pandas as pd

def get_rankings(directory):

    df_results = pd.read_csv(directory + "results.csv", header=None)
    df_rankings = pd.DataFrame()

    low = 1
    increment = 7
    high = low + increment
    for index, row in df_results.iterrows():
        print("Row {} of 1609...".format(index))
        if (index % increment == 0):
            
            df_current = df_results.iloc[low:high, :]
            pred_time = df_current.iloc[:, 2].rank(ascending=True)
            pred_perf = df_current.iloc[:, 3].rank(ascending=False)
            actual_time = df_current.iloc[:, 4].rank(ascending=True)
            actual_perf = df_current.iloc[:, 5].rank(ascending=False)

            temp_df = pd.concat([pred_time, pred_perf, actual_time, actual_perf], axis=1)

            df_rankings = df_rankings.append(temp_df)

            low += increment
            high += increment

    pd.DataFrame(df_rankings).to_csv("~/Desktop/results_ranked.csv", index = False, header = False)

directory = "~/Desktop/Thesis/Results/"
get_rankings(directory)

