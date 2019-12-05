'''
    Author: Adam Jilling
    Build tensors for input/output
'''

import rpy2.robjects as ro
import numpy as np
import pandas as pd
from sklearn import preprocessing
np.set_printoptions(threshold=np.inf)

df = pd.DataFrame()

def build_input_vectors(directory):

    # load metafeature CSV files
    stat_df = pd.read_csv(directory + "stat_features.csv")
    dist_df = pd.read_csv(directory + "dist_features.csv")

    # merge all input values into one df
    raw_df = stat_df.join(dist_df.set_index('name'), on='name')

    global df
    
    # for each entry, create 7 entries for each algorithm
    for index, row in raw_df.iterrows():
        row_aa = pd.Series(row.append(pd.Series([1,0,0,0,0,0,0])).values[1:])
        row_ca = pd.Series(row.append(pd.Series([0,1,0,0,0,0,0])).values[1:])
        row_gmd = pd.Series(row.append(pd.Series([0,0,1,0,0,0,0])).values[1:])
        row_gmf = pd.Series(row.append(pd.Series([0,0,0,1,0,0,0])).values[1:])
        row_km = pd.Series(row.append(pd.Series([0,0,0,0,1,0,0])).values[1:])
        row_mk = pd.Series(row.append(pd.Series([0,0,0,0,0,1,0])).values[1:])
        row_wa = pd.Series(row.append(pd.Series([0,0,0,0,0,0,1])).values[1:])
        df = df.append(row_aa, ignore_index=True)
        df = df.append(row_ca, ignore_index=True)
        df = df.append(row_gmd, ignore_index=True)
        df = df.append(row_gmf, ignore_index=True)
        df = df.append(row_km, ignore_index=True)
        df = df.append(row_mk, ignore_index=True)
        df = df.append(row_wa, ignore_index=True)

    print("Input vectors {0} ..... ✅".format(df.shape))


def build_output_vectors(directory):

    global df

    aa_df = pd.read_csv(directory + "aa_results.csv")
    ca_df = pd.read_csv(directory + "ca_results.csv")
    gmd_df = pd.read_csv(directory + "gmd_results.csv")
    gmf_df = pd.read_csv(directory + "gmf_results.csv")
    km_df = pd.read_csv(directory + "km_results.csv")
    mk_df = pd.read_csv(directory + "mk_results.csv")
    wa_df = pd.read_csv(directory + "wa_results.csv")
    dfs = [aa_df, ca_df, gmd_df, gmf_df, km_df, mk_df, wa_df]

    # loop through 7 dataframes and create 'avg performance' columns
    for dfx in dfs:

        # normalize first 5 metrics on interval [0, 1] - max is better
        cols_to_normalize = ['calinski', 'silhouette', 'dunn', 'pearson', 'tau']
        x1 = dfx[cols_to_normalize].values
        x1_scaled = preprocessing.MinMaxScaler().fit_transform(x1)

        # normalize next 5 metrics on interval [0, 1] and inverse - min is better
        cols_to_normalize_and_flip = ['davies', 'xie', 'sds', 'sdd', 'ray']
        x2 = dfx[cols_to_normalize_and_flip].values
        x2_scaled = abs(preprocessing.MinMaxScaler().fit_transform(x2) - 1)
        
        # combine results and get average of all 10
        df_temp = np.concatenate((x1_scaled, x2_scaled), axis=1)
        df_temp = np.mean(df_temp, axis = 1)
        
        # add 'avg performance' to df
        dfx['avg performance'] = df_temp

    # loop through 7 dataframes and add 'avg runtime' and 'avg performance' in appropriate locations
    outputs = pd.DataFrame(columns=['avg runtime', 'avg performance'])
    for row in range(0, 230):
        for dfx in dfs:
            temp_runtime = dfx.iloc[row]['avg runtime']
            temp_performance = dfx.iloc[row]['avg performance']
            temp_df = pd.DataFrame({"avg runtime":[temp_runtime], "avg performance":[temp_performance]}) 
            outputs = outputs.append(temp_df, ignore_index=True)
    
    print("Output vectors {0} ..... ✅".format(outputs.shape))
    df['avg runtime'] = outputs['avg runtime']
    df['avg performance'] = outputs['avg performance']
    print("Overall vectors {0} ..... ✅".format(df.shape))

    pd.DataFrame(df).to_csv("~/Desktop/tensors.csv", index = False, header = False)
    
    
directory = "~/Desktop/Thesis/Results/"
build_input_vectors(directory)
build_output_vectors(directory)
