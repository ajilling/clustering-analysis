'''
    Author: Adam Jilling
    Scans through a folder of datasets and calculates 
    19 distance-based metafeatures
'''

import rpy2.robjects as ro
import numpy as np
from scipy.stats import skew, kurtosis, zscore
import math
from sklearn import preprocessing
import pandas

def calculate_distance_metafeatures(directory):

    metafeatures = []
    metafeatures.append(["name", "mf1", "mf2", "mf3", "mf4", "mf5", "mf6", 
        "mf7", "mf8", "mf9", "mf10", "mf11", "mf12", "mf13", 
        "mf14", "mf15", "mf16", "mf17", "mf18", "mf19"])
    ro.r.assign('directory', directory)
    ro.r('fileNames = list.files(directory, pattern = ".csv")')
    fileNames = ro.r('fileNames')

    for name in fileNames:
        ro.r.assign('name', name)
        ro.r('mf = read.csv(file = paste0(directory, "/", name), header = FALSE, sep = ",")')
        array = np.array(ro.r('matrix = as.matrix(mf)'))
        print("DATASET: ", name)

        num_rows = np.shape(array)[0]
        d = []

        # pairwise row iteration
        for x in range(num_rows):
            for y in range(x+1, num_rows):
                rowA = array[x]
                rowB = array[y]
                d_val = 0
                for z in range(np.size(rowA)):
                    d_val += (rowA[z]-rowB[z]) ** 2
                d_val = math.sqrt(d_val)
                d.append(d_val)
        
        m = preprocessing.minmax_scale(d)

        # MF1 (mean of m)
        mf1 = sum(m) / len(m)
        print("MF1 = ", mf1)

        # MF2 (variance of m)
        mf2 = np.var(m)
        print("MF2 = ", mf2)

        # MF3 (standard deviation of m)
        mf3 = np.std(m)
        print("MF3 = ", mf3)

        # MF4 (skewness of m)
        mf4 = skew(m)
        print("MF4 = ", mf4)

        # MF5 (kurtosis of m)
        mf5 = kurtosis(m)
        print("MF5 = ", mf5)


        # Calculate next 10 features in 1 pass
        mf6 = mf7 = mf8 = mf9 = mf10 = mf11 = mf12 = mf13 = mf14 = mf15 = 0

        # pad for floating-point rounding errors
        pad = 0.000000000001

        for val in m:
            if val <= 0.1 + pad:
                mf6 += 1
            elif val <= 0.2 + pad:
                mf7 += 1
            elif val <= 0.3 + pad:
                mf8 += 1
            elif val <= 0.4 + pad:
                mf9 += 1
            elif val <= 0.5 + pad:
                mf10 += 1
            elif val <= 0.6 + pad:
                mf11 += 1
            elif val <= 0.7 + pad:
                mf12 += 1
            elif val <= 0.8 + pad:
                mf13 += 1
            elif val <= 0.9 + pad:
                mf14 += 1
            elif val <= 1.0 + pad:
                mf15 += 1

        length = np.size(m)

        # MF6 (% of values in the interval [0.0, 0.1])
        mf6 = mf6 / length
        print("MF6 = ", mf6)

        # MF7 (% of values in the interval [0.1, 0.2])
        mf7 = mf7 / length
        print("MF7 = ", mf7)

        # MF8 (% of values in the interval [0.2, 0.3])
        mf8 = mf8 / length
        print("MF8 = ", mf8)

        # MF9 (% of values in the interval [0.3, 0.4])
        mf9 = mf9 / length
        print("MF9 = ", mf9)

        # MF10 (% of values in the interval [0.4, 0.5])
        mf10 = mf10 / length
        print("MF10 = ", mf10)

        # MF11 (% of values in the interval [0.5, 0.6])
        mf11 = mf11 / length
        print("MF11 = ", mf11)

        # MF12 (% of values in the interval [0.6, 0.7])
        mf12 = mf12 / length
        print("MF12 = ", mf12)

        # MF13 (% of values in the interval [0.7, 0.8])
        mf13 = mf13 / length
        print("MF13 = ", mf13)

        # MF14 (% of values in the interval [0.8, 0.9])
        mf14 = mf14 / length
        print("MF14 = ", mf14)

        # MF15 (% of values in the interval [0.9, 1.0])
        mf15 = mf15 / length
        print("MF15 = ", mf15)


        # Calculate next 4 features in 1 pass
        mf16 = mf17 = mf18 = mf19 = 0

        z_scores = zscore(m)

        for val in z_scores:
            if abs(val) < 1 + pad:
                mf16 += 1
            elif abs(val) < 2 + pad:
                mf17 += 1
            elif abs(val) < 3 + pad:
                mf18 += 1
            else:
                mf19 += 1

        length = np.size(z_scores)

        # MF16 (% of values with absolute Z-score in the interval [0, 1))
        mf16 = mf16 / length
        print("MF16 = ", mf16)

        # MF17 (% of values with absolute Z-score in the interval [1, 2))
        mf17 = mf17 / length
        print("MF17 = ", mf17)

        # MF18 (% of values with absolute Z-score in the interval [2, 3))
        mf18 = mf18 / length
        print("MF18 = ", mf18)

        # MF19 (% of values with absolute Z-score in the interval [3, ∞))
        mf19 = mf19 / length
        print("MF19 = ", mf19)

        print("\n")

        # tests to verify calculations
        print("VERIFY DATA:")
        print("MF6 -> MF15 (should equal 1): ", mf6 + mf7 + mf8 + mf9 +
            mf10 + mf11 + mf12 + mf13 + mf14 + mf15)
        print("MF16 -> MF19 (should equal 1): ", mf16 + mf17 + mf18 + mf19)

        print("\n")

        # add to result
        metafeatures.append([name, mf1, mf2, mf3, mf4, mf5, mf6, 
            mf7, mf8, mf9, mf10, mf11, mf12, mf13, 
            mf14, mf15, mf16, mf17, mf18, mf19])

    # write to file
    pd = pandas.DataFrame(metafeatures)
    pd.to_csv("~/Desktop/Thesis/Results/dist_features.csv", index = False, header = False)

directory = "~/Desktop/Thesis/Data"
calculate_distance_metafeatures(directory)

