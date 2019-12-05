'''
    Author: Adam Jilling
    Evaluate each dataset's rankings for both runtime and performance
'''

import rpy2.robjects as ro
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

def get_success_per_place(directory):

    df_rankings = pd.read_csv(directory + "results_ranked.csv", header=None)
    df_rankings = df_rankings.drop([df_rankings.index[0]])

    # store predicted values here, ideally 'runtime1_places' would be mostly 1's and 2's
    runtime1_places = []
    runtime2_places = []
    runtime3_places = []
    runtime4_places = []
    runtime5_places = []
    runtime6_places = []
    runtime7_places = []

    performance1_places = []
    performance2_places = []
    performance3_places = []
    performance4_places = []
    performance5_places = []
    performance6_places = []
    performance7_places = []

    for index, row in df_rankings.iterrows():

        actual_run_rank = int(round(float(row[4])))
        pred_run_rank = int(round(float(row[2])))
        actual_perf_rank = int(round(float(row[5])))
        pred_perf_rank = int(round(float(row[3])))

        if (actual_run_rank == 1):
            runtime1_places.append(pred_run_rank)
        if (actual_run_rank == 2):
            runtime2_places.append(pred_run_rank)
        if (actual_run_rank == 3):
            runtime3_places.append(pred_run_rank)
        if (actual_run_rank == 4):
            runtime4_places.append(pred_run_rank)
        if (actual_run_rank == 5):
            runtime5_places.append(pred_run_rank)
        if (actual_run_rank == 6):
            runtime6_places.append(pred_run_rank)
        if (actual_run_rank == 7):
            runtime7_places.append(pred_run_rank)

        if (actual_perf_rank == 1):
            performance1_places.append(pred_perf_rank)
        if (actual_perf_rank == 2):
            performance2_places.append(pred_perf_rank)
        if (actual_perf_rank == 3):
            performance3_places.append(pred_perf_rank)
        if (actual_perf_rank == 4):
            performance4_places.append(pred_perf_rank)
        if (actual_perf_rank == 5):
            performance5_places.append(pred_perf_rank)
        if (actual_perf_rank == 6):
            performance6_places.append(pred_perf_rank)
        if (actual_perf_rank == 7):
            performance7_places.append(pred_perf_rank)


    # count and clean runtime values
    run1_count = [runtime1_places.count(1), runtime1_places.count(2), 
        runtime1_places.count(3), runtime1_places.count(4), runtime1_places.count(5), 
        runtime1_places.count(6), runtime1_places.count(7)]
    run2_count = [runtime2_places.count(1), runtime2_places.count(2), 
        runtime2_places.count(3), runtime2_places.count(4), runtime2_places.count(5), 
        runtime2_places.count(6), runtime2_places.count(7)]
    run3_count = [runtime3_places.count(1), runtime3_places.count(2), 
        runtime3_places.count(3), runtime3_places.count(4), runtime3_places.count(5), 
        runtime3_places.count(6), runtime3_places.count(7)]
    run4_count = [runtime4_places.count(1), runtime4_places.count(2), 
        runtime4_places.count(3), runtime4_places.count(4), runtime4_places.count(5), 
        runtime4_places.count(6), runtime4_places.count(7)]
    run5_count = [runtime5_places.count(1), runtime5_places.count(2), 
        runtime5_places.count(3), runtime5_places.count(4), runtime5_places.count(5), 
        runtime5_places.count(6), runtime5_places.count(7)]
    run6_count = [runtime6_places.count(1), runtime6_places.count(2), 
        runtime6_places.count(3), runtime6_places.count(4), runtime6_places.count(5), 
        runtime6_places.count(6), runtime6_places.count(7)]
    run7_count = [runtime7_places.count(1), runtime7_places.count(2), 
        runtime7_places.count(3), runtime7_places.count(4), runtime7_places.count(5), 
        runtime7_places.count(6), runtime7_places.count(7)]

    print("RT #1: \t", run1_count)
    print("RT #2: \t", run2_count)
    print("RT #3: \t", run3_count)
    print("RT #4: \t", run4_count)
    print("RT #5: \t", run5_count)
    print("RT #6: \t", run6_count)
    print("RT #7: \t", run7_count)

    # count and clean performance values
    perf1_count = [performance1_places.count(1), performance1_places.count(2), 
        performance1_places.count(3), performance1_places.count(4), performance1_places.count(5), 
        performance1_places.count(6), performance1_places.count(7)]
    perf2_count = [performance2_places.count(1), performance2_places.count(2), 
        performance2_places.count(3), performance2_places.count(4), performance2_places.count(5), 
        performance2_places.count(6), performance2_places.count(7)]
    perf3_count = [performance3_places.count(1), performance3_places.count(2), 
        performance3_places.count(3), performance3_places.count(4), performance3_places.count(5), 
        performance3_places.count(6), performance3_places.count(7)]
    perf4_count = [performance4_places.count(1), performance4_places.count(2), 
        performance4_places.count(3), performance4_places.count(4), performance4_places.count(5), 
        performance4_places.count(6), performance4_places.count(7)]
    perf5_count = [performance5_places.count(1), performance5_places.count(2), 
        performance5_places.count(3), performance5_places.count(4), performance5_places.count(5), 
        performance5_places.count(6), performance5_places.count(7)]
    perf6_count = [performance6_places.count(1), performance6_places.count(2), 
        performance6_places.count(3), performance6_places.count(4), performance6_places.count(5), 
        performance6_places.count(6), performance6_places.count(7)]
    perf7_count = [performance7_places.count(1), performance7_places.count(2), 
        performance7_places.count(3), performance7_places.count(4), performance7_places.count(5), 
        performance7_places.count(6), performance7_places.count(7)]

    print("Pf #1: \t", perf1_count)
    print("Pf #2: \t", perf2_count)
    print("Pf #3: \t", perf3_count)
    print("Pf #4: \t", perf4_count)
    print("Pf #5: \t", perf5_count)
    print("Pf #6: \t", perf6_count)
    print("Pf #7: \t", perf7_count)

    #pd.DataFrame(df_rankings).to_csv("~/Desktop/results.csv", index = False, header = False)


def get_one_two_three_shot(directory):

    df_rankings = pd.read_csv(directory + "results_ranked.csv", header=None)
    df_rankings = df_rankings.drop([df_rankings.index[0]])

    runtime_1_to_1 = runtime_1_to_2 = runtime_1_to_3 = 0
    performance_1_to_1 = performance_1_to_2 = performance_1_to_3 = 0

    for _, row in df_rankings.iterrows():

        if (float(row[4]) == 1.0 and float(row[2]) == 1.0):
            runtime_1_to_1+=1
        if (float(row[4]) == 1.0 and float(row[2]) == 2.0):
            runtime_1_to_2+=1
        if (float(row[4]) == 1.0 and float(row[2]) == 3.0):
            runtime_1_to_3+=1
        if (float(row[5]) == 1.0 and float(row[3]) == 1.0):
            performance_1_to_1+=1
        if (float(row[5]) == 1.0 and float(row[3]) == 2.0):
            performance_1_to_2+=1
        if (float(row[5]) == 1.0 and float(row[3]) == 3.0):
            performance_1_to_3+=1
    
    print("RT 1:1 \t", runtime_1_to_1)
    print("RT 1:2 \t", runtime_1_to_2)
    print("RT 1:3 \t", runtime_1_to_3)
    print("Pf 1:1 \t", performance_1_to_1)
    print("Pf 1:2 \t", performance_1_to_2)
    print("Pf 1:3 \t", performance_1_to_3)



def get_algos_ranked_top_three(directory):

    df_rankings = pd.read_csv(directory + "results_ranked.csv", header=None)
    df_rankings = df_rankings.drop([df_rankings.index[0]])

    # get all top 3 hits for predicted runtime column
    pred_rt1 = []
    pred_rt2 = []
    pred_rt3 = []
    for index, row in df_rankings.iterrows():
        pred_run_rank = int(round(float(row[2])))
        if (pred_run_rank == 1):
            pred_rt1.append(row[1])
        if (pred_run_rank == 2):
            pred_rt2.append(row[1])
        if (pred_run_rank == 3):
            pred_rt3.append(row[1])

    # get all top 3 hits for predicted performance column
    pred_perf1 = []
    pred_perf2 = []
    pred_perf3 = []
    for index, row in df_rankings.iterrows():
        pred_perf_rank = int(round(float(row[3])))
        if (pred_perf_rank == 1):
            pred_perf1.append(row[1])
        if (pred_perf_rank == 2):
            pred_perf2.append(row[1])
        if (pred_perf_rank == 3):
            pred_perf3.append(row[1])

    # get all top 3 hits for actual runtime column
    act_rt1 = []
    act_rt2 = []
    act_rt3 = []
    for index, row in df_rankings.iterrows():
        actual_run_rank = int(round(float(row[4])))
        if (actual_run_rank == 1):
            act_rt1.append(row[1])
        if (actual_run_rank == 2):
            act_rt2.append(row[1])
        if (actual_run_rank == 3):
            act_rt3.append(row[1])

    # get all top 3 hits for actual performance column
    act_perf1 = []
    act_perf2 = []
    act_perf3 = []
    for index, row in df_rankings.iterrows():
        actual_perf_rank = int(round(float(row[5])))
        if (actual_perf_rank == 1):
            act_perf1.append(row[1])
        if (actual_perf_rank == 2):
            act_perf2.append(row[1])
        if (actual_perf_rank == 3):
            act_perf3.append(row[1])  

    total_pred_rt1 = [pred_rt1.count('aa'), pred_rt1.count('ca'), 
        pred_rt1.count('gmd'), pred_rt1.count('gmf'), pred_rt1.count('km'),
        pred_rt1.count('mk'), pred_rt1.count('wa')]
    total_pred_rt2 = [pred_rt2.count('aa'), pred_rt2.count('ca'), 
        pred_rt2.count('gmd'), pred_rt2.count('gmf'), pred_rt2.count('km'),
        pred_rt2.count('mk'), pred_rt2.count('wa')]
    total_pred_rt3 = [pred_rt3.count('aa'), pred_rt3.count('ca'), 
        pred_rt3.count('gmd'), pred_rt3.count('gmf'), pred_rt3.count('km'),
        pred_rt3.count('mk'), pred_rt3.count('wa')]

    total_pred_perf1 = [pred_perf1.count('aa'), pred_perf1.count('ca'), 
        pred_perf1.count('gmd'), pred_perf1.count('gmf'), pred_perf1.count('km'),
        pred_perf1.count('mk'), pred_perf1.count('wa')]
    total_pred_perf2 = [pred_perf2.count('aa'), pred_perf2.count('ca'), 
        pred_perf2.count('gmd'), pred_perf2.count('gmf'), pred_perf2.count('km'),
        pred_perf2.count('mk'), pred_perf2.count('wa')]
    total_pred_perf3 = [pred_perf3.count('aa'), pred_perf3.count('ca'), 
        pred_perf3.count('gmd'), pred_perf3.count('gmf'), pred_perf3.count('km'),
        pred_perf3.count('mk'), pred_perf3.count('wa')]

    total_act_rt1 = [act_rt1.count('aa'), act_rt1.count('ca'), 
        act_rt1.count('gmd'), act_rt1.count('gmf'), act_rt1.count('km'),
        act_rt1.count('mk'), act_rt1.count('wa')]
    total_act_rt2 = [act_rt2.count('aa'), act_rt2.count('ca'), 
        act_rt2.count('gmd'), act_rt2.count('gmf'), act_rt2.count('km'),
        act_rt2.count('mk'), act_rt2.count('wa')]
    total_act_rt3 = [act_rt3.count('aa'), act_rt3.count('ca'), 
        act_rt3.count('gmd'), act_rt3.count('gmf'), act_rt3.count('km'),
        act_rt3.count('mk'), act_rt3.count('wa')]

    total_act_perf1 = [act_perf1.count('aa'), act_perf1.count('ca'), 
        act_perf1.count('gmd'), act_perf1.count('gmf'), act_perf1.count('km'),
        act_perf1.count('mk'), act_perf1.count('wa')]
    total_act_perf2 = [act_perf2.count('aa'), act_perf2.count('ca'), 
        act_perf2.count('gmd'), act_perf2.count('gmf'), act_perf2.count('km'),
        act_perf2.count('mk'), act_perf2.count('wa')]
    total_act_perf3 = [act_perf3.count('aa'), act_perf3.count('ca'), 
        act_perf3.count('gmd'), act_perf3.count('gmf'), act_perf3.count('km'),
        act_perf3.count('mk'), act_perf3.count('wa')]

    print("Order: \t\t\t [aa, ca, gmd, gmf, km, mk, wa]")
    print("total_pred_rt1: \t", total_pred_rt1)
    print("total_pred_rt2: \t", total_pred_rt2)
    print("total_pred_rt3: \t", total_pred_rt3)
    print("total_pred_perf1: \t", total_pred_perf1)
    print("total_pred_perf2: \t", total_pred_perf2)
    print("total_pred_perf3: \t", total_pred_perf3)
    print("total_act_rt1: \t\t", total_act_rt1)
    print("total_act_rt2: \t\t", total_act_rt2)
    print("total_act_rt3: \t\t", total_act_rt3)
    print("total_act_perf1: \t", total_act_perf1)
    print("total_act_perf2: \t", total_act_perf2)
    print("total_act_perf3: \t", total_act_perf3)


directory = "~/Desktop/Thesis/Results/"
get_success_per_place(directory)
get_one_two_three_shot(directory)
get_algos_ranked_top_three(directory)

