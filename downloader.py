import openml
import pandas as pd 

#openml.config.apikey = 'cf1fee030297dc9a8fdbcf04bde5725a'

# removed 374 (corrupt), 501 (corrupt), 1240 (> 100MB)
dataset_ids = [53, 62, 8, 39, 40, 41, 43, 48, 187, 164, 329, 285, 336, 337, 338, 461, 463, 444, 446, 860, 936, 1115, 1167, 1565, 1600, 1551, 1554, 1556, 1559, 1523, 1524, 1412, 1413, 1511, 1512, 1513, 1519, 1520, 1455, 1482, 1490, 1473, 1463, 1465, 1508, 1495, 1498, 1500, 1441, 1442, 1446, 1447, 1448, 1449, 1450, 4153, 4329, 4340, 23499, 40700, 40663, 40665, 40669, 40710, 40671, 40681, 40682, 40686, 40711, 1121, 1071, 996, 1488, 1059, 1026, 829, 987, 1064, 450, 448, 721, 730, 748, 749, 685, 768, 746, 742, 753, 766, 773, 754, 778, 750, 767, 794, 814, 747, 464, 475, 732, 745, 719, 764, 756, 789, 765, 838, 775, 694, 808, 776, 714, 818, 724, 855, 744, 763, 879, 783, 796, 876, 850, 726, 828, 805, 865, 886, 889, 40496, 811, 907, 736, 906, 863, 885, 801, 788, 779, 900, 908, 884, 890, 918, 925, 782, 784, 716, 878, 935, 733, 762, 916, 875, 920, 873, 880, 832, 824, 830, 941, 812, 955, 926, 915, 769, 973, 793, 870, 974, 937, 820, 792, 851, 921, 1012, 933, 867, 1006, 902, 943, 1013, 932, 969, 952, 834, 909, 1005, 1025, 771, 896, 895, 1061, 1066, 1506, 1048, 1011, 868, 965, 924, 1100, 1054, 922, 877, 869, 10, 1065, 956, 911, 888, 1045, 61, 1075, 59]

print("Downloading " + str(len(dataset_ids)) + " datasets...")

# iterate through each dataset
for id in dataset_ids:
    dataset = openml.datasets.get_dataset(id)
    data = dataset.get_data()
    name = dataset.name

    print("...writing dataset %s" %id)
    pd.DataFrame(data).to_csv("~/Desktop/Thesis/Data/" + str(id) + "_" + name + ".csv", 
        header=None, index=None
    )