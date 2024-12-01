import os
import csv
import re
import matplotlib.pyplot as plt
import numpy as np


def get_trader_names(file_name, trader_names):
    parts = file_name.split("-")
    trader1 = None
    trader2 = None
    for i in range(1, len(parts)):
        if parts[i] == "09" or parts[i] == "09.csv":
            trader1 = trader_names[i]
        if parts[i] == "01" or parts[i] == "01.csv":
            trader2 = trader_names[i]
    return [trader1, trader2]


def process_csv_file(file_name,master_trader):
    trader1_wins = 0
    trader2_wins = 0
    trader1_count = 0
    row_count = 0
    with open(file_name, newline="") as f:
        reader = csv.reader(f)
        print(file_name)
        for row in reader:
            row_count += 1
            trader1_profit = float(row[4])
            trader2_profit = float(row[11])
            if trader1_profit >= trader2_profit:
                trader1_wins += 1
            elif trader1_profit < trader2_profit:
                trader2_wins += 1
    trader1_count/=2

    if row_count != 1000:
        raise ValueError(f"Error: {file_name} does not have 1000 rows. It has {row_count} rows.")

    with open(file_name, newline="") as f:
        reader = csv.reader(f)
        first_row = next(reader)

    trader1_name = first_row[1][1:]
    trader1_count = int(first_row[3])
    trader2_count = int(first_row[10])
    if trader1_name==master_trader:
        return [trader1_count/2,trader1_wins, trader2_wins]
    else:
        return [trader2_count/2,trader2_wins, trader1_wins]


def process_csv_folder(folder_path):

    match = re.search(r'/(?P<trader1>\w+)_vs_', folder_path)
    master_trader = match.group('trader1')
    total_wins = [0, 0]
    all_wins = {}
    trader1_wins = 0
    trader2_wins = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            try:
                trader1_count, trader1_wins, trader2_wins = process_csv_file(os.path.join(folder_path, file_name),
                                                                             master_trader)
                all_wins[trader1_count] = [trader1_wins, trader2_wins]
                total_wins[0] += trader1_wins
                total_wins[1] += trader2_wins
            except ValueError as e:
                print(e)
                continue  # 跳过有错误的文件，继续处理其他文件

    sorted_all_wins = dict(sorted(all_wins.items()))


    comparison = re.search(r'result6\/(.+)', folder_path).group(1)
    win_diffs = {k: v[0] - v[1] for k, v in sorted_all_wins.items()}

    for n,ratio in all_wins.items():
        if sum(ratio)!=1000:
            print(f"Error with trader number{n} and ratio {ratio} \n. Comes from comparison {comparison} ")
    
    #plt.plot(win_diffs.keys(), win_diffs.values())
    x = np.arange(1, 10)
    plt.plot(x, list(win_diffs.values()))
    plt.plot([1, 9], [0, 0], 'k--', lw=1)  # Add this line to draw a solid line at y=0
    plt.xlabel("Number of "+master_trader+" traders")
    plt.ylabel("Wins Difference")
    #plt.title(comparison+" Delta Curve")
    plt.xlim(1, 9)
    #plt.ylim(-1100, 1100)  # might not need this
    plt.ylim(-1000, 1000)
    plt.grid(axis='y', linestyle='-', alpha=0.7)
    plt.box(False)
    plt.savefig('/Users/lengjiwei/sourcescode/Pycharmprojects/Bristol-Frequent-Batch-Stock-Exchange_V2/analysis/results_sensitive/result6/'+ comparison +".pdf")
    plt.show()
    plt.clf()
    return sorted_all_wins, total_wins

### NEW ###
data = []

path = "/Users/lengjiwei/sourcescode/Pycharmprojects/Bristol-Frequent-Batch-Stock-Exchange_V2/analysis/results_sensitive/result6"  # the path to the directory containing the folders
for folder_name in os.listdir(path):
    if os.path.isdir(os.path.join(path, folder_name)):
        all_wins, total_wins = process_csv_folder(os.path.join(path, folder_name))
        # extract the two trader names from the folder name
        trader_names = folder_name.split("_vs_")
        trader1 = trader_names[0]
        trader2 = trader_names[1]
        # add the total_wins to the data structure
        data.append((trader1, trader2, total_wins[0], total_wins[1]))

print("data =", data)
