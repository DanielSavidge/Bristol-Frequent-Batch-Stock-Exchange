import matplotlib.pyplot as plt
import  os
# 数据
batch_intervals = [0.05, 0.1, 0.25, 0.5, 2]
algorithms = ['ZIP', 'MIX', 'GVWY', 'AA', 'RaForest', 'ZIC']

static_data_noise = {
    'ZIP': [38349, 37073, 34449, 32939, 29525],
    'MIX': [28444, 31143, 33755, 35446, 37875],
    'GVWY': [28263, 31190, 34586, 35882, 38027],
    'AA': [26042, 22453, 19478, 18033, 17399],
    'RaForest': [10740, 9976, 9740, 9920, 10672],
    'ZIC': [3162, 3165, 2992, 2780, 1502]
}

dynamic_data_noise = {
    'ZIP': [38775, 36844, 33856, 32572, 28763],
    'MIX': [28205, 31459, 34171, 35623, 38000],
    'GVWY': [28700, 31411, 34572, 35894, 38335],
    'AA': [23604, 20558, 18294, 17384, 17129],
    'RaForest': [11609, 10847, 10676, 10671, 11559],
    'ZIC': [4107, 3881, 3431, 2856, 1214]
}

static_data = {
    'ZIP': [37214, 35596, 35173, 34248, 30948],
    'MIX': [30915, 32589, 33328, 34401, 37115],
    'GVWY': [30485, 32538, 33608, 34724, 37254],
    'AA': [20635, 17982, 16076, 14782, 12977],
    'RaForest': [7684, 8794, 8947, 8876, 9402],
    'ZIC': [8067, 7501, 7868, 7969, 7304]
}

dynamic_data = {
    'ZIP': [36757, 35155, 34758, 33956, 31060],
    'MIX': [31726, 33084, 33644, 34706, 37154],
    'GVWY': [30802, 32923, 33836, 35110, 37461],
    'AA': [18112, 16062, 14536, 12856, 10913],
    'RaForest': [9521, 9853, 10557, 10415, 11321],
    'ZIC': [8082, 7923, 7669, 7957, 7091]
}

dynamic_sensitive_data = {
    'ZIP': [37888, 36469, 34809, 33740, 30067],
    'MIX': [28212, 30861, 32657, 33945, 37423],
    'GVWY': [27798, 30170, 32387, 34583, 37284],
    'AA': [24179, 21674, 19906, 18026, 16473],
    'RaForest': [12287, 11790, 11992, 12078, 12785],
    'ZIC': [4636, 4036, 3249, 2628, 968]
}

datasets = [
    ("Noisy Experiment", static_data_noise),
    ("Simulation Experiment", dynamic_data_noise),
    ("Baseline Experiment", static_data),
    ("Dynamic Market Experiment", dynamic_data),
    ("Sensitivities Experiment", dynamic_sensitive_data)
]

def create_plot(title, data):
    plt.figure(figsize=(10, 8))
    for algo in algorithms:
        plt.plot(batch_intervals, data[algo], marker='o', label=algo)

    plt.title(title)
    plt.xlabel('Batch Interval')
    plt.ylabel('Number of Wins')
    plt.xscale('log')
    plt.ylim(0, 40000)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    return plt

# 创建输出目录
current_directory = os.getcwd()
output_directory = os.path.join(current_directory, "results-figure")
os.makedirs(output_directory, exist_ok=True)

# 生成并保存每个图表
for title, data in datasets:
    plot = create_plot(title, data)
    output_file = os.path.join(output_directory, f"{title.replace(' ', '_').lower()}.png")
    plot.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图表以释放内存
