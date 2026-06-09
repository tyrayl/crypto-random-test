import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import chi2, norm

# ==================== 生成随机数序列 ====================
def generate_sequence(length, n_range=10):
    return [random.randint(0, n_range-1) for _ in range(length)]

def frequency_histogram(seq, n_range=10):
    counts = [seq.count(i) for i in range(n_range)]
    plt.bar(range(n_range), counts, tick_label=range(n_range))
    plt.title('随机数频数分布')
    plt.xlabel('数值')
    plt.ylabel('出现次数')
    plt.savefig('frequency_histogram.png')
    plt.close()
    return counts

def interval_analysis(seq, n_range=10):
    intervals = defaultdict(list)
    last_pos = [-1] * n_range
    for pos, val in enumerate(seq):
        if last_pos[val] != -1:
            intervals[val].append(pos - last_pos[val])
        last_pos[val] = pos
    print("\n=== 间隔分布分析 ===")
    for val in range(n_range):
        if intervals[val]:
            avg = np.mean(intervals[val])
            print(f"数字 {val} 的平均出现间隔: {avg:.2f}")
    return intervals

# ==================== 整数序列随机性检验 ====================
def chi_square_test(seq, n_range=10):
    """卡方拟合优度检验：检验每个数字出现频率是否均匀"""
    counts = [seq.count(i) for i in range(n_range)]
    expected = len(seq) / n_range
    chi2_stat = sum((c - expected)**2 / expected for c in counts)
    p_value = 1 - chi2.cdf(chi2_stat, n_range-1)
    return p_value

def runs_test_integer(seq):
    """游程检验（基于序列的升/降）：检验序列是否独立"""
    n = len(seq)
    # 生成符号序列：1表示上升，-1表示下降（相等时忽略）
    signs = []
    for i in range(1, n):
        if seq[i] > seq[i-1]:
            signs.append(1)
        elif seq[i] < seq[i-1]:
            signs.append(-1)
        # 相等时跳过，不产生符号
    if len(signs) == 0:
        return 1.0
    # 计算游程数
    runs = 1
    for i in range(1, len(signs)):
        if signs[i] != signs[i-1]:
            runs += 1
    n1 = signs.count(1)
    n2 = signs.count(-1)
    # 期望和方差
    expected = 2 * n1 * n2 / (n1 + n2) + 1
    variance = (expected - 1) * (expected - 2) / (n1 + n2 - 1)
    if variance <= 0:
        return 1.0
    z = (runs - expected) / np.sqrt(variance)
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return p_value

def poker_test_integer(seq, m=3):
    """扑克检验：将序列分成m个数字一组，检验各组出现频率是否均匀"""
    n = len(seq)
    n_blocks = n // m
    if n_blocks == 0:
        return 1.0
    # 统计每组元组的出现次数
    counts = {}
    for i in range(n_blocks):
        block = tuple(seq[i*m : (i+1)*m])
        counts[block] = counts.get(block, 0) + 1
    # 可能的组合数：10^m
    possible = 10 ** m
    expected = n_blocks / possible
    chi2_stat = sum((c - expected)**2 / expected for c in counts.values())
    df = possible - 1
    p_value = 1 - chi2.cdf(chi2_stat, df)
    return p_value

def autocorrelation_test_integer(seq, lag=1):
    """自相关检验：检验序列与自身偏移lag步后的相关性（使用皮尔逊相关系数）"""
    n = len(seq)
    if n <= lag:
        return 1.0
    x = seq[:-lag]
    y = seq[lag:]
    # 计算相关系数
    r = np.corrcoef(x, y)[0,1]
    # Fisher变换近似正态
    if abs(r) == 1:
        return 1.0
    z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(n - lag - 3)
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return p_value

def run_all_tests_integer(seq, n_range=10):
    """运行所有整数序列检验"""
    print("\n========== 整数序列随机性检验结果 ==========")
    # 1. 卡方检验
    p_chi2 = chi_square_test(seq, n_range)
    print(f"卡方拟合优度检验: P-value = {p_chi2:.6f}  {'✓ 通过' if p_chi2 >= 0.01 else '✗ 不通过'}")
    # 2. 游程检验（升降）
    p_runs = runs_test_integer(seq)
    print(f"游程检验 (升降): P-value = {p_runs:.6f}  {'✓ 通过' if p_runs >= 0.01 else '✗ 不通过'}")
    # 3. 扑克检验（m=3）
    p_poker = poker_test_integer(seq, m=3)
    print(f"扑克检验 (m=3): P-value = {p_poker:.6f}  {'✓ 通过' if p_poker >= 0.01 else '✗ 不通过'}")
    # 4. 自相关检验（lag=1）
    p_auto = autocorrelation_test_integer(seq, lag=1)
    print(f"自相关检验 (lag=1): P-value = {p_auto:.6f}  {'✓ 通过' if p_auto >= 0.01 else '✗ 不通过'}")
    passed = sum([p_chi2 >= 0.01, p_runs >= 0.01, p_poker >= 0.01, p_auto >= 0.01])
    print(f"\n测试总结: {passed}/4 项测试通过。")

# ==================== 蒙特卡罗法求π ====================
def monte_carlo_pi(num_points=100000):
    inside = 0
    for _ in range(num_points):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside += 1
    pi_estimate = 4 * inside / num_points
    error = abs(pi_estimate - np.pi)
    print(f"\n=== 蒙特卡罗法估算π ===")
    print(f"点数: {num_points}, 估算π = {pi_estimate:.6f}, 误差 = {error:.6f}")

# ==================== 主程序 ====================
if __name__ == "__main__":
    print("正在生成随机数序列...")
    seq = generate_sequence(100000, 10)

    print("\n========== 频数统计 ==========")
    counts = frequency_histogram(seq, 10)
    print(f"各数字出现次数: {counts}")

    interval_analysis(seq, 10)

    run_all_tests_integer(seq, 10)

    monte_carlo_pi(100000)