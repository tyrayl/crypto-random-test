import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ========== 解决中文显示问题 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC']  # 优先使用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常

def generate_sequence(length=100000, n_range=10):
    return [random.randint(0, n_range-1) for _ in range(length)]

def runs_analysis_visual(seq, max_points=200):
    """
    对整数序列进行游程检验（升降），并可视化前 max_points 个符号的游程模式
    """
    # 生成符号序列：1=上升，-1=下降，忽略相等
    signs = []
    for i in range(1, len(seq)):
        if seq[i] > seq[i-1]:
            signs.append(1)
        elif seq[i] < seq[i-1]:
            signs.append(-1)
        # 相等时跳过，不产生符号

    if len(signs) == 0:
        print("符号序列为空，无法进行游程分析")
        return

    # 计算游程数
    runs = 1
    run_changes = [0]  # 记录每个游程开始的位置（符号索引）
    for i in range(1, len(signs)):
        if signs[i] != signs[i-1]:
            runs += 1
            run_changes.append(i)

    n1 = signs.count(1)   # 上升的次数
    n2 = signs.count(-1)  # 下降的次数
    total = n1 + n2

    # 期望游程数及其方差（经典游程检验公式）
    expected_runs = 2 * n1 * n2 / total + 1
    variance = (expected_runs - 1) * (expected_runs - 2) / (total - 1) if total > 1 else 0
    z = (runs - expected_runs) / np.sqrt(variance) if variance > 0 else 0
    p_value = 2 * (1 - norm.cdf(abs(z)))

    print("\n=== 游程检验详细结果 ===")
    print(f"符号总数（上升+下降）: {total}")
    print(f"上升次数 (n1): {n1}")
    print(f"下降次数 (n2): {n2}")
    print(f"实际游程数: {runs}")
    print(f"期望游程数: {expected_runs:.2f}")
    print(f"方差: {variance:.2f}")
    print(f"Z值: {z:.4f}")
    print(f"P-value: {p_value:.6f}")
    print("结论:", "不通过 (P<0.01)" if p_value < 0.01 else "通过")

    # --- 可视化：前 max_points 个符号及游程分段 ---
    plot_points = min(max_points, len(signs))
    x = range(plot_points)
    y = signs[:plot_points]

    fig, ax = plt.subplots(figsize=(12, 5))

    # 绘制符号点
    ax.plot(x, y, 'o-', markersize=3, linewidth=0.8, color='blue', label='升降符号')

    # 用不同背景色标记不同的游程
    change_in_range = [c for c in run_changes if c < plot_points]
    change_in_range.append(plot_points)

    colors = ['#e6f2ff', '#ffe6e6']
    for idx in range(len(change_in_range)-1):
        start = change_in_range[idx]
        end = change_in_range[idx+1]
        ax.axvspan(start, end, alpha=0.3, color=colors[idx % 2], label='游程区间' if idx==0 else "")

    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel("符号序号（相邻比较步数）")
    ax.set_ylabel("符号值 (1=上升, -1=下降)")
    ax.set_title(f"前{plot_points}个升降符号序列及游程分段\n实际游程数={runs}, 期望={expected_runs:.2f}, P-value={p_value:.4f}")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("runs_visualization.png", dpi=150)
    plt.close()
    print("\n可视化图片已保存为 runs_visualization.png")

    return runs, expected_runs, p_value

if __name__ == "__main__":
    print("正在生成随机整数序列...")
    seq = generate_sequence(100000, 10)

    runs_analysis_visual(seq, max_points=200)

    from random_test_integer import runs_test_integer
    p_original = runs_test_integer(seq)
    print(f"\n原游程检验函数给出的 P-value: {p_original:.6f}")