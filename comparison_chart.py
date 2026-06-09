import matplotlib.pyplot as plt
import numpy as np

# 理论均匀分布 vs 实际频数
theoretical = [10000] * 10
actual = [9840, 9980, 10031, 10116, 9830, 10111, 10035, 10025, 10083, 9949]

x = np.arange(10)
width = 0.35
plt.bar(x - width/2, theoretical, width, label='理论均匀分布', color='skyblue')
plt.bar(x + width/2, actual, width, label='实际观测频数', color='orange')
plt.xlabel('数字')
plt.ylabel('出现次数')
plt.title('频数分布对比')
plt.legend()
plt.savefig('comparison.png')
print("对比图已保存为 comparison.png")