from scipy import stats
import numpy as np
# 示例数据
group1 = [684, 664, 673, 652, 650, 661, 664]
group2 = [420, 411, 408, 493, 405, 410, 330]

# F检验（需手动计算）
t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
print(f"t检验 p值: {p_value:.3f}")
# Levene检验（更稳健）
stat, p_levene = stats.levene(group1, group2)
print(f"Levene检验 p值: {p_levene:.3f}")