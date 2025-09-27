import numpy as np
from sko.PSO import PSO
import matplotlib.pyplot as plt

# دالة الهدف
def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2

# تهيئة PSO
pso = PSO(func=demo_func, n_dim=3, pop=40, max_iter=150,
          lb=[0, -1, 0.5], ub=[1, 1, 1], w=0.8, c1=0.5, c2=0.5)

# تشغيل الخوارزمية
pso.run()
print('PSO - best_x:', pso.gbest_x, '\nPSO - best_y:', pso.gbest_y)

# رسم تطور القيمة عبر الأجيال
plt.plot(pso.gbest_y_hist)
plt.title("PSO Convergence")
plt.xlabel("Iteration")
plt.ylabel("Best fitness")
plt.show()
