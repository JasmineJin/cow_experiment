import numpy as np 
import matplotlib.pyplot as plt
import os 

big_points = np.load('roc_big_points.npz')
baseline_points = np.load('roc_baseline_points.npz')

plt.figure()
plt.title('ROC on Multiple Points')
plt.plot(big_points['pf'], big_points['pd'], 'r')
plt.plot(baseline_points['pf'], baseline_points['pd'], 'b')
plt.legend(['Neural Network', 'Baseline'])

big_mixed = np.load('roc_big_mixed.npz')
baseline_mixed = np.load('roc_baseline_mixed.npz')

plt.figure()
plt.title('ROC on Mixed Objects')
plt.plot(big_mixed['pf'], big_mixed['pd'], 'r')
plt.plot(baseline_mixed['pf'], baseline_mixed['pd'], 'b')
plt.legend(['Neural Network', 'Baseline'])
plt.show()