import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1, 101)

y = np.exp(0.001 * x) * 10e-2 + np.random.normal(loc=0.0, scale=0.001, size=100)

print(y)
hu = []
su = []
alpha = 0.1
for i in range(len(y)):
    if (i + 1) % 10 == 0 or i==0:
        hu.append(y[i])
    else:
        hu.append(hu[-1])

    if i==0:
        su.append(y[0])
    elif (i + 1) % 2 == 0:
        su.append(y[i] * alpha + su[-1] * (1 - alpha))
    else:
        su.append(su[-1])

su = np.array(su)
hu = np.array(hu)

print(x.shape)
print(hu.shape)
print(su.shape)
    
plt.style.use('dark_background')
plt.plot(x, y, label='Q-Network Parameter', color='red')
plt.plot(x, su, label='Target-Network Parameter (soft update)')
plt.plot(x, hu, label='Target-Network Parameter (hard update)')

plt.xlabel('Action')
plt.ylabel('Parameter')
plt.title('Q-Network and Target-Network corresponding weight through time')
plt.legend()

plt.show()


