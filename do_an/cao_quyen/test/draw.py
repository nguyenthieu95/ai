import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

x = np.arange(0, 2*np.pi, 0.1)
plt.plot(x, np.sin(x))
plt.text(x[len(x)//2], .5, r'$p_c$')
plt.show()