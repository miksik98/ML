from scipy.interpolate import make_interp_spline, BSpline
import numpy as np
import matplotlib.pyplot as plt


colors = {
    2: 'r',
    4: 'b',
    8: 'g',
    16: 'y',
    32: 'c',
    64: 'm',
    128: 'k',
    256: 'bisque',
    512: 'deepskyblue'
}

for i in [2, 4, 8, 16, 32, 64, 128, 256, 512]:

    x = []
    y = []
    with open('results_{}.csv'.format(i), 'r', encoding='UTF8') as f:
        header_skipped = False
        for line in f.readlines():
            if line.strip() != '' and header_skipped:
                x.append(float(line.split(',')[0]))
                y.append(float(line.split(',')[1]) / (500 + i))
            if not header_skipped:
                header_skipped = True

        xx = np.linspace(0.1, 0.9, 100)
        print(x)
        spl = make_interp_spline(x, y)
        print(spl)

        plt.plot(xx, spl(xx), colors[i], label='n = {}'.format(i))
plt.grid()
plt.legend(loc='best')
plt.show()