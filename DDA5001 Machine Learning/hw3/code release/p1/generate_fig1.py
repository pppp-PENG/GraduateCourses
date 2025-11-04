import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
    })
except Exception as e:
    pass

df = pd.read_csv('training_data.csv')
x_data = df['x']
y_data = df['y']

xx = np.arange(-1.5, 1.55, 0.05)
yy = xx**2

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(x_data, y_data, label='data', 
           facecolors='none', edgecolors='#2E8CC9', linewidths=1.5)
ax.plot(xx, yy, linewidth=2, color='#D95319', label='target function')

ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-0.5, 2.5])
ax.set_xticks([-1, 0, 1])

legend_font_properties = {'family': 'Times New Roman', 'size': 20}
ax.legend(prop=legend_font_properties, edgecolor='black')

fig.set_facecolor('white')

ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Times New Roman')
    label.set_fontsize(20)

ax.tick_params(direction='in', width=1.5, length=6)

ax.set_xlabel('$x$', fontsize=25, fontname='Times New Roman')
ax.set_ylabel('$y$', fontsize=25, fontname='Times New Roman')

plt.tight_layout()

# To save the figure, uncomment the line below:
# fig.savefig('figure1.pdf', dpi=300, bbox_inches='tight')

plt.show()

