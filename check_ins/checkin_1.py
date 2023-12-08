import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


def make_plots_part_a(data):
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown', 'pink', 'gray', 'lightblue']
    legend = [Patch(color=colors[i], label='MFCC Index: ' + str(i)) for i in range(13)]
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))
    for digit, ax in enumerate(axes.flat):
        data_block0 = data.filter_by_digit(digit).get()[0]
        data_block1 = data.filter_by_digit(digit).get()[1]
        print(np.array(data_block0.mfccs).shape)
        print(np.array(data_block1.mfccs).shape)
        data_block = data_block0
        for mfcc_index in range(13):
            x = len(data_block.mfccs)
            y = [val[mfcc_index] for val in data_block.mfccs]
            ax.plot(range(x), y, label=legend[mfcc_index].get_label(), color=colors[mfcc_index])
            ax.set_title(f'Digit {digit}')
            ax.set_xlabel('Block Index')
            ax.set_ylabel('MFCC Value')
            ax.set_ylim(-10, 6)
    fig.legend(handles=legend, loc='upper right', bbox_to_anchor=(0.05, 0.99), fontsize=6)
    plt.tight_layout()
    plt.show()


def make_plots_part_b(data):
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))
    for digit, ax in enumerate(axes.flat):
        data_block = data.filter_by_digit(digit).get()[0]
        y_1v2 = [val[0] for val in data_block.mfccs]
        x_1v2 = [val[1] for val in data_block.mfccs]
        y_1v3 = [val[0] for val in data_block.mfccs]
        x_1v3 = [val[2] for val in data_block.mfccs]
        y_2v3 = [val[1] for val in data_block.mfccs]
        x_2v3 = [val[2] for val in data_block.mfccs]
        ax.scatter(x_1v2, y_1v2, label='MFCC 1 (y) vs MFCC 2 (x)', s=5)
        ax.scatter(x_1v3, y_1v3, label='MFCC 1 (y) vs MFCC 3 (x)', s=5)
        ax.scatter(x_2v3, y_2v3, label='MFCC 2 (y) vs MFCC 3 (x)', s=5)
        ax.set_title(f'Digit {digit}')
        ax.legend(fontsize=6, loc="best")
    plt.tight_layout()
    plt.show()




