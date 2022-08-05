import os
import numpy as np

from typing import List


def unicode_block(s: int):
    assert s > 0 and s < 9, 's to small or too large, needs to be 1-8'
    return [
        u'\u2581',
        u'\u2582',
        u'\u2583',
        u'\u2584',
        u'\u2585',
        u'\u2586',
        u'\u2587',
        u'\u2588',
    ][s-1]


def plot_terminal_graph(x: np.ndarray, name: str, max_size: int = 80):
    h = np.max(x)
    idxs = np.rint(np.linspace(0, x.shape[0] - 1, max_size)).astype('int16')
    line = ''.join([unicode_block(int(x[idx]/h*7)+1) for idx in idxs])
    print(f"{name}:\t", line)

def terminal_screen(values: np.ndarray, labels: List[str], name: str, ncols: int = 5, max_decimals: int = 4, clear: bool = True, color_idx: int = None):
    assert len(labels) == values.shape[1], 'Your labels list is not the same size as your values.'

    if clear:
        os.system('cls' if os.name=='nt' else 'clear')
    
    rows = values.shape[0] // ncols
    for row in range(rows if values.shape[0] % rows == 0 else rows + 1):
        item_idx = row * ncols
        end_item_idx = item_idx+ncols if item_idx+ncols < values.shape[0] else values.shape[0]
        items = values[item_idx:end_item_idx]

        print(f"{name}:" + "".join("\t" for _ in range(2 - len(name) // 8)) + "\t".join([ '\033[44;33m{}\033[m'.format(num) if color_idx != None and color_idx == num-1 else str(num) for num in range(item_idx+1, end_item_idx+1)]))
        for i in range(values.shape[1]):
            print(f"{labels[i]}:" + "".join('\t' for _ in range(2 - len(labels[i]) // 8)) + "\t".join([str(round(x, max_decimals)) for x in items[:, i]]))

        print('\n')