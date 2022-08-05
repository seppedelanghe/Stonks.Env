import numpy as np

from utils import terminal_screen

if __name__ == "__main__":
    terminal_screen(np.random.random((12, 4)), ['a', 'b', 'c', 'd'], 'agent')