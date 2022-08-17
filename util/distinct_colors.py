import torch
import numpy as np


class DistinctColors:

    def __init__(self):
        colors = [
            '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#bfef45', '#fabed4', '#469990',
            '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#f032e6', '#ffffff'
        ]
        # 0 = crimson / red, 1 = green, 2 = yellow, 3 = blue
        # 4 = orange, 5 = purple, 6 = sky blue, 7 = lime green
        self.colors = [hex_to_rgb(c) for c in colors]
        self.color_assignments = {}
        self.color_ctr = 0
        self.fast_color_index = torch.from_numpy(np.array([hex_to_rgb(colors[i % len(colors)]) for i in range(8096)] + [hex_to_rgb('#000000')]))

    def get_color(self, index):
        if index not in self.color_assignments:
            self.color_assignments[index] = self.colors[self.color_ctr % len(self.colors)]
            self.color_ctr += 1
        return self.color_assignments[index]

    def get_color_fast_torch(self, index):
        return self.fast_color_index[index]

    def get_color_fast_numpy(self, index):
        return self.fast_color_index[index].numpy()

    def apply_colors(self, arr):
        out_arr = torch.zeros([arr.shape[0], 3])

        for i in range(arr.shape[0]):
            out_arr[i, :] = torch.tensor(self.get_color(arr[i].item()))
        return out_arr

    def apply_colors_fast_torch(self, arr):
        return self.fast_color_index[arr]

    def apply_colors_fast_numpy(self, arr):
        return self.fast_color_index.numpy()[arr]


def hex_to_rgb(x):
    return [int(x[i:i + 2], 16) / 255 for i in (1, 3, 5)]
