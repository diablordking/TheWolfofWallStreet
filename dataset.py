import pandas as pd
import numpy as np


class Dataset:

    def __init__(self, data, class_attr):
        self.class_attr = class_attr
        self.class_values = []
        self.attr_values = {}
        self.col_index = {}
        self.data = None

        self.set_data(data)

    def set_data(self, data):

        data.reset_index()
        col_names = list(data.columns.values)

        self.attr_values = dict.fromkeys(col_names)
        for name in col_names:
            self.attr_values[name] = list(pd.unique(data[name]))

        self.class_values = self.attr_values[self.class_attr]
        del self.attr_values[self.class_attr]

        self.col_index = dict.fromkeys(col_names)
        for idx in range(len(col_names)):
            self.col_index[col_names[idx]] = data.columns.get_loc(col_names[idx])

        self.data = np.array(data.values)

        return

    def data_updating(self):

        attr = list(self.attr_values.keys())

        self.attr_values = {}
        self.attr_values = dict.fromkeys(attr)
        for name in attr:
            col_values = np.array(self.data[:, self.col_index[name]])
            self.attr_values[name] = list(np.unique(col_values))

        self.class_values = []
        col_class_values = np.array(self.data[:, self.col_index[self.class_attr]])
        self.class_values = list(np.unique(col_class_values))

        return

    def get_real_classes(self):

        real_classes = list(self.data[:, self.col_index[self.class_attr]])

        return real_classes
