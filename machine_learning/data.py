import numpy as np

class Data:
    def __init__(self, dataset=None):
        if dataset:
            self.dataset = np.load(dataset)
        
    def generate_data(self, data_range, num_rows):
        x1 = np.random.uniform(-data_range, data_range, num_rows)
        x2 = np.random.uniform(-data_range, data_range, num_rows)
        y_add = np.add(x1, x2)
        y_sub = np.subtract(x1, x2)
        y_mul = np.multiply(x1, x2)
        y_div = np.divide(x1, x2)

        self.dataset = np.column_stack((x1, x2, y_add, y_sub, y_mul, y_div))
        np.save('datasets/train_dataset.npy', self.dataset)
    
    def get_x(self):
        return self.dataset[:, [0, 1]]
    
    def get_y_add(self):
        return self.dataset[:, 2]
    
    def get_y_sub(self):
        return self.dataset[:, 3]

    def get_y_mul(self):
        return self.dataset[:, 4]

    def get_y_div(self):
        return self.dataset[:, 5]