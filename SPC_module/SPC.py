import river
import matplotlib.pyplot as plt


status_dict = {
    "In-control"   : 0,
    "Warning Level": 1,
    "Out-control"  : 2
}

class SPCAlgorithm:
    def __init__(self, init_estimator):
        self.Pmin = 1.0 # initialization
        self.Smin = 0.0 # initialization
        self.num_negative = 0
        self.num_examples = 0
        self.error_rates = [] # for plotting
        self._init_estimator = init_estimator
        self._reset_model()
        self.warn = -1
        self.warning_level = []
        self.drift_level = []
        self.states = []

    def _update(self, y):
        # Update counts
        self.num_examples += 1
        if (y == False):
            self.num_negative += 1
        # Calculate p and s
        p = self.num_negative / self.num_examples
        s = (p * (1 - p) / self.num_examples) ** 0.5
        # Update Pmin and Smin
        if p + s != 0 and p + s < self.Pmin + self.Smin:
            self.Pmin = p
            self.Smin = s
        #print(f"{p}, {s}, {self.Pmin}, {self.Smin}")
        # Check process status
        self.warning_level.append(self.Pmin + 2 * self.Smin)        
        self.drift_level.append(self.Pmin + 3 * self.Smin)
        if p + s < self.Pmin + 2 * self.Smin:
            status = "In-control"
        elif p + s >= self.Pmin + 3 * self.Smin:
            status = "Out-control"
        else:
            status = "Warning Level"
        self.error_rates.append(p)
        return status        

    def _model_train(self, data):
        for i in range(data.shape[0]):
            try:
                x, y = data.iloc[i, :-1], data.iloc[i, -1]
            except: # single sample fitting
                x, y = data[:-1], data[-1]
        self.model.learn_one(x, y)
    
    def _reset_model(self):
        self.model = self._init_estimator()

    def model_control(self, data, sample_id):
        x = data.iloc[sample_id, :-1]
        y = data.iloc[sample_id, -1]
        y_pred = self.model.predict_one(x)

        status = self._update(y_pred==y)        
        # check detector status
        if status == 'Warning Level' and self.warn == -1 and sample_id!=0:
            self.warn = sample_id
        elif status == 'Out-control':
            self._reset_model()
            self._model_train(data.iloc[self.warn:sample_id+1,:])
        else:
            self._model_train(data.iloc[sample_id,:])
        
        self.states.append(status_dict[status])
        return status, y, y_pred

    def process_plot(self):
        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        # Plot
        ax1.plot(range(1, len(self.error_rates) + 1), self.error_rates, marker='o', markersize=1, label='Error Rate')
        ax1.plot(range(1, len(self.warning_level) + 1), self.warning_level, color='r', linestyle='--', label='Warning Level')
        ax1.plot(range(1, len(self.drift_level) + 1), self.drift_level, color='g', linestyle='--', label='Drift Level')
        ax1.set_xlabel('Number of processed samples')
        ax1.set_ylabel('Error rate')
        ax1.set_title('Error Rate Across Processed Samples with SPC Indicators')
        ax1.grid(True)
        ax1.legend()
        # Plotting states
        ax2.plot(range(1, len(self.states) + 1), self.states, marker='o', linestyle='-', color='b')
        ax2.set_xlabel('Number of processed samples')
        ax2.set_ylabel('State')
        ax2.set_title('State Across Processed Samples')
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(['Normal', 'Warning', 'Out of Control'])
        ax2.grid(True)
        plt.tight_layout()
        plt.show()