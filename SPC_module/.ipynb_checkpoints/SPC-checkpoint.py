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
        self.models = [] # to plot decision boundaries
        self.p = 0.0 # initialization for exponentially smootinhg update
        self.ps = [] # debug
        self.ss = [] # debug
        self.Pmins = [] # debug
        self.Smins = [] # debug

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
        #self.error_rates.append(p+s)
        return status, p, s

    def update(self, y):
        # Update counts
        self.num_examples += 1
        if (y == False):
            self.num_negative += 1
        # Calculate p and s
        p = self.num_negative / self.num_examples
        s = (p * (1 - p) / self.num_examples) ** 0.5

        # Check process status
        if self.num_examples >= 30:
            if p + s >= self.Pmin + 3 * self.Smin:
                status = "Out-control"
            elif p + s >= self.Pmin + 2 * self.Smin:
                status = "Warning Level"
            else:
                status = "In-control"
                self.warn = -1 # false alarm error keeps decreasing
                
        else: status = "In-control"

        self.warning_level.append(self.Pmin + 2 * self.Smin)        
        self.drift_level.append(self.Pmin + 3 * self.Smin)
        self.Pmins.append(self.Pmin)
        self.Smins.append(self.Smin)

        # Update Pmin and Smin
        if p+s != 0.0 and p + s < self.Pmin + self.Smin:
            self.Pmin = min(p, self.Pmin) # p
            self.Smin = (self.Pmin * (1 - self.Pmin) / self.num_examples) ** 0.5 # s
            self.warn = -1 # false alarm error keeps decreasing

        return status, p, s

    def _exponential_smoothing_update(self, y, alpha=0.999):
        # Update counts
        self.num_examples += 1
        
        # Calculate p and s
        self.p += self.p*alpha + ((y == False)+0) / self.num_examples # current update
        s = (self.p * (1 - self.p) / self.num_examples) ** 0.5
        
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
        self.error_rates.append(p+s)
        return status
        

    def _model_train(self, data):
        if len(data.shape) > 1:
            # batch learning
            #x, y = data.iloc[i, :-1], data.iloc[i, -1]
            try:
                self.model.learn_many(data.iloc[:, :-1], data.iloc[:, -1])
            except:
                self.model.partial_fit(data.iloc[:, :-1], data.iloc[:, -1])
        else:
            # single sample fitting
            x, y = data[:-1], data[-1]
            self.model.learn_one(x, y)
    
    def _reset_model(self):
        try:
            self.models.append(self.model)
        except:
            self.models = [] # first call
        self.model = self._init_estimator()
        #self.Pmin = 1.0 # initialization
        #self.Smin = 0.0 # initialization
        self.num_negative = 0 # initialization
        self.num_examples = 0 # initialization

    def model_control(self, data, sample_id):
        last_retrain = 0
        x = data.iloc[sample_id, :-1]
        y = data.iloc[sample_id, -1]
        y_pred = self.model.predict_one(x)

        status, p, s = self.update(y_pred==y)
        self.error_rates.append(p+s)
        self.ps.append(p)
        self.ss.append(s)
        # check detector status
        if status == 'Warning Level' and self.warn == -1 and sample_id != last_retrain:
            self.warn = sample_id
        elif status == 'Out-control' and sample_id != last_retrain:
            self._reset_model()
            #if self.warn == -1: self.warn
            if self.warn == -1: self.warn = sample_id
            self._model_train(data.iloc[self.warn:sample_id+1,:])

            n, e = 0, 0
            for i in range(self.warn,sample_id+1):
                x = data.iloc[i, :-1]
                y = data.iloc[i, -1]
                y_pred = self.model.predict_one(x)
                n += 1
                e += (y!=y_pred)+0
                #self._update(y_pred==y)
                #self.warning_level = self.warning_level[:-1]
                #self.drift_level = self.drift_level[:-1]
            print('Num examples post retraining = ', n)
            print('Num negatives post retraining = ', e)
            p = e/n
            self.ps[-1] = p
            self.ss[-1] = (p * (1 - p) / n) ** 0.5
            self.num_examples = n
            self.num_negative = e
            self.Pmin = self.ps[-1]
            self.Smin = self.ss[-1]
            self.Pmins[-1] = self.ps[-1]
            self.Smins[-1] = self.ss[-1]
            
            self.warn = -1 # initialization
            last_retrain = sample_id+1
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
        ax1.set_ylim(0.0, max(self.error_rates))
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