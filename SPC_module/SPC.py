import river
import matplotlib.pyplot as plt
import pyformulas as pf
import numpy as np


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
        self.ps = [] # debug
        self.ss = [] # debug
        self.Pmins = [] # debug
        self.Smins = [] # debug
        self.window = []

    def _update(self, y):
        # update counts
        self.num_examples += 1
        if (y == False):
            self.num_negative += 1
            
        p = self.num_negative / self.num_examples
        s = (p * (1 - p) / self.num_examples) ** 0.5

        # check process status
        if p + s >= self.Pmin + 3 * self.Smin:
            status = "Out-control"
        elif p + s >= self.Pmin + 2 * self.Smin:
            status = "Warning Level"
        else:
            status = "In-control"
            self.warn = -1 # false alarm error keeps decreasing
                
        self.warning_level.append(self.Pmin + 2 * self.Smin)        
        self.drift_level.append(self.Pmin + 3 * self.Smin)
        self.Pmins.append(self.Pmin)
        self.Smins.append(self.Smin)

        # update Pmin and Smin
        if p+s != 0.0 and p + s < self.Pmin + self.Smin:
            self.Pmin = min(p, self.Pmin) # p
            self.Smin = (self.Pmin * (1 - self.Pmin) / self.num_examples) ** 0.5 # s
            self.warn = -1 # false alarm error keeps decreasing

        return status, p, s

    def _exponential_smoothing_update(self, y, alpha=0.9):
        # update counts
        self.num_examples += 1
        p = (alpha*self.num_negative + (y == False)+0)/self.num_examples
        s = (p * (1 - p) / self.num_examples) ** 0.5
        self.num_negative += (y == False)+0
        
        # check process status
        if p + s >= self.Pmin + 3 * self.Smin:
            status = "Out-control"
        elif p + s >= self.Pmin + 2 * self.Smin:
            status = "Warning Level"
        else:
            status = "In-control"
            self.warn = -1 # false alarm error keeps decreasing
                
        self.warning_level.append(self.Pmin + 2 * self.Smin)        
        self.drift_level.append(self.Pmin + 3 * self.Smin)
        self.Pmins.append(self.Pmin)
        self.Smins.append(self.Smin)

        # update Pmin and Smin
        if p+s != 0.0 and p + s < self.Pmin + self.Smin:
            self.Pmin = min(p, self.Pmin) # p
            self.Smin = (self.Pmin * (1 - self.Pmin) / self.num_examples) ** 0.5 # s
            self.warn = -1 # false alarm error keeps decreasing
        
        return status, p, s
        
    
    def _sliding_window_update(self, y, max_window_size=100):
        # update window
        self.window.append(y)
        if len(self.window) > max_window_size:
            self.window.pop(0)

        # Update counts
        self.num_examples += 1
        if not y:
            self.num_negative += 1

        # Calculate p and s using only the window
        window_size = len(self.window)
        p = self.window.count(False) / window_size if window_size > 0 else 0
        s = (p * (1 - p) / window_size) ** 0.5

        # check process status
        if p + s >= self.Pmin + 3 * self.Smin:
            status = "Out-control"
        elif p + s >= self.Pmin + 2 * self.Smin:
            status = "Warning Level"
        else:
            status = "In-control"
            self.warn = -1  # false alarm error keeps decreasing

        self.warning_level.append(self.Pmin + 2 * self.Smin)
        self.drift_level.append(self.Pmin + 3 * self.Smin)
        self.Pmins.append(self.Pmin)
        self.Smins.append(self.Smin)

        # update Pmin and Smin
        if p + s != 0.0 and p + s < self.Pmin + self.Smin:
            self.Pmin = min(p, self.Pmin)  # p
            self.Smin = (self.Pmin * (1 - self.Pmin) / window_size) ** 0.5  # s
            self.warn = -1  # false alarm error keeps decreasing

        return status, p, s
    

    def _model_train(self, data):
        if len(data.shape) > 1:
            # batch learning
            self.model.learn_many(data.iloc[:, :-1], data.iloc[:, -1])
        else:
            # single sample fitting
            self.model.learn_one(data[:-1], data[-1])
    
    def _reset_model(self):
        try:
            self.models.append(self.model)
        except:
            self.models = [] # first call
        self.model = self._init_estimator()


    def model_control(self, data, sample_id, exponential=False, alpha=0.9, sliding_window=False, window_size=100):
        x = data.iloc[sample_id, :-1]
        y = data.iloc[sample_id, -1]
        y_pred = self.model.predict_one(x)

        if exponential: 
            status, p, s = self._exponential_smoothing_update(y_pred==y, alpha=alpha)
        elif sliding_window:
            status, p, s = self._sliding_window_update(y_pred==y, max_window_size=window_size)
        else:
            status, p, s = self._update(y_pred==y)
            
        to_return = [status, y, y_pred]
        self.error_rates.append(p+s)
        self.ps.append(p)
        self.ss.append(s)
        
        # check detector status
        if status == 'Warning Level' and self.warn == -1:
            self.warn = sample_id
            
        elif status == 'Out-control':
            self._reset_model()
            if self.warn == -1: self.warn = sample_id
            self._model_train(data.iloc[self.warn:sample_id+1,:])

            n, e = 0, 0
            for i in range(self.warn,sample_id+1):
                x = data.iloc[i, :-1]
                y = data.iloc[i, -1]
                y_pred = self.model.predict_one(x)
                n += 1
                e += (y!=y_pred)+0
            print('Num examples post retraining = ', n)
            print('Num negatives post retraining = ', e)
            p = e/n
            s = (p * (1 - p) / n) ** 0.5
            self.ps[-1] = p
            self.ss[-1] = s
            self.num_examples = n
            self.num_negative = e
            self.Pmin = p
            self.Smin = s
            self.Pmins[-1] = p
            self.Smins[-1] = s
            self.warn = -1
            
        else:
            self._model_train(data.iloc[sample_id,:])
        
        self.states.append(status_dict[status])
        return to_return

    def process_plot(self):
        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        # Plot
        ax1.plot(range(1, len(self.error_rates) + 1), self.error_rates, marker='o', markersize=1, label='Error Rate')
        ax1.plot(range(1, len(self.warning_level) + 1), self.warning_level, color='r', linestyle='--', label='Warning Level')
        ax1.plot(range(1, len(self.drift_level) + 1), self.drift_level, color='g', linestyle='--', label='Drift Level')
        ax1.set_xlabel('Number of processed samples')
        ax1.set_ylabel('Error rate')
        ax1.set_title('Error Rate Across Processed Samples with SPC Indicators')
        ax1.grid(True)
        ax1.set_ylim(0.0, max(self.error_rates[100:]))
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



def live_error(error, ranges, drifts, size=100, screen=None, fig=None):
    if screen == None:
        screen = pf.screen(title='Error Evolution')
        fig = plt.figure(figsize=(12, 3))
        
    if len(error) > size:
        error = error[-size:] # plot the most recent error
        ranges = ranges[-size:]

    plt.clf()
    plt.title('Error evolution')
    plt.grid(True)
    plt.ylabel('Error')
    plt.xlabel('Examples')
    plt.xlim(ranges[0], ranges[-1])
    plt.ylim(min(error)-0.001, max(error)+0.001)
    plt.plot(ranges, error, color='black', marker='.', linestyle='--')
    
    real = 15000*np.arange(1, 8) # real drifts
    plt.vlines(x=[d-1 for d in real if d-1 >= ranges[0] and d-1 <= ranges [-1]],
               ymin=min(error)-0.001, ymax=max(error)+0.001, label='Real Drift',
               linestyle='--', color='blue')
    plt.vlines(x=[d-1 for d in drifts if d-1 >= ranges[0] and d-1 <= ranges [-1]],
                   ymin=min(error)-0.001, ymax=max(error)+0.001, label='Detected Drift',
                   linestyle='--', color='red')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.21), ncols=2)
    
    #if [d-1 for d in real if d-1 >= ranges[0] and d-1 <= ranges [-1]] != []:
    #    for point in [(d-1, error[d%500]) for d in real if d-1 >= ranges[0] and d-1 <= ranges [-1]]:
    #        plt.annotate('Real Drift', xy=point, xycoords='axes points', color='blue')
        
    #if [d-1 for d in drifts if d-1 >= ranges[0] and d-1 <= ranges [-1]] != []:
    #    for point in [(d-1, error[d-1]) for d in drifts if d-1 >= ranges[0] and d-1 <= ranges [-1]]:
    #        plt.annotate('Detected Drift', xy=point, xycoords='axes points', color='red')
    plt.tight_layout()
    
    # Draw the figure
    fig.canvas.draw()

    # Convert the figure to an image
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    screen.update(image)
    
    return screen, fig




def error_analysis_plot(spc_detector, baseline, N, E, model_name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    ax1.plot(N[1:], np.array(spc_detector.ps), label='SPC Error')
    ax1.plot(N[1:], np.array(E[1:])/np.array(N[1:]), color='lightgreen', label=f"{model_name}\nwith SPC Error")
    ax1.plot(N[1:], baseline, color='salmon', label=f"Baseline {model_name} Error")

    ax2.plot(N[1:], np.array(spc_detector.ps), label='SPC Error')
    ax2.fill_between(N[1:], np.array(spc_detector.Pmins), np.array(spc_detector.Pmins) + np.array(spc_detector.Smins),
                    alpha=0.3, color='lightgreen', label='In-control')
    ax2.fill_between(N[1:], np.array(spc_detector.Pmins) + np.array(spc_detector.Smins),
                            np.array(spc_detector.Pmins) + 2*np.array(spc_detector.Smins), alpha=0.3, color='gold', label='Warning-level')
    ax2.fill_between(N[1:], np.array(spc_detector.Pmins) + 2*np.array(spc_detector.Smins),
                            np.array(spc_detector.Pmins) + 3*np.array(spc_detector.Smins), alpha=0.3, color='salmon', label='Out-control')
    ax2.vlines([i for i in range(len(spc_detector.states)) if spc_detector.states[i] == 2], ymin=0.1, ymax=0.2,
            color='grey', linestyles='dashed', label='Drift detected')

    ax1.set_ylim(0.1, 0.2)
    ax2.set_ylim(0.1, 0.2)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax2.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax1.set_title(f"SPC and {model_name} Error")
    ax2.set_title('SPC Bounded Error')
    ax1.set_xlabel('Examples')
    ax2.set_xlabel('Examples')
    ax1.set_ylabel('Error')
    ax2.set_ylabel('Error')
    plt.tight_layout()
    plt.show()