from math import log

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from functions import mean, std, median, first_quartile, third_quartile, iqr, outliers, remove_outliers, uncertainties_formating, dispersion, mustache_plot
import pandas as pd


class GrowthMonitoring:
    def __init__(self, data_name, data_list, time):
        self.name = data_name
        self.data = data_list
        self.time = np.array(time)

        self.t_denombrement = [0]
        for i in range(len(data_list) - 1):
            d = (log(self.data[i + 1]) - log(self.data[i])) / (self.time[i + 1] - self.time[i])
            if d < 0:
                d = 0
            self.t_denombrement.append(d)
        self.t_denombrement = np.array(self.t_denombrement)

        self.t_dedoublement = []
        for i in range(len(self.t_denombrement)):
            if self.t_denombrement[i] != 0:
                d = log(2) / self.t_denombrement[i]
            else:
                d = 0
            self.t_dedoublement.append(d)
        self.t_dedoublement = np.array(self.t_dedoublement)

    def __str__(self):
        message = f"\n---------- {self.name} ----------"
        for i, time in enumerate(self.time):
            h = time // 60
            m = time % 60
            if h == 0:
                time = str(m)
            else:
                time = f'{h}h{m} min'
            message += f"\n| - {time} min: µ={self.t_denombrement[i] * 100:.3e}/min et Td={self.t_dedoublement[i]:.0f} min"
        message += '\n'
        message += '-' * len(f"---------- {self.name} ----------")
        return message

    def plot(self, smoothing=1):
        """
        Plots the growth rate and the doubling time of the data
        :param smoothing: the smoothing factor
        :return:
        """
        fig, ax1 = plt.subplots()

        spline = make_interp_spline(self.time, self.t_denombrement)
        time_ = np.linspace(self.time, self.t_denombrement, len(self.time) * smoothing)
        color = 'tab:red'
        ax1.set_xlabel('time (m)')
        ax1.set_ylabel('min^-1', color=color)

        ax1.plot([t[-1] for t in time_],
                 [val[-1] for val in spline(time_)],
                 color=color, label='µ')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        spline = make_interp_spline(self.time, self.t_dedoublement)
        color = 'tab:blue'
        ax2.set_ylabel('min', color=color)  # we already handled the x-label with ax1
        ax2.plot([t[-1] for t in time_], [val[-1] for val in spline(time_)], color=color, label='Td')
        ax2.tick_params(axis='y', labelcolor=color)

        ax1.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
        ax2.legend(loc='upper right')
        ax1.set_title(f'Évolution du taux de croissance et du temps de doublement'
                      f'\n de {self.name.lower()} en fonction du temps')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()


class Stat:
    def __init__(self, data_name, data_list, unit=None, discrete=True):
        self.name = data_name
        self.data = data_list
        self.unit = unit
        self.discrete = discrete

        self.mean = mean(data_list)
        self.std = std(data_list)
        self.median = median(data_list)
        self.first_quartile = first_quartile(data_list)
        self.third_quartile = third_quartile(data_list)
        self.iqr = iqr(data_list)
        self.outliers = outliers(data_list)
        self.data_no_outliers = remove_outliers(data_list)

    def __str__(self):
        message = (f"\n---------- {self.name} ----------"
                   f"\n| - Value: {self.mean:.3f} ± {self.std:.3f} {self.unit if self.unit else ''}"
                   f"\n| - Median: {self.median:.3f} {self.unit if self.unit else ''}"
                   f"\n| - First Quartile: {self.first_quartile:.3f} {self.unit if self.unit else ''}"
                   f"\n| - Third Quartile: {self.third_quartile:.3f} {self.unit if self.unit else ''}"
                   f"\n| - Interquartile Range: {self.iqr:.3f} {self.unit if self.unit else ''}"
                   f"\n| - Outliers: {', '.join([str(x) for x in self.outliers])}"
                   f"\n| - Dispersion: {dispersion(self.data)} {self.unit if self.unit else ''}"
                   f"\n| - Dispersion without outliers: {dispersion(self.data_no_outliers)} {self.unit if self.unit else ''}"
                   f"\n| - Standard error: {self.std / len(self.data) ** 0.5:.3f} {self.unit if self.unit else ''}")
        message += '\n'
        message += '-' * len(f"---------- {self.name} ----------")
        return message

    def freq(self, string=True):
        dt = {}
        for data in self.data:
            if data not in dt:
                dt[data] = 1
            else:
                dt[data] += 1

        if string:
            message = f"\n--- Fréquence de {self.name} ---"
            message += f"\n| - Nombre de données: {len(self.data)}"
            for data in dt:
                message += f"\n| - {data} : {dt[data]} fois soit {dt[data] / len(self.data) * 100:.2f}%"
            message += '\n'
            message += '-' * len(f"---- Fréquence de {self.name} ----")
        else:
            message = dt
        return message

    def plot(self):
        if self.discrete: self.classes_plot()
        else: self.freq_plot()
        mustache_plot([self.data], [self.name])

    def freq_plot(self):
        data = self.freq(string=False)
        x = list(data.keys())
        if self.discrete:
            print('\u001b[31m' + 'WARNING: the data is discrete, the graph may not be accurate.'
                  + '\033[0m')

        x.sort()
        y = [data[key] / len(self.data) * 100 for key in x]
        fig, ax1 = plt.subplots()

        ax1.bar(x, y)
        ax1.set_xlabel(self.unit if self.unit else 'Valeur')
        ax1.set_ylabel('Fréquence (%)')
        ax1.set_title(f'Fréquence des valeurs de {self.name.lower()}')
        plt.show()

    def classes_plot(self):
        sturges = int(1 + log(len(self.data)))  # Sturges' formula
        dt = self.data.copy()  # copy of the data to avoid modifying it
        dt.sort()  # sort the data
        pas = (dt[-1] - dt[0])/sturges  # calculate the step
        y = [dt[0]+pas*i for i in range(sturges+1)]  # calculate the intervals

        x = []
        for i in range(len(y)-1):  # for each interval
            count = 0
            for data in dt:
                if y[i] <= data < y[i+1]:
                    count += 1
            x.append(count)
        x = [i/len(self.data)*100 for i in x]  # calculate the frequency
        # y_mid = [(y[i]+y[i+1])/2 for i in range(len(y)-1)]

        fig, ax1 = plt.subplots()

        # I stole that from stackoverflow and I don't know how it works
        N = len(x)
        df = pd.Series(np.random.randint(10, 50, N), index=np.arange(1, N + 1))

        cmap = plt.cm.tab10
        colors = cmap(np.arange(len(df)) % cmap.N)
        # end of the stealing

        ax1.bar([f"[{y[i]:.2f}, {y[i+1]:.2f}[" for i in range(len(y)-1)],
                x, width=1, color=colors, edgecolor='black', linewidth=1.2)
        ax1.set_xlabel(self.unit if self.unit else 'Valeur')
        ax1.set_ylabel('Fréquence (%)')
        ax1.set_title(f'Fréquence des valeurs de {self.name.lower()}')
        fig.tight_layout()
        plt.show()


class Denombrement:
    def __init__(self, name, dilutions: dict):
        self.dilutions = dilutions
        self.name = name

    def __str__(self):
        message = f"\n------------- Dénombrement de {self.name} -------------"
        for dilution in self.dilutions:
            message += f"\n| - Dilution {10**dilution:.0e} : {self.dilutions[dilution] if self.dilutions[dilution] is not None else 'NC'} UFC"
        m, s = self.get_ufc_per_ml()
        message += f"\n|\n| - Concentration : {uncertainties_formating(m, s)} UFC/mL\n"
        message += "-"*len(f"------------- Dénombrement de {self.name} -------------")
        message = message.replace('1e', '10^')

        return message

    def get_ufc_per_ml(self):
        ufcs = []
        for dilution, ufc in self.dilutions.items():
            if ufc is None:
                continue
            if ufc < 30 or ufc > 600:
                continue
            else:
                ufcs.append(ufc * (10**-dilution))
        return mean(ufcs), std(ufcs)