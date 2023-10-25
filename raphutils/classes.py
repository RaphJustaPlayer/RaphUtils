from math import log

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import pandas as pd

from raphutils.functions import uncertainties_formating, units_combining, box_plot, prettify


class GrowthMonitoring:
    def __init__(self, path=None, data_name=None, data_list=None, time=None):
        """
        :param data_name: the name of the data
        :param data_list: the list of the data
        :param time: the list of the time
        """
        self.path = path
        self.name = data_name
        self.data = np.array(data_list)
        self.time = np.array(time)

        if path is not None:
            with open(path) as f:
                dt = f.readlines()

            dt = [x.replace('\n', '').split('\t') for x in dt]

            try:
                self.data = np.array([float(x[0]) for x in dt])
                self.time = np.array([int(x[1]) for x in dt])
                self.name = path.split('/')[-1].split('.')[0]
            except ValueError:
                raise ValueError('File must contain only numbers')

        self.package = {t: d for t, d in zip(self.time, self.data)}

        self.µ = [0]  # growth rate
        for i in range(len(data_list) - 1):
            d = (log(self.data[i + 1]) - log(self.data[i])) / (
                    self.time[i + 1] - self.time[i])  # calculate the growth rate
            if d < 0:
                d = 0
            self.µ.append(d)
        self.µ = np.array(self.µ)  # convert the list to a numpy array

        self.dt = []  # doubling time
        for i in range(len(self.µ)):
            if self.µ[i] != 0:
                d = log(2) / self.µ[i]  # calculate the doubling time
            else:
                d = 0
            self.dt.append(d)
        self.dt = np.array(self.dt)  # convert the list to a numpy array

    def __str__(self):
        message = f"\n---------- {self.name} ----------"
        for i, time in enumerate(self.time):
            h = time // 60
            m = time % 60
            if h == 0:
                time = str(m)
            else:
                time = f'{h}h{m} min'
            message += f"\n| - {time} min: µ={self.µ[i] * 100:.3e}/min et Td={self.dt[i]:.0f} min"
        message += '\n'
        message += '-' * len(f"---------- {self.name} ----------")
        return message

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.package.items())

    def plot(self, smoothing=False, smoothing_val=10, title=None):
        """
        Plots the growth rate and the doubling time of the data
        :param smoothing: the smoothing factor
        :param smoothing_val: the smoothing value
        :param title: the title of the graph
        :return:
        """
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('time (m)')
        ax1.set_ylabel('min⁻¹', color=color)
        if smoothing:
            spline = make_interp_spline(self.time, self.µ)  # create a spline
            time_ = np.linspace(self.time, self.µ,
                                len(self.time) * smoothing_val)  # create a list of time with the smoothing factor

            ax1.plot([t[-1] for t in time_],
                     [val[-1] for val in spline(time_)],
                     color=color, label='µ')
        else:
            ax1.plot(self.time, self.µ, color=color, label='µ')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate second axes that share the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('min', color=color)  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor=color)
        if smoothing:
            spline = make_interp_spline(self.time, self.dt)
            ax2.plot([t[-1] for t in time_], [val[-1] for val in spline(time_)], color=color, label='Td')
        else:
            ax2.plot(self.time, self.dt, color=color, label='Td')

        if title:
            fig.legend(loc='lower right', fancybox=True, shadow=True)
            ax1.set_title(f'Évolution du taux de croissance et du temps de doublement'
                          f'\n de {self.name.lower()} en fonction du temps')
        else:
            fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=2, fancybox=True, shadow=True)

        fig.tight_layout()  # otherwise, the right y-label is slightly clipped
        plt.show()


class Stat:
    def __init__(self, path=None, data_name=None, data_list=None, unit=None, discrete=False):
        """
        :param path: the path of the file
        :param data_name: the name of the data
        :param data_list: the list of the data
        :param unit: the unit of the data
        :param discrete: if the data is discrete
        """
        self.path = path
        self.name = data_name
        self.data = np.array(data_list)
        self.unit = unit
        self.discrete = discrete

        if path is not None:
            with open(self.path) as f:
                data = f.readlines()

            try:
                self.data = [float(x.replace('\n', '')) for x in data]
                self.name = self.path.split('/')[-1].split('.')[0]
            except ValueError:
                raise ValueError('File must contain only numbers')

        if discrete: self.b_mean = np.mean(self.data)
        else: self.mean = np.average(self.data, weights=[x*self.data[i] for i, x in enumerate(self.data)])
        self.mean = sum(self.data) / (len(self.data) - 1)
        self.b_var = np.var(self.data)
        self.b_std = np.std(self.data)
        self.var = (sum([(x - self.mean) ** 2 for x in self.data]) / (len(self.data) - 1))
        self.std = self.var ** 0.5
        self.median = np.median(self.data)
        self.first_quartile = np.percentile(self.data, 25)
        self.third_quartile = np.percentile(self.data, 75)
        self.iqr = np.percentile(self.data, 75) - np.percentile(self.data, 25)
        self.outliers = np.array([x for x in self.data if
                                  x < self.first_quartile - 1.5 * self.iqr or x > self.third_quartile + 1.5 * self.iqr])
        self.data_no_outliers = np.array([x for x in self.data if x not in self.outliers])

    def __str__(self):
        message = (f"\n---------- {self.name} ----------"
                   f"\n| - Value: {prettify(self.mean)} ± {prettify(self.std)} {self.unit if self.unit else ''}"
                   f"\n| - Biased value: {prettify(self.b_mean)} ± {prettify(self.b_std)} {self.unit if self.unit else ''}"
                   f"\n| - Median: {prettify(self.median)} {self.unit if self.unit else ''}"
                   f"\n| - First Quartile: {prettify(self.first_quartile)} {self.unit if self.unit else ''}"
                   f"\n| - Third Quartile: {prettify(self.third_quartile)} {self.unit if self.unit else ''}"
                   f"\n| - Interquartile Range: {prettify(self.iqr)} {self.unit if self.unit else ''}"
                   f"\n| - Max acceptable value: {prettify(self.third_quartile + 1.5 * self.iqr)} {self.unit if self.unit else ''}"
                   f"\n| - Min acceptable value: {prettify(self.first_quartile - 1.5 * self.iqr)} {self.unit if self.unit else ''}"
                   f"\n| - Outliers: {', '.join([str(x) for x in self.outliers])}"
                   f"\n| - Extent: {prettify(self.extent())} {self.unit if self.unit else ''}"
                   f"\n| - Extent without outliers: {prettify(self.extent(outliers=False))} {self.unit if self.unit else ''}"
                   f"\n| - Dispersion index: {prettify(self.std**2 / self.mean)} {self.unit if self.unit else ''}"
                   f"\n| - Standard error: {prettify(self.std / len(self.data) ** 0.5)} {self.unit if self.unit else ''}")
        message += '\n|'
        if self.discrete: message += self.freq()
        message += '-' * len(f"---------- {self.name} ----------")
        return message

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, item):
        return item in self.data

    def __add__(self, other):
        if type(other) != Stat:
            raise TypeError(f"unsupported operand type(s) for +: 'Stat' and '{type(other)}'")
        elif self.unit != other.unit:
            raise ValueError(f"the units of the two data are not the same: '{self.unit}' and '{other.unit}'")
        data = [x + y for x, y in zip(self.data, other.data)]

        return Stat(data_name=self.name, data_list=data, unit=self.unit, discrete=self.discrete)

    def __sub__(self, other):
        if type(other) != Stat:
            raise TypeError(f"unsupported operand type(s) for -: 'Stat' and '{type(other)}'")
        elif self.unit != other.unit:
            raise ValueError(f"the units of the two data are not the same: '{self.unit}' and '{other.unit}'")
        data = [x - y for x, y in zip(self.data, other.data)]

        return Stat(data_name=self.name, data_list=data, unit=self.unit, discrete=self.discrete)

    def __mul__(self, other):
        if type(other) != Stat:
            raise TypeError(f"unsupported operand type(s) for *: 'Stat' and '{type(other)}'")
        data = [x * y for x, y in zip(self.data, other.data)]
        unit = units_combining([self.unit, other.unit], '*')

        return Stat(data_name=f"{self.name}*{other.name}", data_list=data, unit=unit, discrete=self.discrete)

    def __truediv__(self, other):
        if type(other) != Stat:
            raise TypeError(f"unsupported operand type(s) for /: 'Stat' and '{type(other)}'")
        data = [x / y for x, y in zip(self.data, other.data)]
        unit = units_combining([self.unit, other.unit], '/')

        return Stat(data_name=f"{self.name}*{other.name}", data_list=data, unit=unit, discrete=self.discrete)

    def extent(self, outliers=True):
        """
        Calculates the extent of the data
        """
        if outliers:
            dt = self.data.copy()
        else:
            dt = self.data_no_outliers.copy()
        dt.sort()
        return dt[-1] - dt[0]

    def __copy__(self):
        return Stat(self.name, self.data.copy(), self.unit, self.discrete)

    def freq(self, string=True):
        """
        Calculates the frequency of each modality
        """
        dt = {}
        for data in self.data:
            if data not in dt:
                dt[data] = 1
            else:
                dt[data] += 1

        if string:
            message = f"\n| - Number of modalities: {len(self.data)}"
            for data in dt:
                message += f"\n| - {data} : {dt[data]} -> {dt[data] / len(self.data) * 100:.2f}%"
            message += '\n'
        else:
            message = dt

        return message

    def plot(self, title=False, save=False, path=None):
        if self.discrete:
            self.freq_plot(title=title, save=save, path=path)
        else:
            self.classes_plot(title=title, save=save, path=path)
        box_plot([self.data], [self.name], title=title, save=save, path=path)

    def freq_plot(self, title=False, save=False, path=None):
        """
        Plots the frequency of each modality
        """
        data = self.freq(string=False)
        x = list(data.keys())
        y = list(data.values())
        if not self.discrete:
            print('\u001b[31m' + 'WARNING: the data is continuous, the graph may not be accurate.' + '\033[0m')

        x.sort()
        percentage = [data[key] / len(self.data) * 100 for key in x]
        fig, ax1 = plt.subplots()

        bp = ax1.bar([str(name) for name in x], y)
        ax1.set_xlabel(self.unit)
        ax1.set_ylabel('Workforce')
        if title: ax1.set_title(f'Workforce distribution of {self.name.lower()}')

        for i, rect in enumerate(bp):
            ax1.annotate(f'{percentage[i]:.1f}%',
                         xy=(rect.get_x() + rect.get_width() / 2, percentage[i]),
                         xytext=(0, 1),
                         textcoords="offset points",
                         ha='center', va='bottom')

        fig.tight_layout()
        plt.show()
        if save:
            if save:
                if path is None:
                    path = f'Frequency distribution of {self.name}.png'
                else:
                    path = f'{path}/Frequency distribution of {self.name}.png'
            fig.savefig(path, dpi=600)

    def classes_plot(self, title=False, save=False, path=None):
        """
        Plots the frequency of each class
        """
        failed = 0
        x = [0]
        while 0 in x:  # this is done to avoid having a class with 0 values in it
            # even sturges can be wrong sometimes... or my code is terrible, probably both!
            x = [0]
            sturges = int(1 + log(len(self.data)))-failed  # Sturges' formula
            dt = self.data.copy()  # copy of the data to avoid modifying it
            dt.sort()  # sort the data
            pas = (dt[-1] - dt[0]) / sturges  # calculate the step
            y = [dt[0] + pas * i for i in range(sturges + 1)]  # calculate the intervals

            for i in range(len(y) - 1):  # for each interval
                count = 0
                for data in dt:
                    if y[i] <= data < y[i + 1]:
                        count += 1
                x.append(count)
            x = [i / len(self.data) * 100 for i in x]  # calculate the frequency
            x.pop(0)
            failed += 1

        fig, ax1 = plt.subplots()

        # I stole that from stackoverflow, and I don't know how it works
        # I only use it for putting pretty colors on the graph
        df = pd.Series(np.random.randint(10, 50, len(x)), index=np.arange(1, len(x) + 1))

        cmap = plt.cm.tab10
        colors = cmap(np.arange(len(df)) % cmap.N)
        # end of the stealing

        bp = ax1.bar([f"[{y[i]:.2f}, {y[i + 1]:.2f}[" for i in range(len(y) - 1)],
                     x, width=1, color=colors, edgecolor='black', linewidth=1.2)
        ax1.set_xlabel(self.unit if self.unit else 'Valeur')
        ax1.set_ylabel('Frequency (%)')
        if title: ax1.set_title(f'Class distribution of {self.name.lower()}')

        for rect in bp:
            height = rect.get_height()
            ax1.annotate('{}'.format(height),
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 1),
                         textcoords="offset points",
                         ha='center', va='bottom')

        fig.tight_layout()
        plt.show()
        if save:
            if path is None:
                path = f'Class distribution of {self.name.lower()}.png'
            else:
                path = f'{path}/Class distribution of {self.name.lower()}.png'
            fig.savefig(path, dpi=600)


class Counting:
    def __init__(self, name, dilutions: dict):
        """
        Allow for the calculation of the concentration of a bacteria
        :param name: the name of the data
        :param dilutions: the dilutions of the data
        dilutions = {
            -4: None,
            -5: 354,
            -6: 35,
            -7: 3}
        """
        self.dilutions = dilutions
        self.name = name

    def __str__(self):
        message = f"\n------------- Counting of {self.name} -------------"
        for dilution in self.dilutions:
            message += f"\n| - Dilution {10 ** dilution:.0e}: {self.dilutions[dilution] if self.dilutions[dilution] is not None else 'NC'} CFU"
        m, s = self.get_cfu_per_ml()
        message += f"\n|\n| - Concentration: {uncertainties_formating(m, s)} CFU/mL\n"
        message += "-" * len(f"------------- Counting of {self.name} -------------")
        message = message.replace('1e', '10^')

        return message

    def get_cfu_per_ml(self):
        ufcs = []
        for dilution, ufc in self.dilutions.items():
            if ufc is None:
                continue
            if ufc < 30 or ufc > 600:
                continue
            else:
                ufcs.append(ufc * (10 ** -dilution))
        return np.mean(ufcs), np.std(ufcs)
