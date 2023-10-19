from matplotlib import pyplot as plt
import numpy as np


def mean(data):
    """
    Calculates the mean of a list of numbers
    """
    return sum(data) / len(data)


def std(data):
    """
    Calculates the standard deviation of a list of numbers
    """
    return (sum([(x - mean(data)) ** 2 for x in data]) / len(data)) ** 0.5


def median(dt):
    """
    Calculates the median of a list of numbers
    """
    data = dt.copy()
    data.sort()
    if len(data) % 2 == 0:
        return mean([data[len(data) // 2], data[len(data) // 2 - 1]])
    else:
        return data[len(data) // 2]


def first_quartile(dt):
    """
    Calculates the first quartile of a list of numbers
    """
    data = dt.copy()
    data.sort()
    if len(data) % 2 == 0:
        return median(data[:len(data) // 2])
    else:
        return median(data[:len(data) // 2])


def third_quartile(dt):
    """
    Calculates the third quartile of a list of numbers
    """
    data = dt.copy()
    data.sort()
    if len(data) % 2 == 0:
        return median(data[len(data) // 2:])
    else:
        return median(data[len(data) // 2 + 1:])


def iqr(data):
    """
    Calculates the interquartile range of a list of numbers
    """
    return third_quartile(data) - first_quartile(data)


def outliers(data):
    """
    Calculates the outliers of a list of numbers
    """
    return [x for x in data if x < first_quartile(data) - 1.5 * iqr(data) or x > third_quartile(data) + 1.5 * iqr(data)]


def remove_outliers(data):
    """
    Removes the outliers of a list of numbers
    """
    return [x for x in data if x not in outliers(data)]


def dispersion(data):
    """
    Calculates the dispersion of a list of numbers
    """
    dt = data.copy()
    dt.sort()
    return dt[-1] - dt[0]


def read_file(file):
    """
    Reads a file and returns a list of numbers
    """
    t_data = {}
    k_data = {}
    m_data = []
    with open(file) as f:
        data = f.readlines()
    data = [x.replace('\n', '') for x in data]
    for i, raw in enumerate(data):
        raw = raw.split('\t')
        if len(raw) == 1:
            m_data.append(float(raw[0]))
        else:
            value, time = raw[0], raw[1]
            t_data[eval(time)] = float(value)

    if len(m_data) > 0:
        return m_data
    f_time = list(t_data.keys())[0]
    for i, (time, value) in enumerate(t_data.items()):
        k_data[time - f_time] = value

    return [t for t in k_data.keys()], [v for v in k_data.values()]


def mustache_plot(data, name, hide_outliers=True, vertical=True):
    fig, ax1 = plt.subplots()
    title = 'Boîtes à moustaches des données de\n{}'.format("\n".join(name))
    ax1.set_title(title)

    ax1.boxplot(data, labels=name, showfliers=hide_outliers, vert=vertical)
    fig.tight_layout()
    plt.show()


def timelaps_graph(data, time, name, labels=None):
    fig, ax1 = plt.subplots()
    ax1.set_title(f'Représentation temporelle des données de {name}')
    if labels:
        ax1.set_xlabel(labels[0])
        ax1.set_ylabel(labels[1])
    ax1.plot(time, data)
    fig.tight_layout()
    plt.show()


def measurement_addition(*measurements):
    """
    Adds two measurements together (mean and std)
    """
    units = [m.unit for m in measurements]
    if False in [u == units[0] for u in units]:
        raise ValueError(f"Units are not the same: {', '.join(units)}")
    means = [m.mean for m in measurements]
    stds = [m.std for m in measurements]

    return sum(means), sum(stds), units[0]


def measurement_subtraction(*measurements):
    """
    Adds two measurements together (mean and std)
    """
    units = [m.unit for m in measurements]
    if False in [u == units[0] for u in units]:
        raise ValueError(f"Units are not the same: {', '.join(units)}")
    means = [m.mean for m in measurements]
    stds = [m.std for m in measurements]
    mean = 0
    for m in means:
        mean -= m

    return mean, sum(stds), units[0]


def measurement_multiplication(*measurements):
    """
    Multiplies two measurements together (mean and std)
    """
    units = [m.unit for m in measurements]
    means = [m.mean for m in measurements]
    stds = [m.std for m in measurements]
    std = 0
    for i in range(len(stds)):
        std += stds[i] / means[i]

    return np.prod(means), std * np.prod(means), units_combining(units, '*')


def measurement_division(*measurements):
    """
    Divides two measurements together (mean and std)
    """
    units = [m.unit for m in measurements]
    means = []
    for i, m in enumerate(measurements):
        if i % 2:
            means.append(m.mean ** -1)
        else:
            means.append(m.mean)

    std = 0
    for m in measurements:
        std += m.std / m.mean

    return np.prod(means), std * np.prod(means), units_combining(units, '/')


def uncertainties_formating(mean, std):
    t_mean = f'{mean:.3e}'.split('e')
    t_std = f'{std:.3e}'.split('e')
    f_mean = int(t_mean[1])
    f_std = int(t_std[1])

    if f_mean > f_std:
        return f"({mean * 10 ** -f_mean:.3f} ± {std * 10 ** -f_mean:.3f})e{f_mean}"
    else:
        return f"({mean * 10 ** -f_std:.3f} ± {std * 10 ** -f_std:.3f})e{f_std}"


def units_combining(units, operation):
    if len(units) > 2: raise ValueError("Too many units")  # We can only combine two units at a time
    message = ''

    # If we are adding or subtracting, we don't need to combine the units
    if operation == '+':
        return units[0]
    elif operation == '-':
        return units[0]

    units = [u.split('/') for u in units]  # We split the units into numerator and denominator
    if operation in ['/', '*']:  # If we are multiplying or dividing
        if len(units[0]) == 1 and len(units[1]) == 1:  # If there is no division
            if operation == '/':
                message = units[0][0] + '/' + units[1][0]
            elif operation == '*':
                message = units[0][0] + units[1][0]

        elif len(units[0]) == 1 and len(units[1]) == 2:  # If there is a division in the denominator
            if operation == '/':
                units[1].reverse()
            message = units[0][0] + '.' + units[1][0] + '/' + units[1][1]

        elif len(units[0]) == 2 and len(units[1]) == 1:  # If there is a division in the numerator
            if operation == '/':
                message = units[0][0] + '/' + units[0][1] + units[1][0]
            if operation == '*':
                message = units[0][0] + '.' + units[1][0] + '/' + units[0][1]
        elif len(units[0]) == 2 and len(units[1]) == 2:  # If there is a division in the numerator and denominator
            if operation == '/':
                units[1].reverse()
            message = units[0][0] + '.' + units[1][0] + '/' + units[0][1] + '.' + units[1][1]

    units = message.split('/')
    units = [u.split('.') for u in units]
    _ = units[0].copy()  # We copy the list to avoid modifying it while iterating over it

    for u in _:
        if u in units[1]:
            units[1].remove(u)
            units[0].remove(u)

    message = '.'.join(units[0]) + '/' + '.'.join(units[1])
    if '' in message.split('/'): message = message.split('/')[0]
    return message