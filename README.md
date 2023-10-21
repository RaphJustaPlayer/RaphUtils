# RaphUtils
My little personal toolbox for my daily work

## Installation
First thing first, clone the repo
```bash
git clone https://github.com/RaphJustaPlayer/RaphUtils
```

Then, install the requirements

```bash
pip install -r requirements.txt
```
## Content

It is divided in 2 files:
: `classes.py` contains all the classes I use

: `functions.py` contains all the functions I use

I'm not happy with this architecture, and I'm going to change it soon

## Classes
### `GrowthMonitoring`
This class is used to monitor the growth of a bacterian population through the scope of enzymatic activity. 
It is used to plot the growth curve of the population and to calculate the growth rate of the population.

This is the constructor of the class. It takes six arguments:
- `data_name`(str): the name of the data
- `data_list`(list): the data of the growth curve 
- `time`(list): the time of the growth curve

You can have the growth rate and the doubling time for every timepoint by printing the object.
You also can iterate through the data and get its length.
You alse can plot the evolution of the growth rate and the doubling time.
```python
from raphutils.classes import GrowthMonitoring
from raphutils.functions import read_file

file_name = 'Enzymatic monitoring of NH4+.txt'
t, d = read_file('data/{}'.format(file_name))

g = GrowthMonitoring(file_name[:-4], d, t)
print(g)
# ---------- Enzymatic monitoring of NH4+ ----------
# ...
# --------------------------------------------------

for time, dt in g:
    print(time, dt)
# 0 0.0
# 1 0.0.1
# ...

print(len(g))
# 10

g.plot(smoothing=False, smoothing_val=10, title=None)
# Plot the growth curve
```

You have to build your files like
```txt
data1    time1
data2    time2
...
dataN    timeN
```
Time is always in minutes and can be writen like `652` or `1 * 60 + 12`.
`time1` will be substracted to all the timepoints, so you can write your current time in the file.

### `Stat`

Just an easy way to get everything you need from a list of data. 
Take a name, a data list and can take a unit. You can presice if it's discrete or a continuous data (by default, it's continuous).

You can power, multiply, divide, add or substract two `Stat` objects!
```python
from raphutils.classes import Stat

s1 = Stat('test', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], unit='m', discrete=True)
s2 = Stat('test', [11, 12, 13, 14, 15, 16, 17, 18, 19, 20], unit='s', discrete=True)

print(s1)
# ---------- test ----------
# | - Value: 5.500 ± 2.872 m
# | - Median: 5.500 m
#...
# --------------------------

s1 + s2
# ValueError: the units of the two data are not the same: 'm' and 's'
# Yeah you can't add two values that don't have the same unit, you silly! 

s1 * s2

print(s1)
# ---------- test ----------
# | - Value: 93.500 ± 60.754 m.s
# | - Median: 85.500 m.s
# ...
# --------------------------
# Yup it combines units too! Sometimes it doesn't work, so check the unit of the result.

s1.plot()
# Will automatically choose the right plot for the data, but you can use
# s1.freq_plot() for discrete data
# s1.classes_plot() for continuous data
# The plots aren't gorgeous, but they do the job. I'll improve them later.
```

### `Counting`

Used for counting colonies on a plate. Give it a name, a list of data and a unit (by default, it's `cfu`).

```python
from raphutils.classes import Counting

dilutions = {
    -4: None,
    -5: 354,
    -6: 35,
    -7: 3
}
d = Counting('test', dilutions)
print(d)
# ------------- Counting of test -------------
# | - Dilution 10^-04: NC CFU
# | - Dilution 10^-05: 354 CFU
# | - Dilution 10^-06: 35 CFU
# | - Dilution 10^-07: 3 CFU
# |
# | - Concentration: (3.520 ± 0.020)e7 CFU/mL
# --------------------------------------------
```