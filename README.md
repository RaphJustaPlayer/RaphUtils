# RaphUtils
My little personal toolbox for my daily work. Typing this allows people to know what I'm doing and to help me if they want to.

It also makes me look like a schyzophrenic person, but that's not the point.

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

`classes.py` contains all the classes I use

`functions.py` contains all the functions I use (this will probably be removed soon)

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

g = GrowthMonitoring('data/{}'.format('Enzymatic monitoring of NH4+.txt'))
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

mol = Stat('data/mols.txt', unit='mol')
vol = Stat('data/volumes.txt', unit='ml')
path = 'data'

print(mol)
# ---------- test ----------
# | - Value: 5.500 ± 2.872 m
# | - Median: 5.500 m
#...
# --------------------------

# mol + vol
# ValueError: the units of the two data are not the same: 'm' and 's'
# Yeah you can't add two values that don't have the same unit, you silly! 

concentration = mol / vol
concentration.name = 'Concentration'

print(concentration)
# ---------- Concentration ----------
# | - Value: 3.315e-02 ± 1.666e-02 mol/ml
# | - Median: 2.889e-02 mol/ml
# ...
# -----------------------------------

concentration.plot(title=True, save=True, path=path)
```
![boxplot](https://github.com/RaphJustaPlayer/RaphUtils/blob/main/data/Boxplot%20of%20concentration.png?raw=true)
![classes](https://github.com/RaphJustaPlayer/RaphUtils/blob/main/data/Class%20distribution%20of%20concentration.png?raw=true)

```python
from raphutils.classes import Stat
import numpy as np

path = 'data'
test = Stat(data_name='test', data_list=np.random.randint(0, 5, 100), unit='m', discrete=True)
print(test)
# ---------- test ----------
# | - Value: 2.020e+00 ± 1.319e+00 m
# | - Median: 2.000e+00 m
# ...
# | - Number of modalities: 100
# | - 3 : 30 -> 30.00%
# | - 4 : 13 -> 13.00%
# | - 0 : 19 -> 19.00%
# | - 1 : 16 -> 16.00%
# | - 2 : 22 -> 22.00%
# --------------------------

test.freq_plot(title=True, save=True, path=path)
```
![freq](https://github.com/RaphJustaPlayer/RaphUtils/blob/main/data/Frequency%20distribution%20of%20test.png?raw=true)


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