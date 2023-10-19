from classes import Stat, Denombrement, GrowthMonitoring
from functions import read_file, measurement_multiplication, mustache_plot

if __name__ == "__main__":
    measurement1 = Stat("Exercice 1",
                               [18.3, 12.2, 13.9, 18.5, 19.6, 2.8, 12.9, 10.3, 13.7, 17.6, 19.4, 17.3, 15, 19.6, 5.7])
    measurement3 = Stat("Exercice 3",
                               [6, 5, 5, 6, 4, 4, 4, 3, 4, 3, 2, 2, 1, 1])
    measurement2 = Stat("Exercice 2",
                               [0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 0, 0, 1, 1, 2, 2, 2, 3, 3,
                                4, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 0, 1, 1, 1, 2, 2, 2, 3, 3, 5, 0, 1, 1, 1, 2, 2, 2, 3,
                                3, 5, 0, 1, 1, 1, 2, 2, 3, 3, 3, 5, 0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 0, 1, 1, 2, 2, 2, 3,
                                3, 4, 6, 0, 1, 1, 2, 2, 2, 3, 3, 4, 7])
    measurement2.discrete = False
    measurement2.plot()
