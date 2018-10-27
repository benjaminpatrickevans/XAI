from helpers import read_data
from src.base import EvolutionaryBase

data_x, data_y = read_data("data/balloons.data")

print(data_x)

forest = EvolutionaryBase()
forest.fit(data_x, data_y)