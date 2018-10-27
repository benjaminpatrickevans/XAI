from helpers import read_data
from src.forest import EvolutionaryForest

data_x, data_y = read_data("data/balloons.data")

print(data_x)

forest = EvolutionaryForest()
forest.fit(data_x, data_y)