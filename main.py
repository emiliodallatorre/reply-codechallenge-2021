from matplotlib import pyplot as plt

from genetics import do_genetics
from utils import get_distance, get_score

grid_size: tuple

buildings_count: int
antennas_count: int
reward: int

buildings_positions: list = []
buildings_latency_score: list = []
buildings_speed_score: list = []

antennas_positions: list = []
antennas_ids: list = []
antennas_speeds: list = []
antennas_range: list = []

with open("data/data_scenarios_a_example.in", "r") as f:
    grid_size = tuple(map(int, f.readline().split(" ")))
    buildings_count, antennas_count, reward = tuple(map(int, f.readline().split(" ")))

    for i in range(buildings_count):
        x, y, l, c = tuple(map(int, f.readline().split(" ")))
        buildings_positions.append((x, y))
        buildings_latency_score.append(l)
        buildings_speed_score.append(c)

    for i in range(antennas_count):
        r, s = tuple(map(int, f.readline().split(" ")))
        antennas_speeds.append(s)
        antennas_range.append(r)

# Seleziono le posizioni significative, in modo da impedire di piazzare un'antenna in punti in cui non si collega ad
# alcun edificio
max_antenna_range: int = max(antennas_range)
meaningful_positions: list = []
for x in range(grid_size[0]):
    for y in range(grid_size[1]):
        if not all([get_distance((x, y), building_position) > max_antenna_range for building_position in
                    buildings_positions]):
            meaningful_positions.append((x, y))

solution, stats = do_genetics(buildings_positions, buildings_speed_score, buildings_latency_score, antennas_range,
                              antennas_speeds,
                              reward, meaningful_positions)

for antenna_id, antenna_position in enumerate(solution):
    antennas_ids.append(antenna_id)
    antennas_positions.append(antenna_position)

with open("data/data_scenarios_a_example.out", "w") as f:
    f.write(str(len(antennas_ids)) + "\n")
    for antenna_id, antenna_position in zip(antennas_ids, antennas_positions):
        f.write(str(antenna_id) + " " + str(antenna_position[0]) + " " + str(antenna_position[1]) + "\n")

print(
    f"Completato con score: {get_score(buildings_positions, antennas_positions, buildings_speed_score, buildings_latency_score, antennas_range, antennas_speeds, reward)}")

plt.plot(*stats)
plt.ylabel("Score")
plt.xlabel("Generazione")
plt.show()
