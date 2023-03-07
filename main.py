from matplotlib import pyplot as plt
from tqdm import tqdm

from genetics import do_genetics
from utils import get_distance, get_score


def resolve_input(input_file: str, max_loops: int, mutation_rate: float) -> tuple:
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

    with open(input_file, "r") as f:
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
    meaningful_positions: set = set()
    for building_position in tqdm(buildings_positions, desc="Selecting meaningful positions"):
        for x in range(building_position[0] - max_antenna_range, building_position[0] + max_antenna_range):
            max_y: int = abs(building_position[0] + max_antenna_range - x)

            for y in range(-max_y + building_position[1], max_y + building_position[1] + 1):
                if get_distance(building_position, (x, y)) <= max_antenna_range:
                    meaningful_positions.add((x, y))
    meaningful_positions: list = list(meaningful_positions)

    solution, stats = do_genetics(buildings_positions, buildings_speed_score, buildings_latency_score, antennas_range,
                                  antennas_speeds,
                                  reward, meaningful_positions, max_loops, mutation_rate)

    for antenna_id, antenna_position in enumerate(solution):
        antennas_ids.append(antenna_id)
        antennas_positions.append(antenna_position)

    with open(input_file.replace(".in", ".out"), "w") as f:
        f.write(str(len(antennas_ids)) + "\n")
        for antenna_id, antenna_position in zip(antennas_ids, antennas_positions):
            f.write(str(antenna_id) + " " + str(antenna_position[0]) + " " + str(antenna_position[1]) + "\n")

    print(
        f"Completato con score: {get_score(buildings_positions, antennas_positions, buildings_speed_score, buildings_latency_score, antennas_range, antennas_speeds, reward)}")

    from utils import represent_situation
    represent_situation(buildings_positions, antennas_positions, antennas_range)
    return stats


input_files: list = [
    "data/data_scenarios_a_example.in",
    "data/data_scenarios_b_mumbai.in",
    "data/data_scenarios_c_metropolis.in",
    "data/data_scenarios_d_polynesia.in",
    "data/data_scenarios_e_sanfrancisco.in"
    "data/data_scenarios_f_tokyo.in"
]

# Constants
MAX_LOOPS: int = 1000
MUTATION_RATE: float = 0.2

stats: list = []
for input_file in input_files:
    stats.append(resolve_input(input_file, MAX_LOOPS, MUTATION_RATE))

# Plot every element of stats in a subplot
fig, axs = plt.subplots(len(stats), 1)
for i, stat in enumerate(stats):
    axs[i].plot(*stat)

    axs[i].set_xlabel("Generazione")
    axs[i].set_ylabel("Score")

    axs[i].set_title(input_files[i] + " - Score: " + str(stat[1][-1]))
    plt.show()
