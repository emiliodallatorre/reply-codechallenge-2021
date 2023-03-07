from matplotlib import pyplot as plt


def get_distance(a: tuple, b: tuple) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_score(buildings_positions: list, antennas_positions: list, buildings_speed_score: list,
              buildings_latency_score: list, antennas_range: list, antennas_speeds: list, reward: int) -> int:
    score: int = 0
    are_all_buildings_connected: bool = True

    for building_index, building_position in enumerate(buildings_positions):
        connected_antennas_ids: list = []
        connected_antennas_distances: list = []

        for antenna_index, antenna_position in enumerate(antennas_positions):
            distance: int = get_distance(building_position, antenna_position)
            if distance <= antennas_range[antenna_index]:
                connected_antennas_ids.append(antenna_index)
                connected_antennas_distances.append(distance)

        if len(connected_antennas_ids) > 0:
            max_connection_score: int = 0

            for connected_antennas_index, antenna_id in enumerate(connected_antennas_ids):
                connection_score: int = buildings_speed_score[building_index] * antennas_speeds[
                    antenna_id] - buildings_latency_score[building_index] * connected_antennas_distances[
                                            connected_antennas_index]

                if connection_score > max_connection_score:
                    max_connection_score = connection_score

            score += max_connection_score
        else:
            are_all_buildings_connected = False

    score += reward if are_all_buildings_connected else 0

    return score


def represent_situation(buildings_positions: list, antennas_positions: list, antennas_ranges: list) -> None:
    buildings_positions_x = [building_position[0] for building_position in buildings_positions]
    buildings_positions_y = [building_position[1] for building_position in buildings_positions]
    plt.scatter(buildings_positions_x, buildings_positions_y, color="red", label="Buildings", marker="s")

    antennas_positions_x = [antenna_position[0] for antenna_position in antennas_positions]
    antennas_positions_y = [antenna_position[1] for antenna_position in antennas_positions]
    plt.scatter(antennas_positions_x, antennas_positions_y, color="blue", label="Antennas", marker="P")

    covered_positions_x: list = []
    covered_positions_y: list = []
    for antenna_index, antenna_position in enumerate(antennas_positions):
        for x in range(antenna_position[0] - antennas_ranges[antenna_index], antenna_position[0] + antennas_ranges[
            antenna_index] + 1):
            max_y: int = antennas_ranges[antenna_index] - abs(x - antenna_position[0])

            for y in range(antenna_position[1] - max_y, antenna_position[1] + max_y + 1):

                if (x, y) not in buildings_positions and (x, y) not in antennas_positions:
                    covered_positions_x.append(x)
                    covered_positions_y.append(y)

    plt.scatter(covered_positions_x, covered_positions_y, color="green", label="Covered positions", marker="x")

    plt.legend()
    plt.show()
