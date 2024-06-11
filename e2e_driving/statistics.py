import argparse
import json

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Data Collection")
    parser.add_argument("--json-file", required=True, type=str, help="Path to the json file")
    return parser.parse_args()


def cal_std(score_list):
    total_length = len(score_list)
    run1_list = score_list[::3]
    run2_list = score_list[1::3]
    run3_list = score_list[2::3]

    mean = sum(score_list) / total_length
    mean1 = sum(run1_list) / (total_length / 3)
    mean2 = sum(run2_list) / (total_length / 3)
    mean3 = sum(run3_list) / (total_length / 3)

    std = np.sqrt(((mean1 - mean) ** 2 + (mean2 - mean) ** 2 + (mean3 - mean) ** 2) / 3)
    return std


if __name__ == "__main__":
    args = parse_args()
    score_composed = []
    score_penalty = []
    score_route = []
    with open(args.json_file) as f:
        data = json.load(f)
        records = data["_checkpoint"]["records"]
        for record in records[:15]:
            score_composed.append(record["scores"]["score_composed"])
            score_route.append(record["scores"]["score_route"])
            score_penalty.append(record["scores"]["score_penalty"])

    print("score_composed =", sum(score_composed) / len(score_composed))
    print("score_penalty =", sum(score_penalty) / len(score_penalty))
    print("score_route =", sum(score_route) / len(score_route))

    for item in [
        "collisions_layout",
        "collisions_pedestrian",
        "collisions_vehicle",
        "red_light",
        "stop_infraction",
        "vehicle_blocked",
        "outside_route_lanes",
    ]:
        infraction_score_list = []
        for i in range(3):
            length = 0
            veh_collision = 0
            with open(path) as f:
                data = json.load(f)
                records = data["_checkpoint"]["records"][i::3]
                for record in records:
                    length += (
                        record["scores"]["score_route"]
                        / 100
                        * record["meta"]["route_length"]
                        / 1000
                    )
                    veh_collision += len(record["infractions"][item])
            # print(veh_collision)
            # print(length)
            infraction_score_list.append(veh_collision / length)
        print(item, "=", sum(infraction_score_list) / len(infraction_score_list))
