import argparse
import glob
import os
import subprocess


def count_current_saved(output_dir):
    if not os.path.exists(output_dir):
        return 0
    else:
        save_front = len(list(glob.glob(os.path.join(output_dir, "front/*.png"))))
        save_bev = len(list(glob.glob(os.path.join(output_dir, "bev/*.png"))))
        save_waypoints = len(list(glob.glob(os.path.join(output_dir, "waypoints/*.txt"))))
        return min(save_front, save_bev, save_waypoints)


def collect_loop(num_to_collect, output_dir):
    cur_num = count_current_saved(output_dir)

    while cur_num < num_to_collect:
        # Do something
        process = subprocess.Popen(
            [
                "python",
                "misc/data_collect.py",
                "--save-path",
                output_dir,
                "--save-num",
                str(num_to_collect),
                "--off-screen",
            ]
        )
        while process.poll() is None:
            pass
        cur_num = count_current_saved(output_dir)
        print(f"Current collected: {cur_num}/{num_to_collect}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-num", type=int, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    args = parser.parse_args()
    collect_loop(args.save_num, args.save_path)
