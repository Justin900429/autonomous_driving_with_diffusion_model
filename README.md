# Carla Diffusion

## A. Setup 🚀

Please ensure you have installed the Carla simulator and the Python API.

> [!WARNING]
> You should checkout your own cuda version and install your own PyTorch version. We provide an example of PyTorch 2.2.2 with cuda 12.1.

```shell
git clone https://github.com/Justin900429/carla_diffusion.git
conda create -n carla-diffusion python=3.8 -y
conda activate carla-diffusion
conda install pytorch==2.2.2 torchvision==0.17.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

If you would like to collect data by yourself, please install the Carla Python API and [Carla simulator](https://github.com/carla-simulator/carla):

```shell
# Carla version < 0.9.12
easy_install install {CARLA_ROOT}/PythonAPI/carla/dist/carla-{CARLA_VERSION}-py{CHECK_THIS_VERSION}-linux-x86_64.egg

# Carla version >= 0.9.12
pip install carla=={CARLA_VERSION}
```

Afterwards, please modify the `carla_sh_path` in `config/train_rl.yaml` to yours.

## B. Data setup 📊

>[!NOTE]
>Users can choose to download the provided data or collect the data by themselves.

### (Option 1) Downloading the data 📦

```shell
gdown 1JfHD3bW0oBrjwQJ-nZz5GhVfLN7Nkn8R -O data.zip
unzip -q data.zip && rm data.zip
```

### (Option 2) Collecting the data 📡

```shell
python misc/data_collect.py --save-path {PLACE_TO_SAVE_DATA} --save-num {NUM_OF_DATA}

# Concrete example
python misc/data_collect.py --save-path data/ --save-num 5000
```

> If you would like to collect data under `off-screen` mode, please add the flag `--off-screen`.

If you find the data collection process fail during the simulation, please try the following way (this always run in `off-screen` mode):

```shell
python misc/collect_loop.py --save-path {PLACE_TO_SAVE_DATA} --save-num {NUM_OF_DATA}

# Concrete example
python misc/collect_loop.py --save-path data/ --save-num 5000
```

This helps restart the simulation when the simulation is crashed but the number of data does not reach the target.

## C. Usage 🛠

### C-1. Model training 🧠

Users can choose the config file as shown below to train the model.

| Config path                                 | Description                                                |
| ------------------------------------------- | ---------------------------------------------------------- |
| `configs/default.yaml`                      | Train the model without any guidance. (For ablation study) |
| `configs/guidance/free_guidance.yaml`       | Train the model with classifier-free guidance.             |
| `configs/guidance/classifier_guidance.yaml` | Train the model with classifier guidance.                  |

```shell
# with single-gpu
python train.py --config {CONFIG_PATH}

# with multi-gpus
accelerate launch --multi_gpu --num_processes={NUM_OF_GPU} train.py --config {CONFIG_PATH}
```

### C-2. Interact with the model 🕹

>[!TIP]
> Check the description above to choose the config file.

```shell
python interact.py --config {CONFIG_PATH} --plot-on-world --save-bev-path {PATH_TO_SAVE_BEV_IMAGES} --opts EVAL.CHECKPOINT final.pth

# Concrete example
# 1. without any guidance
python interact.py --config configs/default.yaml --plot-on-world --save-bev-path bev_images  --opts EVAL.CHECKPOINT final.pth

# 2. with classifier-free guidance
python interact.py --config configs/guidance/free_guidance.yaml --plot-on-world --save-bev-path bev_images  --opts EVAL.CHECKPOINT final.pth

# 3. with classifier guidance
python interact.py --config configs/guidance/classifier_guidance.yaml --plot-on-world --save-bev-path bev_images  --opts EVAL.CHECKPOINT final.pth
```

> [!NOTE]
> Both `--plot-on-world` and `--save-bev-path` are optional.

## D. Leaderboard 🏆

### (Optional) D-0. Download the pre-trained model 📦

We provide our pre-trained model for users to evaluate the performance on the Carla leaderboard.

| Model                    | Link                                                                                        |
| :------------------------ | :------------------------------------------------------------------------------------------- |
| Classifier-free guidance | [drive](https://drive.google.com/file/d/12jZFmxaNRq2NhY2cHL95KvydbEuiJH4Y/view?usp=sharing) |

```shell
mkdir checkpoints

# Classifier-free guidance
gdown 12jZFmxaNRq2NhY2cHL95KvydbEuiJH4Y -O checkpoints/free_guidance.pth
```

### D-1. Environment setup

>[!TIP]
> This environment is different from the previous one and requires python 3.7 with Carla 0.9.10.

```shell
conda create -n carla-leaderboard python=3.7 -y
conda activate carla-leaderboard
conda install pytorch==1.13.0 torchvision==0.14.0 -c pytorch
pip install -r requirements-leaderboard.txt
```

### D-2. Run the evaluator

Please check the below table and command to run different benchmarks.

| Benchmark | Scenario file                                    | Route file                                             |
| --------- | ------------------------------------------------ | ------------------------------------------------------ |
| Longest 6 | `leaderboard/data/scenarios/eval_scenarios.json` | `leaderboard/data/evaluation_routes/longest6_tiny.xml` |

```shell
bash leaderboard/scripts/run_evaluation.sh <carla_server_path> <scenario_file> <route_file> <agent_config_file> <save_folder> <save_file>

# Concrete example (take longest 6 as an example)
bash leaderboard/scripts/run_evaluation.sh \
     /path/to/carla/0.9.10/CarlaUE4.sh \
     leaderboard/data/scenarios/eval_scenarios.json \
     leaderboard/data/evaluation_routes/longest6_tiny.xml \
     configs/guidance/free_guidance.yaml \
     free_guidance_longest_6 \
     free_guidance_longest_6/result_longest_6.json
```

>[!IMPORTANT]
> Users should specify the checkpoint directly in the agent config file by setting `EVAL.CHECKPOINT`.

```yaml
...
EVAL:
    CHECKPOINT: /path/to/checkpoint.pth
```

## Acknowledgement 🙏

* Our environment is adapted from [Carla Roach](https://github.com/zhejz/carla-roach).
* The project template is obtained from [deep-learning-template](https://github.com/Justin900429/deep-learning-template).
* The agent for carla leaderboard is adapted from [TCP](https://github.com/OpenDriveLab/TCP).
