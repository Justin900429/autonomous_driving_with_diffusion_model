# Carla Diffusion

## Setup 🚀

Please ensure you have installed the Carla simulator and the Python API.

> [!WARNING]
> You should checkout your own cuda version and install your own PyTorch version. We provide an example of PyTorch 2.2.0 with cuda 12.1.

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

## Data collection 📊

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

This help restart the simulation when the simulation is crashed but the number of data does not reach the target.

## Usage

### Model training 🧠

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

### Interact with the model 🕹

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
