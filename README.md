# README

## Setup

Please ensure you have installed the Carla simulator and the Python API.

```shell
git clone https://github.com/Justin900429/carla_diffusion.git
conda create -n carla-diffusion python=3.10 -y
conda activate carla-diffusion
conda install pytorch==2.2.2 torchvision==0.17.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt

# Carla version < 0.9.12
easy_install install {CARLA_ROOT}/PythonAPI/carla/dist/carla-{CARLA_VERSION}-py3.7-linux-x86_64.egg

# Carla version >= 0.9.12
pip install carla=={CARLA_VERSION}
```

Modify the `carla_sh_path` in `config/train_rl.yaml` to yours.

## Data collection

```shell
python misc/data_collect.py --save-path {PLACE_TO_SAVE_DATA} --save-num {NUM_OF_DATA}

# Concrete example
python misc/data_collect.py --save-path data/ --save-num 5000
```

If you would like to collect data under `off-screen` mode, please add the flag `--off-screen`.
