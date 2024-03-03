# Monocular Blind Spot Estimation with Occupancy Grid Mapping

The official implementation of [Monocular Blind Spot Estimation with Occupancy Grid Mapping](https://doi.org/10.23919/MVA57639.2023.10215609).

[![test](https://github.com/kvizmd/monoblindspot/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/kvizmd/monoblindspot/actions/workflows/test.yml)

## Installation
The environment is built using Docker. If Docker is not installed on your system, you will need to install the [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). 
If you do not use Docker, please reproduce the environment based on [docker/Dockerfile](docker/Dockerfile).

#### 1. Setup environments.
Build a Docker image based on [docker/Dockerfile](docker/Dockerfile). Here we provide two useful scripts [docker/build_image.sh](docker/build_image.sh) and [docker/run_container.sh](docker/run_container.sh). Each can be run as in the following example.

```shell
$ ./docker/build_image.sh bse-env  # Usage: build_image.sh [image_name]
$ ./docker/run_container.sh bse-env bse /path/to/data /path/to/output  # Usage: run_container.sh [image_name] [container_name] [data_dir] [output_dir]
$ docker exec -it bse bash  # Enter the docker container
```
Once you have restarted the computer, please run `$ docker start bse` and then run `$ docker exec -it bse bash`.

#### 2. Setup dependencies.
Please install the required dependencies using pip. [requirements.txt](./requirements.txt) does not include PyTorch (it is fixed in the Dockerfile with both CUDA versions) and packages for visualization such as Qt are commented in the requirements.txt. 

```shell
$ apt -y update && apt -y upgrade
$ pip install --upgrade pip
$ pip install -r requirements.txt
```
If you use Anaconda, install the necessary packages based on requirements.txt.

## Preparations
Pre-processing is required for training and evaluation with public datasets.
For more details, please refer to the following documents, which also describe directory structure and annotation format.

- [KITTI](splits/kitti)
- [nuScenes](splits/nusc)
- [Cityscapes](splits/cityscapes)
- [DDAD](splits/ddad)
- [Ithaca365](splits/ithaca365)

Currently, there is no support for extensions to custom datasets.

## Training
The entire training pipeline can be executable in `train.py`, and the training target is switched depending on the configuration file passed to the `--config` option.

To configure your settings, such as specifying a path or batch size, please take key/value pairs separated by spaces in the `--opts` option, such as `--opts DATA.ROOT data/kitti DATA.BATCH_SIZE 2 DEVICE cuda:1`.
If you do not want to type long strings on the command line, please create a new configuration file by inheriting the configuration file with the `_BASE_` key in the YAML file as the following example.

```yaml
_BASE_: configs/bsgt_kitti.yaml 
DATA:
    ROOT_DIR: data/kitti
MODEL:
    DEPTH:
        WEIGHT: mlruns/649702204470511544/a981d3c5787c4f7a966f9dbeac565e3c/artifacts/best_checkpoint/depth/state_dict.pth
    POSE:
        WEIGHT: mlruns/649702204470511544/a981d3c5787c4f7a966f9dbeac565e3c/artifacts/best_checkpoint/pose/state_dict.pth
```


For simplicity, the following examples are shown with the `--opts` option.

#### Step1. Train monocular depth estimation network
```shell
$ python train.py \
    --config configs/depth_kitti.yaml
    --opts DATA.ROOT_DIR data/kitti
```

The trained weights are stored in the mlflow `Artifacts` directory, such as `mlruns/649702204470511544/a981d3c5787c4f7a966f9dbeac565e3c/artifacts/best_checkpoint/depth/state_dict.pth`.

#### Step2. Generate training labels using occupancy grid mapping
```shell
$ python train.py \
    --config configs/bsgt_kitti.yaml \
    --opts \
      DATA.ROOT_DIR data/kitti \
      MODEL.DEPTH.WEIGHT /path/to/depth.pth \
      MODEL.POSE.WEIGHT /path/to/pose.pth
```

The JSON files of generated labels are exported into the mlflow `Artifacts` directory, such as `mlruns/281050285719172480/b57166edc1c74975845958bc833e8202/artifacts/json`.

#### Step3. Train blind spot estimation network
```shell
$ python train.py \
    --config configs/bs_kitti_dla34_iam_s8.yaml \
    --opts \
      DATA.ROOT_DIR data/kitti \
      DATA.BS_LABEL /path/to/json
```

## Visualization
The portal of [mlflow](https://mlflow.org/) is available by executing the following command in the root directory of this project.
```shell
$ mlflow ui
```

## Evaluation
Evaluation can be performed in `evaluate.py`, which requires annotated ground truth.

#### Evaluate generated labels

```shell
$ python evaluate.py \
    --config configs/bsgt_kitti.yaml \
    --opts \
      DATA.ROOT_DIR data/kitti \
      MODEL.DEPTH.WEIGHT /path/to/depth.pth \
      MODEL.POSE.WEIGHT /path/to/pose.pth
```

#### Evaluate blind spot estimation network

```shell
$ python evaluate.py \
    --config configs/eval/bs_kitti_dla34_iam_s8.yaml \
    --opts \
      DATA.ROOT_DIR data/kitti \
      MODEL.BS.WEIGHT /path/to/bs.pth
```

## Inference
The trained network can make predictions for any wild image using `infer.py`.

Please specify the configuration file used during training with the `--config` option and the path to an image file or the directory containing some images with the `--input` option. The output files will be exported to the directory specified by `--out_dir`.

```shell
$ python infer.py \
    --config configs/bs_kitti_dla34_iam_s8.yaml \
    --opts MODEL.BS.WEIGHT /path/to/bs.pth
    --input /path/to/input.jpg \
    --out_dir outputs/predictions
```

<!--
## Models
The checkpoints are coming soon.
For code compatibility, these are newly trained weights, not those used in the experiments in the paper.

|**Model**|**FPS**|**Parameter (M)**|**Link**|
|:---:|:---:|:---:|:---:|
|Resnet18-SparseInst-IAM|244.05|16.68|Download|
|DLA34-IAM|127.52|21.15|Download|

**FPS** was measured on NVIDIA TITAN V with `torch.backends.cudnn.benchmark` enabled.
-->

## Development
We employ the `unittest` library for unit tests, which is executable with the following command.

```shell
$ python -m unittest discover tests
```

## Citation
If our project is useful for your research, please consider citing our paper.
```bibtex
@inproceedings{odagiri2023monobse,
  title={Monocular Blind Spot Estimation with Occupancy Grid Mapping},
  author={Kazuya, Odagiri and Kazunori, Onoguchi},
  booktitle={International Conference on Machine Vision and Applications, {MVA}},
  pages={1--6},
  year={2023},
  organization={IEEE}
}
```

## License
Files without a written license are subject to our [MIT license](./LICENSE).   
Files that are licensed by dockstring are not subject to our MIT License, but to their respective licenses.
