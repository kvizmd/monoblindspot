# Cityscapes

`splits/cityscapes` is based on the Cityscapes split of [Zhou et al.](https://github.com/tinghuiz/SfMLearner).
`splits/cityscapes/bs` is a split excluding those that do not have enough frames in before or after.

## Directory Structure
Download `leftImg8bit_sequence_trainvaltest.zip (324GB)` and `camera_trainvaltest.zip (2MB)` from [the official site](https://www.cityscapes-dataset.com/) and place them as follows

```
└─ cityscapes/
   ├─ camera/
   │  ├─ aachen/
   │  │  ├─ aachen_000000_000000_camera.json
   │  │  ├─ aachen_000000_000001_camera.json
   │  │  ├─ aachen_000000_000002_camera.json
   │  │  └─ ...
   │  ├─ berlin/
   │  ├─ bielefeld/
   │  ├─ bochum/
   │  ├─ bonn/
   │  ├─ bremen/
   │  └─ ...
   │
   └─ leftImg8bit_sequence/
      ├─ aachen/
      │  ├─ aachen_000000_000000_leftImg8bit.jpg
      │  ├─ aachen_000000_000001_leftImg8bit.jpg
      │  ├─ aachen_000000_000002_leftImg8bit.jpg
      │  └─ ...
      ├─ berlin/
      ├─ bielefeld/
      ├─ bochum/
      ├─ bonn/
      ├─ bremen/
      └─ ...
```

## Preparation
We uses the same conversion step as [monodepth2](https://github.com/nianticlabs/monodepth2#-kitti-training-data).
The following command converts the Cityscapes png images into jpeg images and **removes the original images**.

```shell
$ find cityscapes/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```
