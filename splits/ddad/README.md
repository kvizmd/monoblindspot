# DDAD

`splits/ddad` is a split based on train-val published by the official, the static frames are removed by a preprocessing similar to that of SfMLearner.
`splits/ddad/bs` is a split excluding those that do not have enough frames in before or after.

## Directory Structure
Download DDAD.tar from [the official GitHub](https://github.com/TRI-ML/DDAD) and put it as follows

```
└─ ddad/
   └─ ddad_train_val/
      ├─ 000000/
      │  ├─ calibration/
      │  ├─ point_cloud/
      │  ├─ rgb/
      │  │  ├─ CAMERA_01
      │  │  │  ├─ 15621787638931470.jpg
      │  │  │  ├─ 15621787639931422.jpg
      │  │  │  ├─ 15621787640931502.jpg
      │  │  │  ├─ 15621787641931452.jpg
      │  │  │  └─ ...
      │  │  ├─ CAMERA_05
      │  │  ├─ CAMERA_06
      │  │  ├─ CAMERA_07
      │  │  ├─ CAMERA_08
      │  │  └─ CAMERA_09
      │  └─ scene_5c5616b2e583b5fb4f25013580172d1df43b8a31.json
      ├─ 000001/
      ├─ 000002/
      ├─ 000003/
      ├─ 000004/
      ├─ ..
      ├─ LICENSE.md
      └─ ddad.json
```

## Preparation
We uses the same conversion step as [monodepth2](https://github.com/nianticlabs/monodepth2#-kitti-training-data).
The following command converts the DDAD png images into jpeg images and **removes the original images**.

```shell
$ find ddad/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```
