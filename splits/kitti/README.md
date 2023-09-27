# KITTI

`splits/kitti` is based on the [Eigen/Zhou](https://github.com/tinghuiz/SfMLearner) split (also [Monodepth2](https://github.com/nianticlabs/monodepth2) format), which removes static frames.

`splits/kitti/bs` is a split excluding those that do not have enough frames before or after.

## Directory Structure
Download raw data sequences and calibration files from [the official website](https://www.cvlibs.net/datasets/kitti/raw_data.php) and put them as follows.

```
└─ kitti/
   ├─ 2011_09_26/
   │  ├─ 2011_09_26_drive_0001_sync/
   │  │  ├─ image_00/
   │  │  ├─ image_01/
   │  │  ├─ image_02/
   │  │  ├─ image_03/
   │  │  ├─ oxts/
   │  │  ├─ proj_depth/
   │  │  └─ velodyne_points/
   │  ├─ 2011_09_26_drive_0002_sync/
   │  ├─ 2011_09_26_drive_0005_sync/
   │  ├─ 2011_09_26_drive_0009_sync/
   │  ├─ 2011_09_26_drive_0011_sync/
   │  ├─ 2011_09_26_drive_0013_sync/
   │  ├─ 2011_09_26_drive_0014_sync/
   │  └─ ...
   ├─ 2011_09_28/
   ├─ 2011_09_29/
   ├─ 2011_09_30/
   └─ 2011_10_03/
```

To download these files, you can do so in batches based on the list provided by [monodepth2](https://github.com/nianticlabs/monodepth2/blob/master/splits/kitti_archives_to_download.txt). Simply run the following.

```shell
$ wget -i kitti_archives_to_download.txt -P data/kitti
```

## Preparation
We uses the same conversion step as [monodepth2](https://github.com/nianticlabs/monodepth2#-kitti-training-data) for fast training.
The following command converts the KITTI `png` images into `jpeg` images and **removes the original images**.

```shell
$ find data/kitti -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```

## Depth Evaluation
Export the depth map using [export_gt_depth.py](https://github.com/nianticlabs/monodepth2/blob/master/export_gt_depth.py) in monodepth2.

```shell
$ python export_gt_depth.py --data_path /path/to/kitti --split eigen
```

Then, it can be evaluated as the following command.

```shell
$ python evaluate.py \
    --config configs/depth_kitti.yaml \
    --opts \
      DATA.ROOT_DIR data/kitti \
      MODEL.DEPTH.WEIGHT /path/to/depth.pth \
      MODEL.POSE.WEIGHT /path/to/pose.pth
```

## Blind Spot Evaluation
### Annotations
To create annotation data, please use the [labelme](https://github.com/wkentaro/labelme) annotation tool and follow the rules below.

#### Annotation Formats
Annotate with one of the following labels: `blindspot`, `ground`, or `ambiguous`.

|`label`|Available `shape_type`|
|:---|:---|
|`blindspot`|`point`|
|`ground`|`polygon`|
|`ambiguous`|`polygon`, `rectangle`|

`blindspot` is the ground truth of the blind spot position, and there is no limit to the number of one scene.
`ground` is a rough segmentation-like region used to construct a depth map of the ground for evaluation.
`ambiguous` is a region to be ignored during the evaluation.

```json
{
  "version": "5.1.1",
  "shapes": [
    {
      "label": "blindspot",
      "points": [
        [
          560.1,
          201.1
        ]
      ],
      "group_id": null,
      "shape_type": "point",
      "flags": {}
    },
    {
      "label": "blindspot",
      "points": [
        [
          378.173860911271,
          252.98800959232614
        ]
      ],
      "group_id": null,
      "shape_type": "point",
      "flags": {}
    },
    {
      "label": "blindspot",
      "points": [
        [
          687.2004048582996,
          196.3562753036437
        ]
      ],
      "group_id": null,
      "shape_type": "point",
      "flags": {}
    },
    {
      "label": "ground",
      "points": [
        [
          59.54109589041093,
          374.30136986301375
        ],
        ...
      ],
      "group_id": null,
      "shape_type": "polygon",
      "flags": {}
    },
    {
      "label": "ambiguous",
      "points": [
        [
          653.7627118644068,
          201.6949152542373
        ],
        ...
      ],
      "group_id": null,
      "shape_type": "polygon",
      "flags": {}
    },
    {
      "label": "ambiguous",
      "points": [
        [
          747.2663139329807,
          195.40740740740742
        ],
        ...
      ],
      "group_id": null,
      "shape_type": "rectangle",
      "flags": {}
    },
  ],
  "imagePath": "2011_10_03/2011_10_03_drive_0034_sync/image_02/data/0000001215.jpg",
  "imageData": "...",
  "imageHeight": 376,
  "imageWidth": 1241
}
```

The annotated json file is placed at the following location in KITTI dataset.

```
└─ kitti/
   ├─ 2011_09_26/
   ├─ 2011_09_28/
   ├─ 2011_09_29/
   ├─ 2011_09_30/
   └─ 2011_10_03/
      ├─ 2011_10_03_drive_0027_sync/
      ├─ 2011_10_03_drive_0034_sync/
      │  ├─ image_00/
      │  ├─ image_01/
      │  ├─ image_02/
      │  │  ├─ bs/
      │  │  │  ├─ 0000001215.json    <=========== HERE
      │  │  │  ├─ 0000002024.json
      │  │  │  └─ ...
      │  │  ├─ data/
      │  │  └─ timestamps.txt
      │  ├─ image_03/
      │  ├─ oxts/
      │  ├─ proj_depth/
      │  └─ velodyne_points/
      ├─ 2011_10_03_drive_0042_sync/
      ├─ 2011_10_03_drive_0047_sync/
      ├─ calib_cam_to_cam.txt
      ├─ calib_imu_to_velo.txt
      └─ calib_velo_to_cam.txt
```

### Ground Depth Map

#### 1. Prepare the LiDAR point data.
Get the download link of the improved depth map from [here](https://www.cvlibs.net/download.php?file=data_depth_annotated.zip) and download it.
```shell
$ unzip data_depth_annotated.zip
$ cp -r data_depth_annotated/train/* /path/to/kitti
$ cp -r data_depth_annotated/val/* /path/to/kitti
```
#### 2. Generate the depth map of the ground from the LiDAR point cloud.
Export a depth map of the ground using the following script.
```shell
$ python export_ground_depth.py \
    --data_dir /path/to/kitti \
    --split splits/kitti/bs/val_files.txt \
    --out_dir /path/to/output \
    --device cuda:0
```
This script applies RANSAC to the point cloud with refering the annotated mask, whose label is `ground`.

The generated depth maps should be placed in the following directory.

```
└─ kitti/
   ├─ 2011_09_26/
   ├─ 2011_09_28/
   ├─ 2011_09_29/
   ├─ 2011_09_30/
   └─ 2011_10_03/
      ├─ 2011_10_03_drive_0027_sync/
      ├─ 2011_10_03_drive_0034_sync/
      │  ├─ image_00/
      │  ├─ image_01/
      │  ├─ image_02/
      │  │  ├─ bs/
      │  │  ├─ data/
      │  │  ├─ ground_depth/
      │  │  │  ├─ 0000001215.png <=========== HERE
      │  │  │  ├─ 0000002024.png
      │  │  │  └─ ...
      │  │  └─ timestamps.txt
      │  ├─ image_03/
      │  ├─ oxts/
      │  ├─ proj_depth/
      │  └─ velodyne_points/
      ├─ 2011_10_03_drive_0042_sync/
      ├─ 2011_10_03_drive_0047_sync/
      ├─ calib_cam_to_cam.txt
      ├─ calib_imu_to_velo.txt
      └─ calib_velo_to_cam.txt
```
