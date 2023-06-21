# nuScenes

`splits/nusc` is an as-is split of the train-val provided by the official.
It also does not remove static frames as used in the self-supervised monocular depth estimation community.
`splits/nusc/bs` is a split excluding those that do not have enough frames in before or after.

### Directory Structure
Download the entire Full dataset (v1.0) from [the official site](https://www.nuscenes.org/nuscenes#download) and put it in place as follows

```
nuscenes/
  maps/
  samples/
    CAM_FRONT/
      n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915243012465.jpg
      n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915243512465.jpg
      ...
  sweeps/
    CAM_FRONT/
  v1.0-test/
    attribute.json
    calibrated_sensor.json
    category.json
    ego_pose.json
    ...
  v1.0-trainval/
```
