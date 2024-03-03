# nuScenes
 `splits/ithaca365` is a split based on our own split, the static frames are removed by a preprocessing similar to that of [SfMLearner](https://github.com/tinghuiz/SfMLearner).

### Directory Structure
Download the entire Full dataset (v2.2) from [the official site](https://ithaca365.mae.cornell.edu/) and put it in place as follows

```
└─ ithaca365/
   ├─ data/
   │  ├─ 01-16-2022/
   │  │  └─ cam0/
   │  │     ├─ 1642366844221210.jpg
   │  │     ├─ 1642367378229195.jpg
   │  │     ├─ 1642366844321188.jpg
   │  │     ├─ 1642367378329215.jpg
   │  │     └─ ...
   │  └─ 01-17-2022/
   └─ v2.2
      ├─ attribute.json
      ├─ calibrated_sensor.json
      ├─ category.json
      ├─ ego_pose.json
      └─ ...
```
