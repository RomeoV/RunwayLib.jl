# Camera Model

We use the [pinhole camera model](https://en.wikipedia.org/wiki/Pinhole_camera_model "Wikipedia") without any distortion.
We implement two projection plane reference frames: `:centered` and `:offset`, which are defined as illustrated here:

```@eval
using Typstry
render(typst"""
       #include "./figs/camera_models.typ"
       """; output="camera_models.svg", open=false)
```

![For offset, u point right and v points down. and the origin is in the top left. For centered, u points left and v points up, and the origin is at the image center.](camera_models.svg)

We also support defining the camera model either through [`CameraConfig`](@ref) or through [`CameraMatrix`](@ref).


## `:offset` vs `:centered`
```@example camera_models_A
using RunwayLib, Unitful.DefaultSymbols, Rotations
cam_pos = WorldPoint(-10m, 0m, 0m)
cam_rot = RotZYX(zeros(3)...)
world_pt = WorldPoint(0m, 0m, 0m)

focal_length = 25mm
pixel_size = 5μm/px
camconf_centered = CameraConfig{:centered}(focal_length, pixel_size, 4096.0px, 2048.0px)
project(cam_pos, cam_rot, world_pt, camconf_centered)
```

With an offset camera model:

```@example camera_models_A
camconf_offset = CameraConfig{:offset}(focal_length, pixel_size, 4096.0px, 2048.0px)
project(cam_pos, cam_rot, world_pt, camconf_offset)
```

And for a non-centered point:
```@example camera_models_A
world_pt2 = WorldPoint(0m, 1m, 1m)
project(cam_pos, cam_rot, world_pt2, camconf_centered)
```

```@example camera_models_A
project(cam_pos, cam_rot, world_pt2, camconf_offset)
```

## Reference
```@docs; canonical = false
project
CameraConfig
CameraMatrix
```
