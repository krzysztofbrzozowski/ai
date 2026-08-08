## First conv2d activation layer outputs -> each image for one feature map of 32
```python
# Model: "functional"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ input_layer (InputLayer)             │ (None, 180, 180, 3)         │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ rescaling (Rescaling)                │ (None, 180, 180, 3)         │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d (Conv2D)                      │ (None, 178, 178, 32)        │             896 │  <------ HERE
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d (MaxPooling2D)         │ (None, 89, 89, 32)          │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_1 (Conv2D)                    │ (None, 87, 87, 64)          │          18,496 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d_1 (MaxPooling2D)       │ (None, 43, 43, 64)          │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_2 (Conv2D)                    │ (None, 41, 41, 128)         │          73,856 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d_2 (MaxPooling2D)       │ (None, 20, 20, 128)         │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_3 (Conv2D)                    │ (None, 18, 18, 256)         │         295,168 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d_3 (MaxPooling2D)       │ (None, 9, 9, 256)           │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_4 (Conv2D)                    │ (None, 7, 7, 512)           │       1,180,160 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ global_average_pooling2d             │ (None, 512)                 │               0 │
# │ (GlobalAveragePooling2D)             │                             │                 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense (Dense)                        │ (None, 1)                   │             513 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
```
<div style="display: flex; flex-wrap: wrap; gap: 4px;">

  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_0.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_1.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_2.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_3.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_4.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_5.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_6.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_7.png" width="12%">

  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_8.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_9.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_10.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_11.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_12.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_13.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_14.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_15.png" width="12%">

  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_16.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_17.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_18.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_19.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_20.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_21.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_22.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_23.png" width="12%">

  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_24.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_25.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_26.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_27.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_28.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_29.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_30.png" width="12%">
  <img src="imgs/ch_10/feature_maps_first_layer_activation/feature_map_31.png" width="12%">

</div>