# Objects as Points + HarDNet
Object detection using center point detection:
![](readme/fig2.png)
> [**Objects as Points**](http://arxiv.org/abs/1904.07850)

> [**HarDNet: A Low Memory Traffic Network**](https://arxiv.org/abs/1909.00948)



## Highlights

- **Simple Algorithm:** Object as a point is a simple and elegant approach for object detections, it models an object as a single point -- the center point of its bounding box.

- **Simple Network:** A U-shape HarDNet-85 with Conv3x3, ReLU, bilinear interpolation upsampling, and Sum-to-1 layer normalization comprise the whole network. There is NO dilation/deformable convolution, nor any novel activation function being used.

- **Efficient:** CenterNet-HarDNet85 model achieves *42.5* COCO mAP (test-dev) while running at *38* FPS on an Nvidia GTX-1080Ti GPU.


## Main results

### Object Detection on COCO validation

| Backbone     | Parameters |  mAP / FPS(1080ti) | Flip mAP / FPS| Model |
| :----------: | :--------: | :----------------: | :-----------: | :---: |
| HarDNet85    | 37.2M      | 42.4 / 38 |  43.5 / 20   | [Download](https://ping-chao.com/hardnet/centernet_hardnet85_coco.pth) |

The model was trained with Pytorch 1.5.0 on two V100-32GB GPU for six days. Please see [experiment](experiments/ctdet_coco_hardnet85_2x.sh) for detailed hyperperameters.

HarDNet-85 results (no flipping) on COCO **test-dev2017**:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.425
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.618
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.462
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.224
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.472
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.555
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.341
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.563
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.595
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.373
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.655
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.768
```

## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

## Use CenterNet

For object detection on images/ video, run:

~~~
python demo.py ctdet --demo /path/to/image/or/folder/or/video --arch hardnet_85 --load_model centernet_hardnet85_coco.pth
~~~
We provide example images in `CenterNet_ROOT/images/` (from [Detectron](https://github.com/facebookresearch/Detectron/tree/master/demo)). If set up correctly, the output should look like

<p align="center"> <img src='readme/det1.png' align="center" height="230px"> <img src='readme/det2.png' align="center" height="230px"> </p>

For webcam demo, run     

~~~
python demo.py ctdet --demo webcam --arch hardnet_85 --load_model centernet_hardnet85_coco.pth
~~~

## Real-time Demo on NVidia Xavier

| input Size   |     FPS    |   TRT Model   |
| :----------: |  :------:  | :-----------: |
| 512 x 512    |     21     |  [Download](https://ping-chao.com/hardnet/ctdet_hardnet85_trt_512x512.pth) |
| 416 x 320    |     32     |  [Download](https://ping-chao.com/hardnet/ctdet_hardnet85_trt_416x320.pth) |

- Install Jetpack 4.3
- Run follows to install [Torch2TRT](https://github.com/NVIDIA-AI-IOT/torch2trt) with plugin and the demo with or without the above trt models
~~~
# Install torch2trt:
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
sudo python setup.py install --plugins

# Demo
python demo_trt.py ctdet --demo webcam --arch hardnet_85 --load_trt ctdet_hardnet85_trt_416x320.pth --input_w 416 --input_h 320

# or run any size by converting a new trt model:
python demo_trt.py ctdet --demo webcam --arch hardnet_85 --load_model centernet_hardnet85_coco.pth --input_w 640 --input_h 480
~~~

## Benchmark Evaluation and Training

After [installation](readme/INSTALL.md), follow the instructions in [DATA.md](readme/DATA.md) to setup the datasets. Then check [GETTING_STARTED.md](readme/GETTING_STARTED.md) to reproduce the results in the paper.
We provide scripts for all the experiments in the [experiments](experiments) folder.


## License, and Other information

Please see [original CenterNet repo](https://github.com/xingyizhou/CenterNet)
  

## Citation

For CenterNet Citation, please see [original CenterNet repo](https://github.com/xingyizhou/CenterNet)

For HarDNet Citation, please see [HarDNet repo](https://github.com/PingoLH/Pytorch-HarDNet)
