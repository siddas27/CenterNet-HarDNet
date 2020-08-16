# Objects as Points + HarDNet
Object detection using center point detection:
![](readme/fig2.png)
> [**Objects as Points**](http://arxiv.org/abs/1904.07850)

> [**HarDNet: A Low Memory Traffic Network**](https://arxiv.org/abs/1909.00948)



## Highlights

- **Simple Algorithm:** Object as a point is a simple and elegant approach for object detections, it models an object as a single point -- the center point of its bounding box.

- **Simple Network:** A U-shape HarDNet-85 with Conv3x3, ReLU, bilinear interpolation upsampling, and Sum-to-1 layer normalization comprise the whole network. There is NO dilation/deformable convolution, nor any novel activation function being used.

- **Efficient:** CenterNet-HarDNet85 model achieves *43.6* COCO mAP (test-dev) while running at *45* FPS on an Nvidia GTX-1080Ti GPU.


## Main results

### Object Detection on COCO validation

| Backbone     | Parameters | GFLOPs | Input Size |  mAP / FPS(1080ti) | Flip mAP / FPS| Model |
| :----------: | :--------: | :----: | :--------: | :----------------: | :-----------: | :---: |
| HarDNet85    | 37.2M      |  87.9  |  512x512   | 43.5 / 45 |  44.4 / 24   | [Download](https://ping-chao.com/hardnet/centernet_hardnet85_coco.pth) |
| HarDNet85    | 37.2M      |  58.0  |  416x416   | 41.5 / 53 |  42.5 / 31   | as above |

The model was trained with Pytorch 1.5.0 on two V100-32GB GPU for six days. Please see [experiment](experiments/ctdet_coco_hardnet85_2x.sh) for detailed hyperperameters.

HarDNet-85 results (no flipping) on COCO **test-dev2017**:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.436
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.623
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.474
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.227
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.469
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.580
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.351
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.573
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.605
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.374
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.651
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.799
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
| 512 x 512    |     21     | [Download](https://ping-chao.com/hardnet/ctdet_hardnet_85_512x512.trt) |
| 416 x 320    |     32     | [Download](https://ping-chao.com/hardnet/ctdet_hardnet_85_416x320.trt) |

- Install NVIDIA JetPack 4.4
- Pytorch > 1.3 for onnx opset 11 and TensorRT > 7.1
- Run following commands with or without the above trt models
~~~

# Demo
python demo_trt.py ctdet --demo webcam --arch hardnet_85 --load_trt ctdet_hardnet_85_416x320.trt --input_w 416 --input_h 320

# or run with any size (divided by 32) by converting a new trt model:
python demo_trt.py ctdet --demo webcam --arch hardnet_85 --load_model centernet_hardnet85_coco.pth --input_w 480 --input_h 480

# You can also run test on COCO val set with trt model:
python test_trt.py ctdet --arch hardnet_85 --load_trt ctdet_hardnet_85_512x512.trt
~~~

## Benchmark Evaluation and Training

After [installation](readme/INSTALL.md), follow the instructions in [DATA.md](readme/DATA.md) to setup the datasets. Then check [GETTING_STARTED.md](readme/GETTING_STARTED.md) to reproduce the results in the paper.
We provide scripts for all the experiments in the [experiments](experiments) folder.


## License, and Other information

Please see [original CenterNet repo](https://github.com/xingyizhou/CenterNet)
  

## Citation

For CenterNet Citation, please see [original CenterNet repo](https://github.com/xingyizhou/CenterNet)

For HarDNet Citation, please see [HarDNet repo](https://github.com/PingoLH/Pytorch-HarDNet)
