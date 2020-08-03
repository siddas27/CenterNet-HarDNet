# Objects as Points + HarDNet
Object detection using center point detection:
![](readme/fig2.png)
> [**Objects as Points**](http://arxiv.org/abs/1904.07850)

> [**HarDNet: A Low Memory Traffic Network**](https://arxiv.org/abs/1909.00948)



## Highlights

- **Simple Algorithm:** Object as a point is simple and elegant, it models an object as a single point -- the center point of its bounding box.

- **Simple Network:** A U-shape HarDNet-85 + Sum-to-1 layer normalization comprise the whole network. There is NO dilationa nor deformable convlution being used.

- **Efficient:** CenterNet-HarDNet85 model achieves *42.4* COCO mAP while running at *38* FPS on an Nvidia GTX-1080Ti GPU.


## Main results

### Object Detection on COCO validation

| Backbone     | Parameters |  mAP(Val) / FPS(1080ti) | Flip mAP / FPS| Model |
|--------------|------------|-------------------|--------------|-------|
| HarDNet85    | 37.2M      | 42.4 / 38 |  43.5 / 20   | [Download](https://ping-chao.com/hardnet/centernet_hardnet85_coco.pth) |

The model was trained with Pytorch1.5.0 on two V100-32GB GPU for six days. Please see [experiment](experiments/ctdet_coco_hardnet85_2x.sh) for detailed hyperperameters.


## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

## Use CenterNet

For object detection on images/ video, run:

~~~
python demo.py ctdet --demo /path/to/image/or/folder/or/video --load_model centernet_hardnet85_coco.pth
~~~
We provide example images in `CenterNet_ROOT/images/` (from [Detectron](https://github.com/facebookresearch/Detectron/tree/master/demo)). If set up correctly, the output should look like

<p align="center"> <img src='readme/det1.png' align="center" height="230px"> <img src='readme/det2.png' align="center" height="230px"> </p>

For webcam demo, run     

~~~
python demo.py ctdet --demo webcam --load_model centernet_hardnet85_coco.pth
~~~

## Benchmark Evaluation and Training

After [installation](readme/INSTALL.md), follow the instructions in [DATA.md](readme/DATA.md) to setup the datasets. Then check [GETTING_STARTED.md](readme/GETTING_STARTED.md) to reproduce the results in the paper.
We provide scripts for all the experiments in the [experiments](experiments) folder.


## License, and Other information

Please see [original CenterNet repo](https://github.com/xingyizhou/CenterNet)
  

## Citation

For CenterNet Citation, please see [original CenterNet repo](https://github.com/xingyizhou/CenterNet)

For HarDNet Citation, please see [HarDNet repo](https://github.com/PingoLH/Pytorch-HarDNet)
