from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
from time import perf_counter

import torch
import numpy as np
from torch import nn
from torch2trt import torch2trt
from torch2trt import TRTModule

from opts import opts
from detectors.detector_factory import detector_factory
from pytorch_bn_fusion.bn_fusion import fuse_bn_recursively
from models.networks.hardnet import get_pose_net as get_hardnet
from models.model import load_model
from models.decode import ctdet_decode
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

class CenterHarDNet(nn.Module):
  def __init__(self, num_layers, opt):
    super().__init__()
    heads = {'hm': 80,'wh': 4}
    model = get_hardnet(num_layers=num_layers, heads=opt.heads, head_conv=opt.head_conv, trt=True)
    if opt.load_model:
      model = load_model(model, opt.load_model)
    model = fuse_bn_recursively(model)
    model.v2_transform()
    self.model = model
    mean = np.array(opt.mean, dtype=np.float32).reshape( 1, 3, 1, 1)
    std = np.array(opt.std, dtype=np.float32).reshape( 1, 3, 1, 1)
    self.mean = nn.Parameter(torch.from_numpy(mean))
    self.std = nn.Parameter(torch.from_numpy(std))
    self.max_per_image = 100
    self.num_classes = opt.num_classes
    self.scales = opt.test_scales
    self.opt = opt
    
  def forward(self, x):
    x = ( x/(torch.ones(x.shape, dtype=x.dtype, device=x.device)*255) - self.mean) / self.std
    out = self.model(x)

    hm = torch.sigmoid(out[0])
    wh = out[1]
    reg = wh[:,2:,:,:]
    wh  = wh[:,:2,:,:]
    dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
    
    return dets

    
def show_det(dets, image, det_size, debugger, opt, pause=False):
  dets = dets.view(1, -1, dets.shape[2])
  dets = dets.detach().cpu().numpy()
  h,w = image.shape[0:2]
  c = np.array([w / 2, h / 2], dtype=np.float32)
  s = np.array([w, h], dtype=np.float32)
  dets = ctdet_post_process(
        dets.copy(), [c], [s],
        det_size[1], det_size[0], opt.num_classes)
  for j in range(1, opt.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
  dets = dets[0]  

  debugger.add_img(image, img_id='ctdet')
  for j in range(1, opt.num_classes + 1):
    for bbox in dets[j]:
      if bbox[4] > opt.vis_thresh:
        debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
  debugger.show_all_imgs(pause=pause)



def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  debugger = Debugger(dataset=opt.dataset, ipynb=(opt.debug==3),
                        theme=opt.debugger_theme)

  model = CenterHarDNet(85, opt).cuda().half()
  model.eval()
  
  image_size = (opt.input_w, opt.input_h)
  det_size = [image_size[0]//opt.down_ratio, image_size[1]//opt.down_ratio]

  if opt.load_trt:
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(opt.load_trt))
    model.model = model_trt
  else:
    x = torch.randn((1, 3, image_size[1], image_size[0])).float().cuda().half()
    print('Plase waiting for TensorRT engine converting..')
    with torch.no_grad():
      model_trt = torch2trt(model.model, [x], fp16_mode=True)
    model.model = model_trt
    torch.save(model_trt.state_dict(), 'ctdet_hardnet85_trt_%dx%d.pth'%(image_size[0], image_size[1]))
    print('Done convert')

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FPS, 30)
    
    while True:
        t = perf_counter()
        _, img = cam.read()
        img_x = cv2.resize(img, image_size)

        x = torch.from_numpy(img_x).cuda().half()
        x = x.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        
        torch.cuda.synchronize()
        t2 = perf_counter()
        with torch.no_grad():
          dets = model(x)
        torch.cuda.synchronize()
        t3 = perf_counter()
        show_det(dets, img, det_size, debugger, opt)

        print(' Latency = %.2f ms  (net=%.2f ms)'%((perf_counter()-t)*1000, (t3-t2)*1000))
        if cv2.waitKey(1) == 27:
            return  # esc to quit
  else:
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]
    
    for (image_name) in image_names:
      img = cv2.imread(image_name)
      img_x = cv2.resize(img, image_size)

      x = torch.from_numpy(img_x).cuda().half()
      x = x.unsqueeze(0).permute(0, 3, 1, 2).contiguous()

      with torch.no_grad():
        dets = model(x)
      show_det(dets, img, det_size, debugger, opt, True)
      
if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
