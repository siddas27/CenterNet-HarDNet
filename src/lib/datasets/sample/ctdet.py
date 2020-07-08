from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from pycocotools import mask as maskUtils

from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math

class CTDetDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i
  
  
  def img_transform(self, img, anns, flip_en=True):
    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    if self.opt.keep_res:
      input_h = (height | self.opt.pad) + 1
      input_w = (width | self.opt.pad) + 1
      s = np.array([input_w, input_h], dtype=np.float32)
    else:
      s = [img.shape[1], img.shape[0]]
      input_h, input_w = self.opt.input_h, self.opt.input_w

    flipped = False
    rot = 0
    img_s = [img.shape[1], img.shape[0]]
    
    if self.split == 'train':
      s = np.random.choice([
         256, 320, 384, 448, 512, 576, 640, 704, 768, 832])
      sd = np.random.rand()*0.8 - 0.4 + 1
      if img.shape[0] > img.shape[1]:
        s = [s, s*(img.shape[0] / img.shape[1])*sd]
      else:
        s = [s*(img.shape[1] / img.shape[0])*sd, s]

      w_border = self._get_border(128, img.shape[1])
      h_border = self._get_border(128, img.shape[0])
      c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
      c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)

      if flip_en and np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        c[0] =  width - c[0] - 1

    trans_input = get_affine_transform(
      c, img_s, rot, s, [input_w/2, input_h/2])
    inp = cv2.warpAffine(img, trans_input,
                         (input_w, input_h),
                         flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    output_h = input_h // self.opt.down_ratio
    output_w = input_w // self.opt.down_ratio
    out_s = [s[0] * output_w/input_w, s[1]*output_h/input_h ]
    trans_output = get_affine_transform(c, img_s, rot, out_s, [output_w/2,output_h/2])
    
    num_objs = min(len(anns), self.max_objs)
    ann_list = []
    
    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'])
      segm  = ann['segmentation']
      cls_id = int(self.cat_ids[ann['category_id']])
      if flipped:
          bbox[[0, 2]] = width - bbox[[2, 0]] - 1
      bboxi1 = affine_transform(bbox[:2], trans_input)
      bboxi2 = affine_transform(bbox[2:], trans_input)
      bbox[:2] = affine_transform(bbox[:2], trans_output)
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      ann_list.append((bbox, cls_id))
      #end of objs loop
    meta = (c, s)     
    return inp, ann_list, output_w, output_h, meta

    
  def __getitem__(self, index):
    num_classes = self.num_classes
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)
    
    img = cv2.imread(img_path)
    inp, ann_list, output_w, output_h, meta = self.img_transform(img, anns)
    
    img = inp.transpose(1, 2, 0)
    img = (img*self.std + self.mean)*255
      
    
    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    dense_reg = np.zeros((4, output_h, output_w), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    dense_wh_mask = np.zeros((4, output_h, output_w), dtype=np.float32)
    cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
    cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)
    
    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian

    gt_det = []
    xs = np.random.randint(output_w, size=(self.max_objs, 1))
    ys = np.random.randint(output_h, size=(self.max_objs, 1))
    bgs = np.concatenate([xs,ys], axis=1)
    
    for k in range(num_objs):
      bbox, cls_id = ann_list[k]

      oh, ow = bbox[3] - bbox[1], bbox[2] - bbox[0]
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      
      #get center of box
      ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
      ct_int = ct.round().astype(np.int32)

      if (h > 4 or h == oh) and (w > 4 or w == ow):
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        radius = self.opt.hm_gauss if self.opt.mse_loss else radius

        draw_dense_reg(dense_reg, dense_wh_mask, ct_int, bbox, radius)
        draw_gaussian(hm[cls_id], ct_int, radius)

        cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
        cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
                       ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
    
    dense_wh  = dense_reg[:2,:,:]
    dense_off = dense_reg[2:,:,:]

    ret = {'input': inp, 'hm': hm, 'dense_wh': dense_wh, 'dense_off': dense_off, 
           'dense_wh_mask': dense_wh_mask[:2]}
    if self.opt.cat_spec_wh:
      ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
      del ret['wh']
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 6), dtype=np.float32)
      meta = {'c': meta[0], 's': meta[1], 'gt_det': gt_det, 'img_id': img_id}
      ret['meta'] = meta
    return ret
    