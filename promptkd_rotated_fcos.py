# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ROTATED_DETECTORS

from .promptkd_single_stage import PromptKDSingleStage
import torch.nn.functional as F
from mmdet.core import images_to_levels, multi_apply, unmap,reduce_mean
import torch
import torch.nn as nn
import numpy as np 
import cv2
from PIL import Image

@ROTATED_DETECTORS.register_module()
class PromptKDRotatedFCOS(PromptKDSingleStage):
    """Implementation of Rotated `FCOS.`__

    __ https://arxiv.org/abs/1904.01355
    """
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        
        stu_x = self.extract_feat(img)
        tea_x = self.teacher.extract_feat(img)
        stu_cls_scores, stu_bbox_preds, stu_angle_pred, stu_centerness,stu_cls_hold, stu_reg_hold = \
            multi_apply(self.forward_prompt_single,
                        stu_x,
                        self.bbox_head.scales,
                        self.bbox_head.strides, 
                        module=self)
            
        tea_cls_scores, tea_bbox_preds, tea_angle_pred, tea_centerness,tea_cls_hold, tea_reg_hold = \
            multi_apply(self.forward_prompt_single,
                        tea_x,
                        self.teacher.bbox_head.scales,
                        self.teacher.bbox_head.strides, 
                        module=self.teacher)
            
        reused_cls_scores, reused_bbox_preds,reused_angle_pred,reused_centerness = multi_apply(
            self.reuse_teacher_head, 
            stu_cls_hold,
            stu_reg_hold,
            tea_cls_hold,
            tea_reg_hold,
            self.teacher.bbox_head.scales,
            self.teacher.bbox_head.strides
            )    
        losses = self.loss_by_feat(tea_cls_scores, 
                                   tea_bbox_preds, 
                                   tea_angle_pred,
                                   tea_centerness,
                                   tea_x,
                                   stu_cls_scores,
                                   stu_bbox_preds, 
                                   stu_angle_pred,
                                   stu_centerness,
                                   stu_x,
                                   reused_cls_scores,
                                   reused_bbox_preds,
                                   reused_angle_pred,
                                   reused_centerness,
                                   gt_labels,gt_bboxes,img_metas,gt_bboxes_ignore = None)
        
        stu_feature_adaps = self.stu_feature_adap(stu_x)
        masklist = []
        for i in range(len(gt_bboxes)):
            img_size = (img[i].shape[-2], img[i].shape[-1])
            mask = self.generate_rotated_mask(gt_bboxes[i],img_size=img_size)
            masklist.append(mask)
        
        loss_feat_dis,_ = multi_apply(self.forward_prompt_feat_single,
                        stu_cls_scores,
                        tea_cls_scores,
                        stu_feature_adaps, 
                        tea_x)
        
        masklist = torch.tensor(masklist)
        dffslistlist = [] 
        for j in range(len(tea_x)):
            heatmap  = torch.sum(tea_x[j], dim=1)
            dffslist = 0
            for i in range(masklist.shape[0]): 
                dffs = self.compute_dffs(masklist[i],heatmap[i])
                dffslist +=dffs
            dffslistlist.append(dffslist)
        dffslistlist = torch.tensor(dffslistlist)
        dffslistlistsum = dffslistlist.sum()
        dffslistlist = dffslistlist/dffslistlistsum
        for k in range(len(loss_feat_dis)):
            loss_feat_dis[k] = loss_feat_dis[k] * dffslistlist[k]   
        losses.update(
            dict(loss_feat_kd=loss_feat_dis))    
    
        return losses
    
    def generate_rotated_mask(self,bboxes, img_size):
        mask = np.zeros(img_size, dtype=np.float32)
        for i in range(bboxes.size(0)):
            bbox = bboxes[i].cpu().numpy()
            cx, cy, w, h, angle = bbox
            rect = ((cx, cy), (w, h), angle) 
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)
            cv2.fillPoly(mask, [box_points], 1)
        return mask
    
    def compute_dffs(self, heatmap, mask, alpha=2, beta=0.5, gamma=1, thresh=0.5):
      
        H, W = heatmap.shape
        mask = mask.cpu().numpy()
        mask = cv2.resize(mask, (W, H))  
        H_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
        H_norm_thresh = (H_norm > thresh).float()  
        H_fore = (H_norm * mask) ** alpha * H_norm_thresh 
        fore_score = H_fore.sum()  
        total_score = (H_norm ** alpha).sum() + 1e-6  
        H_back = (H_norm * (1 - mask)) ** gamma * (1 - H_norm_thresh) 
        back_penalty = H_back.sum() / ((1 - mask).sum() + 1e-6)
        dffs = (fore_score / total_score) * torch.exp(-beta * back_penalty)
        return dffs.item()
    
    
    def forward_prompt_feat_single(self,
                        stu_cls_score,
                        tea_cls_score,
                        stu_feature_adap,
                        tea_x):
        dis_loss = 0 
        loss_mse = nn.MSELoss(reduction='sum')
        device = stu_feature_adap.device
        
        N, C, H, W = tea_x.shape
        mat_256x256 = torch.rand((N,1,H,W)).to(device)
        mat_256x256 = torch.where(mat_256x256>1-self.distill_feat_weight, 0, 1).to(device)
        
        masked_fea = torch.mul(stu_feature_adap, mat_256x256)
        new_fea = self.generation(masked_fea)
        dis_loss += loss_mse(new_fea, tea_x)/N *self.alpha
        
        tea_128x128 = self.pooling_128x128(tea_x)
        stu_feature_adap_128x128 = self.pooling_128x128(stu_feature_adap)
        N, C, H, W = tea_128x128.shape
        mat_128x128 = torch.rand((N,1,H,W)).to(device)
        mat_128x128 = torch.where(mat_128x128>1-self.distill_feat_weight, 0, 1).to(device)
        masked_fea_128x128 = torch.mul(stu_feature_adap_128x128, mat_128x128)
        new_fea_128x128 = self.generation(masked_fea_128x128)
        dis_loss += loss_mse(new_fea_128x128, tea_128x128)/N *self.alpha
        
        tea_64x64 = self.pooling_64x64(tea_x)
        stu_feature_adap_64x64 = self.pooling_64x64(stu_feature_adap)
        N, C, H, W = tea_64x64.shape
        mat_64x64 = torch.rand((N,1,H,W)).to(device)
        mat_64x64 = torch.where(mat_64x64>1-self.distill_feat_weight, 0, 1).to(device)
        masked_fea_64x64 = torch.mul(stu_feature_adap_64x64, mat_64x64)
        new_fea_64x64 = self.generation(masked_fea_64x64)
        dis_loss += loss_mse(new_fea_64x64, tea_64x64)/N *self.alpha
    
        
        return dis_loss,dis_loss

    
    def forward_prompt_single(self, x, scale, stride ,module):
        cls_feat, reg_feat = x, x
        cls_feat_hold, reg_feat_hold = x, x
        for i, cls_conv in enumerate(module.bbox_head.cls_convs):
            cls_feat = cls_conv(cls_feat, activate=False)
            if i + 1 == self.reused_teacher_head_idx:
                cls_feat_hold = cls_feat
            cls_feat = cls_conv.activate(cls_feat)
        for i, reg_conv in enumerate(module.bbox_head.reg_convs):
            reg_feat = reg_conv(reg_feat, activate=False)
            if i + 1 == self.reused_teacher_head_idx:
                reg_feat_hold = reg_feat
            reg_feat = reg_conv.activate(reg_feat)
            
        cls_score = module.bbox_head.conv_cls(cls_feat)
        bbox_pred = scale(module.bbox_head.conv_reg(reg_feat)).float()
        
        if module.bbox_head.centerness_on_reg:
            centerness = module.bbox_head.conv_centerness(reg_feat)
        else:
            centerness = module.bbox_head.conv_centerness(cls_feat)
        if module.bbox_head.norm_on_bbox:
            bbox_pred = bbox_pred.clamp(min=0)
            if not module.bbox_head.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        angle_pred = module.bbox_head.conv_angle(reg_feat)
        if module.bbox_head.is_scale_angle:
            angle_pred = module.bbox_head.scale_angle(angle_pred).float()
            
        return cls_score, bbox_pred, angle_pred ,centerness,cls_feat_hold, reg_feat_hold
    
        
    
    def reuse_teacher_head(self, stu_cls_feat,stu_reg_feat,tea_cls_feat,tea_reg_feat,scale, stride):
        # reused_cls_feat = self.align_scale(stu_cls_feat, tea_cls_feat)
        # reused_reg_feat = self.align_scale(stu_reg_feat, tea_reg_feat)
        if self.reused_teacher_head_idx != 0:
            reused_cls_feat_stu = F.relu(stu_cls_feat)
            reused_reg_feat_stu = F.relu(stu_reg_feat)
           
           
        module_tea = self.teacher.bbox_head
        module_stu = self.bbox_head
        for i in range(self.reused_teacher_head_idx, module_tea.stacked_convs):
            reused_cls_feat_stu = module_stu.cls_convs[i](tea_cls_feat)
            reused_reg_feat_stu = module_stu.reg_convs[i](tea_reg_feat)
            
        reused_cls_score = module_tea.conv_cls(reused_cls_feat_stu)
        reused_bbox_pred = scale(module_tea.conv_reg(reused_reg_feat_stu)).float()
        if module_tea.centerness_on_reg:
            reused_centerness = module_tea.conv_centerness(reused_reg_feat_stu)
        else:
            reused_centerness = module_tea.conv_centerness(reused_cls_feat_stu)
            
        if module_tea.norm_on_bbox:
            reused_bbox_pred = reused_bbox_pred.clamp(min=0)
            if not module_tea.training:
                reused_bbox_pred *= stride
        else:
            reused_bbox_pred = reused_bbox_pred.exp()
        reused_angle_pred = module_tea.conv_angle(reused_reg_feat_stu)
        if module_tea.is_scale_angle:
            reused_angle_pred = module_tea.scale_angle(reused_angle_pred).float()
             
        return reused_cls_score, reused_bbox_pred,reused_angle_pred,reused_centerness
     
    def align_scale(self, stu_feat, tea_feat):
        N, C, H, W = stu_feat.size()
        # normalize student feature
        stu_feat = stu_feat.permute(1, 0, 2, 3).reshape(C, -1)
        stu_mean = stu_feat.mean(dim=-1, keepdim=True)
        stu_std = stu_feat.std(dim=-1, keepdim=True)
        stu_feat = (stu_feat - stu_mean) / (stu_std + 1e-6)
        #
        # txt_feat = txt_feats.permute(2,0,1).reshape(C, -1)
        tea_feat = tea_feat.permute(1, 0, 2, 3).reshape(C, -1)
        tea_mean = tea_feat.mean(dim=-1, keepdim=True)
        tea_std = tea_feat.std(dim=-1, keepdim=True)
        # txt_std = txt_feat.std(dim =-1 ,keepdim = True)
        stu_feat = stu_feat * (tea_std) + tea_mean
        return stu_feat.reshape(C, N, H, W).permute(1, 0, 2, 3)
    
     
    def loss_by_feat(
            self,
            tea_cls_scores,
            tea_bbox_preds,
            tea_angle_preds,
            tea_centernesses,
            tea_feats,
            cls_scores,
            bbox_preds,
            angle_preds,
            centernesses,
            feats,
            reused_cls_scores,
            reused_bbox_preds,
            reused_angle_preds,
            reused_centernesses,
            gt_labels,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore = None):
             
        assert len(cls_scores) == len(bbox_preds) \
               == len(angle_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.bbox_head.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        labels, bbox_targets, angle_targets = self.bbox_head.get_targets(
            all_level_points, gt_bboxes, gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.bbox_head.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(-1, 1)
            for angle_pred in angle_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        
        flatten_angle_preds = torch.cat(flatten_angle_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_angle_targets = torch.cat(angle_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])
        
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.bbox_head.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        
        flatten_reused_cls_scores = [
            reused_cls_score.permute(0, 2, 3, 1).reshape(-1, self.bbox_head.cls_out_channels)
            for reused_cls_score in reused_cls_scores
        ]
        
        flatten_reused_cls_scores = torch.cat(flatten_reused_cls_scores)
        losses_cls_kd = self.loss_cls_kd(flatten_reused_cls_scores, 
                                         flatten_labels, 
                                         avg_factor=num_pos)
        
        loss_cls = self.bbox_head.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        
        flatten_reused_bbox_preds = [
            reused_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for reused_bbox_pred in reused_bbox_preds
        ]
        flatten_reused_angle_preds = [
            reused_angle_pred.permute(0, 2, 3, 1).reshape(-1, 1)
            for reused_angle_pred in reused_angle_preds
        ]
        flatten_reused_bbox_preds = torch.cat(flatten_reused_bbox_preds)
        flatten_reused_angle_preds = torch.cat(flatten_reused_angle_preds)
        
        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]
        
        
        pos_centerness_targets = self.bbox_head.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            if self.bbox_head.separate_angle:
                bbox_coder = self.bbox_head.h_bbox_coder
            else:
                bbox_coder = self.bbox_head.bbox_coder
                pos_bbox_preds = torch.cat([pos_bbox_preds, pos_angle_preds],
                                           dim=-1)
                pos_bbox_targets = torch.cat(
                    [pos_bbox_targets, pos_angle_targets], dim=-1)
                
            
            pos_decoded_bbox_preds = bbox_coder.decode(pos_points,
                                                       pos_bbox_preds)
            pos_decoded_target_preds = bbox_coder.decode(
                pos_points, pos_bbox_targets)
            
            loss_bbox = self.bbox_head.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            
            
            if self.bbox_head.separate_angle:
                loss_angle = self.bbox_head.loss_angle(
                    pos_angle_preds, pos_angle_targets, avg_factor=num_pos)
            loss_centerness = self.bbox_head.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
            if self.bbox_head.separate_angle:
                loss_angle = pos_angle_preds.sum()

        if self.bbox_head.separate_angle:
            losses =  dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_angle=loss_angle,
                loss_centerness=loss_centerness)
        else:
            losses =  dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_centerness=loss_centerness) 
        losses.update(
            dict(loss_cls_kd=losses_cls_kd))
        
        loss_prompt_dis,_= multi_apply(self.forward_prompt_clss_single,
                                    reused_cls_scores, tea_cls_scores,
                                    img_metas =img_metas)
                                  
        losses.update(
            dict(loss_prompt_dis=loss_prompt_dis))
        
        return losses
            
    def forward_prompt_clss_single(self, stu_cls_scores, tea_cls_scores, 
                            img_metas,):

        clip_images = torch.stack([
            self.preprocess(Image.open(img["filename"])) for img in img_metas
        ]).cuda()
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.clip_model.encode_image(clip_images)
            image_features = F.normalize(image_features, dim=-1)
            text_probs = (100.0 * torch.einsum('bd,nd->bn', image_features, self.text_features)).softmax(dim=-1)
        B, C, H, W = stu_cls_scores.shape
        stu_logits = stu_cls_scores.permute(0,2,3,1).reshape(B, -1,self.bbox_head.cls_out_channels)
        adjusted_stu_logits = stu_logits * text_probs.unsqueeze(1)
        adjusted_stu_logits = adjusted_stu_logits.reshape(-1, self.bbox_head.cls_out_channels)
        tea_logits = tea_cls_scores.permute(0,2,3,1).reshape(-1, self.bbox_head.cls_out_channels)
        adjusted_stu_logits = adjusted_stu_logits * self.logit_scale.exp() + self.bias
        prompt_loss_ce = F.cross_entropy(
            adjusted_stu_logits,
            F.softmax(tea_logits, dim=-1)
        ) * 0.001
        return prompt_loss_ce, prompt_loss_ce
