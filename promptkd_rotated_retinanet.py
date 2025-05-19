# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ROTATED_DETECTORS
from .promptkd_single_stage import PromptKDSingleStage
import torch.nn.functional as F
from mmdet.core import images_to_levels, multi_apply, unmap
import torch
import torch.nn as nn
import numpy as np 
import cv2
from PIL import Image
@ROTATED_DETECTORS.register_module()
class PromptKDRotatedRetinaNet(PromptKDSingleStage):
    """Implementation of Rotated `RetinaNet.`__

    __ https://arxiv.org/abs/1708.02002
    """
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        
        stu_x = self.extract_feat(img)
        tea_x = self.teacher.extract_feat(img)
        
        stu_cls_scores, stu_bbox_preds, stu_cls_hold, stu_reg_hold = \
            multi_apply(self.forward_prompt_single, stu_x, module=self)
        tea_cls_scores, tea_bbox_preds, tea_cls_hold, tea_reg_hold = \
            multi_apply(self.forward_prompt_single, tea_x,module=self.teacher)
        
        reused_cls_scores, reused_bbox_preds  = multi_apply(
            self.reuse_teacher_head, stu_cls_hold,stu_reg_hold,
            tea_cls_hold,tea_reg_hold,img=img)   
        
        losses = self.loss_by_feat(tea_cls_scores, tea_bbox_preds, tea_x,
                                   stu_cls_scores, stu_bbox_preds, stu_x,
                                   reused_cls_scores, reused_bbox_preds,
                                   img,tea_cls_hold,
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
                        tea_x,
                        masklist=masklist)
        
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
    
    def forward_prompt_feat_single(self,
                        stu_cls_score,
                        tea_cls_score,
                        stu_feature_adap,
                        tea_x,
                        masklist):
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

    
    
    def forward_prompt_single(self, x, module):
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
        cls_score = module.bbox_head.retina_cls(cls_feat)
        bbox_pred = module.bbox_head.retina_reg(reg_feat)
        return cls_score, bbox_pred, cls_feat_hold, reg_feat_hold
    
    def reuse_teacher_head(self, stu_cls_feat,stu_reg_feat,tea_cls_feat,tea_reg_feat,img):
        reused_cls_feat = self.align_scale(stu_cls_feat, tea_cls_feat)
        reused_reg_feat = self.align_scale(stu_reg_feat, tea_reg_feat)
        if self.reused_teacher_head_idx != 0:
            reused_cls_feat_stu = F.relu(reused_cls_feat)
            reused_reg_feat_stu = F.relu(reused_reg_feat)
        module_tea = self.teacher.bbox_head
        for i in range(self.reused_teacher_head_idx, module_tea.stacked_convs):
            reused_cls_feat_stu = module_tea.cls_convs[i](reused_cls_feat_stu)
            reused_reg_feat_stu = module_tea.reg_convs[i](reused_reg_feat_stu)
        reused_cls_score = module_tea.retina_cls(reused_cls_feat_stu)
        reused_bbox_pred = module_tea.retina_reg(reused_reg_feat_stu)
        return reused_cls_score, reused_bbox_pred 
    
    
    def align_scale(self, stu_feat, tea_feat):
        N, C, H, W = stu_feat.size()
        stu_feat = stu_feat.permute(1, 0, 2, 3).reshape(C, -1)
        stu_mean = stu_feat.mean(dim=-1, keepdim=True)
        stu_std = stu_feat.std(dim=-1, keepdim=True)
        stu_feat = (stu_feat - stu_mean) / (stu_std + 1e-6)
        
        tea_feat = tea_feat.permute(1, 0, 2, 3).reshape(C, -1)
        tea_mean = tea_feat.mean(dim=-1, keepdim=True)
        tea_std = tea_feat.std(dim=-1, keepdim=True)
        tea_mean  =(tea_mean ).mean(dim=1, keepdim=True)
        stu_feat = stu_feat * (tea_std) + tea_mean
        return stu_feat.reshape(C, N, H, W).permute(1, 0, 2, 3)
    
    
    def loss_by_feat(
            self,
            tea_cls_scores,
            tea_bbox_preds,
            tea_feats,
            cls_scores,
            bbox_preds,
            feats,
            reused_cls_scores,
            reused_bbox_preds,
            img,tea_cls_hold,
            
            gt_labels,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore = None):
        
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.bbox_head.anchor_generator.num_levels
        device = cls_scores[0].device
        
        anchor_list, valid_flag_list = self.bbox_head.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.bbox_head.cls_out_channels if self.bbox_head.use_sigmoid_cls else 1
        cls_reg_targets = self.bbox_head.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        avg_factor = (
            num_total_pos + num_total_neg if self.bbox_head.sampling else num_total_pos)
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        concat_anchor_list = []
        for i, _ in enumerate(anchor_list):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)
        
        losses_cls, losses_bbox = multi_apply(
            self.bbox_head.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=avg_factor)
        losses = dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
    
        losses_cls_kd, _ = multi_apply(
            self.pred_mimicking_loss_single,
            reused_cls_scores,
            tea_cls_scores,
            labels_list,
            label_weights_list,
            avg_factor=avg_factor)
        losses.update(
            dict(loss_cls_kd=losses_cls_kd))
        
        loss_prompt_dis,_= multi_apply(self.forward_prompt_clss_single,
                                    reused_cls_scores, tea_cls_scores,tea_cls_hold,
                                    labels_list,
                                    label_weights_list,
                                    avg_factor=avg_factor,
                                    img_metas =img_metas,
                                    gt_labels=gt_labels,
                                    imgs=img)
        losses.update(
            dict(loss_prompt_dis=loss_prompt_dis))
        
        return losses

    def forward_prompt_clss_single(self, stu_cls_scores, tea_cls_scores, 
                                tea_cls_feat, labels,
                                label_weight, avg_factor, 
                                img_metas,
                                gt_labels,
                                imgs):
    
        clip_images = torch.stack([
            self.preprocess(Image.open(img["filename"])) for img in img_metas
        ]).cuda()
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.clip_model.encode_image(clip_images)
            image_features = F.normalize(image_features, dim=-1)
            text_probs = (100.0 * torch.einsum('bd,nd->bn', image_features, self.text_features)).softmax(dim=-1)
        B, C, H, W = tea_cls_scores.shape
        tea_logits = tea_cls_scores.permute(0,2,3,1).reshape(B, -1,self.bbox_head.cls_out_channels)
        adjusted_tea_logits = tea_logits * text_probs.unsqueeze(1)
        adjusted_tea_logits = adjusted_tea_logits.reshape(-1, self.bbox_head.cls_out_channels)
        adjusted_tea_logits = adjusted_tea_logits * self.logit_scale.exp() + self.bias
        stu_logits = stu_cls_scores.permute(0,2,3,1).reshape(-1, self.bbox_head.cls_out_channels)
        prompt_loss_ce = F.cross_entropy(
            stu_logits,
            F.softmax(adjusted_tea_logits, dim=-1)
        ) * 0.001
        return prompt_loss_ce, prompt_loss_ce

    def pred_mimicking_loss_single(self,
                                    reused_cls_score,
                                    tea_cls_score,
                                    labels,
                                    label_weight,
                                    avg_factor,):
        labels = labels.reshape(-1)
        label_weight = label_weight.reshape(-1)
        reused_cls_score = reused_cls_score.permute(0, 2, 3,1).reshape(-1, self.bbox_head.cls_out_channels)
        loss_cls_kd = self.loss_cls_kd(
            reused_cls_score, labels, label_weight, avg_factor=avg_factor)
        return loss_cls_kd, loss_cls_kd
       