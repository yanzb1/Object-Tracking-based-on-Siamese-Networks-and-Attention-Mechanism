# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from siamban.core.config import cfg
from siamban.models.loss import select_cross_entropy_loss, select_iou_loss
from siamban.models.backbone import get_backbone
from siamban.models.head import get_ban_head
from siamban.models.neck import get_neck
from siamban.models.neck import get_fearure
from siamban.utils.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2,
                       accuracy)
from siamban.models.neck.position_encoding import PositionEmbeddingSine,build_position_encoding
from siamban.utils import box_ops
from siamban.utils.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2,
                       accuracy)


from ltr.models.loss.matcher import build_matcher
from ltr.models.neck.featurefusion_network import build_featurefusion_network
class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        self.featurefusion_network=get_fearure('featurefusion_network',**cfg.FEATUREFUSION.KWARGS)
        hidden_dim = 256
        in_channels=[256,256,256]
        self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
        self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))
        self.loc_scale = nn.Parameter(torch.ones(len(in_channels)))
        self.class_embed = MLP(hidden_dim, hidden_dim, 1 + 1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(1024, hidden_dim, kernel_size=1)
    def template(self, z):

        if not isinstance(z, NestedTensor):
            z = nested_tensor_from_tensor_2(z)
        zf,pos_template=self.joiner(z)
        #zf,cf, pos_template = self.backbone(z)
        self.zf = zf
        self.pos_template = pos_template

    def track(self, x):
        if not isinstance(x, NestedTensor):
            search = nested_tensor_from_tensor_2(x)
        features_searchs, pos_searchs = self.joiner(search)
        feature_templates = self.zf
        pos_templates = self.pos_template
        cls = []
        loc = []
        for idx, (features_search, pos_search,feature_template,pos_template) in enumerate(zip(features_searchs, pos_searchs,feature_templates,pos_templates), start=2):
            src_search, mask_search = features_search.decompose()
            assert mask_search is not None
            src_template, mask_template = feature_template.decompose()
            assert mask_template is not None

            hs = self.featurefusion_network(self.input_proj(src_template), mask_template, self.input_proj(src_search),
                                        mask_search, pos_template, pos_search)

            outputs_class = self.class_embed(hs)
            outputs_coord = self.bbox_embed(hs).sigmoid()
            cls.append(outputs_class[-1])
            loc.append(outputs_coord[-1])

        cls_weight = F.softmax(self.cls_weight, 0)#对列进行归一化操作
        loc_weight = F.softmax(self.loc_weight, 0)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s


        a,b=weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight)
        out = {'pred_logits': a, 'pred_boxes': b}
        return out

    def joiner(self,tensor_list: NestedTensor):
        x=self.backbone(tensor_list.tensors)
        xf=self.neck(x)
        out: Dict[str, NestedTensor] = {}
        m = tensor_list.mask
        assert m is not None
        out: List[NestedTensor] = []
        pos = []
        for im in xf:
            mask = F.interpolate(m[None].float(), size=im.shape[-2:]).to(torch.bool)[0]
            a = NestedTensor(im, mask)
            xs=a
            out.append(xs)
            pos.append(self.pos_embeding(xs).to(xs.tensors.dtype))

        return out, pos
    def pos_embeding(self,tensor_list: NestedTensor):
        num_pos_feats = 128
        temperature = 10000
        normalize = True
        scale = None
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    def pos_embeding_learn(self,tensor_list: NestedTensor):
        num_pos_feats = 256

        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos
    def log_softmax(self, cls):
        if cfg.BAN.BAN:
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = F.log_softmax(cls, dim=3)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()

        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor_2(search)
        if not isinstance(template, NestedTensor):
            template = nested_tensor_from_tensor_2(template)
        features_searchs, pos_searchs = self.joiner(search)
        self.zf,self.pos_template=self.joiner(template)
        feature_templates = self.zf
        pos_templates = self.pos_template
        cls = []
        loc = []
        for idx, (features_search, pos_search, feature_template, pos_template) in enumerate(
                zip(features_searchs, pos_searchs, feature_templates, pos_templates), start=2):
            src_search, mask_search = features_search.decompose()
            assert mask_search is not None
            src_template, mask_template = feature_template.decompose()
            assert mask_template is not None

            hs = self.featurefusion_network(self.input_proj(src_template), mask_template, self.input_proj(src_search),
                                            mask_search, pos_template, pos_search)

            outputs_class = self.class_embed(hs)
            outputs_coord = self.bbox_embed(hs).sigmoid()
            cls.append(outputs_class[-1])
            loc.append(outputs_coord[-1])

        cls_weight = F.softmax(self.cls_weight, 0)  # 对列进行归一化操作
        loc_weight = F.softmax(self.loc_weight, 0)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        a, b = weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight)
        out = {'pred_logits': a, 'pred_boxes': b}
        return out

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x