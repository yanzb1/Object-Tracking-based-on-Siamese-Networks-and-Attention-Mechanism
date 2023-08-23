from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from siamban.models.head.ban import UPChannelBAN, DepthwiseBAN, MultiBAN

from siamban.models.head.head import FCOSOcean
import importlib
BANS = {
        'UPChannelBAN': UPChannelBAN,
        'DepthwiseBAN': DepthwiseBAN,
        'MultiBAN': MultiBAN,
        'FCOSOcean':FCOSOcean
       }


'''def get_ban_head(name, **kwargs):
    head_module = importlib.import_module('siamban.models.head.head')
    head_func = getattr(head_module, name)
    head = head_func(in_channels=256, out_channels=256,
                     towernum=4, align=False)
    return head
    #这个是ocean的
'''
def get_ban_head(name, **kwargs):
    return BANS[name](**kwargs)