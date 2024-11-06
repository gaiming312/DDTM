import torch
from scipy.spatial import Delaunay,qhull
import numpy as np
from ..builder import SSL_MODULES
from mmdet3d.core.bbox import LiDARInstance3DBoxes
import copy
from .utils import mlvl_get,mlvl_set,mlvl_getattr



@SSL_MODULES.register_module()
class maskInputPoints():
    def __init__(self,
                 point_key,
                 in_bboxes_key,
                 out_points_feat_key,
                 out_points_key):

        self.point_key = point_key
        self.in_bboxes_key = in_bboxes_key
        self.out_points_feat_key = out_points_feat_key
        self.out_points_key = out_points_key
    def forward(self,ssl_obj,batch_dict):
        """

        Args:
            points: Input student points
            bboxes: LiDarInstance3DBoxes

        Returns:
            points with mask
        """
        in_points = mlvl_get(batch_dict,self.point_key)
        in_bboxes = mlvl_get(batch_dict,self.in_bboxes_key)

        out_points_feat = []
        positive_points = []
        for batch_idx,(curr_in_points,curr_in_bboxes) in \
                enumerate(zip(in_points,in_bboxes)):
            curr_in_points_cp = copy.deepcopy(curr_in_points)[:,:3]

            if isinstance(curr_in_bboxes, tuple):
                curr_in_bboxes_3d = curr_in_bboxes[0]
                rest = curr_in_bboxes[1:]

            else:
                assert isinstance(curr_in_bboxes, torch.Tensor)
                curr_in_bboxes_3d = curr_in_bboxes

            if not curr_in_bboxes_3d.shape[0] > 0:
                positive_points.append(
                    curr_in_points[torch.randint(len(curr_in_points),(150,))]
                )
                continue

            assert isinstance(curr_in_bboxes_3d, LiDARInstance3DBoxes)

            curr_in_bboxes_3d_corners = curr_in_bboxes_3d.corners #  Tensor (N,8,3)

            device = curr_in_points.device
            N = curr_in_bboxes_3d_corners.size()[0]
            masks = [ self._is_hull(curr_in_points_cp.cpu().numpy(),bbox.squeeze().cpu().numpy()) for bbox \
                      in list(curr_in_bboxes_3d_corners.split(split_size=1,dim=0))] # return list [(8,3)]

            assert N == len(masks)

            mask_matrix = self._compute_logic_or(masks)

            mask_matrix = torch.tensor(mask_matrix,dtype=torch.bool,device=device)

            mask = curr_in_points_cp.new_zeros(curr_in_points_cp.size()[0])

            mask[mask_matrix] = 1

            curr_in_points = torch.cat((curr_in_points,mask.unsqueeze(dim=-1)),dim=1)

            out_points_feat.append(curr_in_points)

            positive_points.append(curr_in_points[mask_matrix])

        mlvl_set(batch_dict,self.out_points_feat_key,out_points_feat)
        mlvl_set(batch_dict,self.out_points_key,positive_points)
        assert len(positive_points) == len(in_points)
        return batch_dict

    def _is_hull(self,p , hull):
        """

        Args:
            p: points :(N,3) points
            hull: (M,K) M corners of a box

        Returns:
             N bool
        """
        try:
            if not isinstance(hull, Delaunay):
                hull = Delaunay(hull)
            flag = hull.find_simplex(p) >= 0
        except qhull.QhullError:
            print('Warning: not a hull %s' % str(hull))
            flag = np.zeros(p.shape[0], dtype=np.bool)

        return flag


    def _compute_logic_or(self,masks):
        mask_matrix = masks[0]
        for i in range(len(masks)-1):
            if i ==0:
                mask_matrix = np.logical_or(masks[i],masks[i+1])
            else:
                mask_matrix = np.logical_or(mask_matrix,masks[i+1])
        return mask_matrix


