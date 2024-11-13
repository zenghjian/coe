import torch
from base import BaseModel
from model.ASAP_diffusionNet.asap_layers import ASAP_DiffusionNet
from utils.geometry_util import compute_hks_autoscale, compute_wks_autoscale
from model.attention import CrossAttentionRefinementNet


class CoeNet(BaseModel):
    def __init__(self, n_eig, input_type="xyz", attention=False):
        super(CoeNet, self).__init__()
        

        self.input_type = input_type
        self.attention = attention

        C_in = {'xyz': 3, 'hks': 16, 'wks': 128}[input_type]  # dimension of input features        

        self.feature_extractor = ASAP_DiffusionNet(
            C_in=C_in,
            C_out=n_eig,
            C_width=128,
            N_block=4,
            dropout=False,
        )
        self.refine = CrossAttentionRefinementNet(
            n_in=n_eig,
            num_head=4,
            gnn_dim=128,
            n_layers=1
        )

    def forward(self, inputs):
        '''
        Params:
        # inputs: tuple with the following elements:
        
        vertices:   (B, 2, N, 3)
        eVecs:      (B, 2, N, D)
        eVals:      (B, 2, D)
        Ls:         (B, 2, N, N)
        Ms:         (B, 2, N)
        gradXs:     (B, 2, N, N)
        gradYs:     (B, 2, N, N)

        Return:
        final_eigens: tensor with shape (batch_size, 2, n_vertices, n_eig) = (B, 2, N, C_out)
        '''
        vertices, eVecs, eVals, Ls, Ms, gradXs, gradYs = inputs
        
        intermediate_embeddings= []

        for i in range(2):  # Since second dimension is fixed to 2
            vertex = vertices[:, i, :, :]
            mass = Ms[:, i, :]
            L = Ls[:, i, :, :]
            evals = eVals[:, i, :]
            evecs = eVecs[:, i, :, :]
            gradX = gradXs[:, i, :, :]
            gradY = gradYs[:, i, :, :]
            face = None

            if self.input_type == "xyz":
                feature = vertex  # (B, N, C_in=3)
            elif self.input_type == "hks":
                feature = compute_hks_autoscale(evals, evecs, count=16)  # (B, N, C_in=16)
            elif self.input_type == "wks":
                feature = compute_wks_autoscale(evals, evecs, mass, n_descr=128)  # (B, N, C_in=128)
            else:
                raise ValueError("Invalid input type")

            # feat: (B, N, C_out)
            feat = self.feature_extractor(feature, mass, L=L, evals=evals, evecs=evecs,
                                          gradX=gradX, gradY=gradY, faces=face)
            intermediate_embeddings.append(feat)

        if self.attention:
            # attention refine
            feat1 = intermediate_embeddings[0]
            feat2 = intermediate_embeddings[1]
            refined_feat1, refined_feat2 = self.refine(feat1, feat2)

            final_embeddings = torch.stack([refined_feat1, refined_feat2], dim=1)  # (B, 2, N, C_out)
        else:
            final_embeddings = torch.stack(intermediate_embeddings, dim=1)
            
        return final_embeddings  # (B, 2, N, C_out)


