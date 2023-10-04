
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from voxelmorph import default_unet_features
from voxelmorph.torch import layers
from voxelmorph.torch.modelio import LoadableModel, store_config_args

"""
The PCMC core code based on:

Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
MICCAI 2018. eprint arXiv:1805.04605
https://github.com/voxelmorph/voxelmorph

"""


class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class PCMC(LoadableModel):

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=0,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True
        self.flows_number = src_feats + trg_feats 
        # device handling
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)

        # 16 > 4 > 4 > 2

        preflow_layers = [4,4]

        print('preflow_layers =')
        print(preflow_layers)

        if len(preflow_layers):
            
            
            self.preflow_1 = nn.ModuleList()
            for ii in range(self.flows_number):
                flow_itter = ConvBlock(ndims,self.unet_model.final_nf, preflow_layers[0]).to(self.device)
                self.preflow_1.append(flow_itter)

            if len(preflow_layers) >1:
                self.preflow_2 = nn.ModuleList()
                for ii in range(self.flows_number):
                    flow_itter = ConvBlock(ndims,preflow_layers[0], preflow_layers[1]).to(self.device)
                    self.preflow_2.append(flow_itter)
            

            deformation_input_layer = preflow_layers[-1]
        else:
            deformation_input_layer = self.unet_model.final_nf

        self.flow = nn.ModuleList()
        for ii in range(self.flows_number):
            flow_itter = Conv(deformation_input_layer, ndims, kernel_size=3, padding=1)
            # init flow layer with small weights and bias
            flow_itter.weight = nn.Parameter(Normal(0, 1e-5).sample(flow_itter.weight.shape).to(self.device))
            flow_itter.bias = nn.Parameter(torch.zeros(flow_itter.bias.shape).to(self.device))
            self.flow.append(flow_itter)

        self.flow_heads = nn.ModuleList()
        for ii in range(self.flows_number):
            if len(preflow_layers) and len(preflow_layers) >1:
                self.flow_heads.append(nn.Sequential(self.preflow_1[ii], self.preflow_2[ii] , self.flow[ii] ))
            elif len(preflow_layers) and len(preflow_layers) == 1:
                self.flow_heads.append(nn.Sequential(self.preflow_1[ii] , self.flow[ii] ))
            else:
                self.flow_heads.append(nn.Sequential(self.flow[ii]))


        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape[:2])


        # ############ BASED MODEL DECODER ############
        # configure core ModelBased unet model
        self.unet_based_T1 = Unet(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # self.M0_head = nn.ModuleList()
        self.M0_head = ConvBlock(ndims,self.unet_model.final_nf, 1).to(self.device)            
        # self.M0_head.append(nn.Sequential(M0_head))

        # self.T1_head = nn.ModuleList()
        self.T1_head = ConvBlock(ndims,self.unet_model.final_nf, 1).to(self.device)            
        # self.T1_head.append(nn.Sequential(T1_head))

    def forward(self, source, seg=None):
        '''
        Parameters:
            source: Source image tensor. [B,1,160,160,11];  
        '''
        device=source.device

        # concatenate inputs and propagate unet
        # all the slices togheter
        x = source.permute(0, 4, 2, 3, 1)[:,:,:,:,0] # x[B,16,160,160]
        input = x   #[B,11,160,160]
        
        if seg is not None:
            if seg.dim() == 5:
                seg = seg.permute(0, 4, 2, 3, 1)[:,:,:,:,0]




        # splitted to two input slices for converting to unet shape 
        unet_output = self.unet_model(x) 
        
        all_flow_field = []
        all_y_source = []
        if seg!=None: y_seg_seq = []

        # t==0  ; no flow
        # t=0
        # flow_field = torch.zeros(x.shape[0],1,2,x.shape[2],x.shape[3]).to(device)
        # all_flow_field.append(flow_field)  #[B,1,2,160,160]
        # all_y_source.append(torch.unsqueeze(input[:,t,:,:],dim=1))#[B,1,160,160] 

        ref = 0
        for flow_index in range(self.flows_number):
            # t = flow_index+1
            if flow_index == ref: #no flow
                flow_field = torch.zeros(x.shape[0],1,2,x.shape[2],x.shape[3]).to(device)
                all_flow_field.append(flow_field)  #[B,1,2,160,160]
                all_y_source.append(torch.unsqueeze(input[:,flow_index,:,:],dim=1))#[B,1,160,160] 

                if seg!=None:
                    y_seg_seq.append(torch.unsqueeze(seg[:,flow_index,:,:],dim=1))

            else: #flow
                flow_field = self.flow_heads[flow_index](unet_output) #[B,2,160,160]
                y_source = self.transformer(torch.unsqueeze(input[:,flow_index,:,:],dim=1), flow_field) #[B,1,160,160] 
                all_flow_field.append(torch.unsqueeze(flow_field,dim=1)) #[B,11,2,160,160]
                all_y_source.append(y_source)     #[B,11,160,160]

                if seg!=None:
                    y_seg = self.transformer(torch.unsqueeze(seg[:,flow_index,:,:],dim=1), flow_field)
                    y_seg_seq.append(y_seg)

        all_flow_field = torch.cat(all_flow_field,1)
        if seg!=None: 
            y_seg_seq = torch.cat(y_seg_seq,1) 
        else: 
            y_seg_seq=None
        all_y_source = torch.cat(all_y_source,1)
        
        # ############ BASED MODEL DECODER ############
        unet_based_output = self.unet_based_T1(x) 
        T1 = torch.tensor(800) - self.T1_head(unet_based_output)
        M0 = torch.tensor(180) - self.M0_head(unet_based_output)

        return all_y_source, all_flow_field, M0, T1, y_seg_seq


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out
