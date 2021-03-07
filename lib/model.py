from __future__ import print_function, division
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import numpy.matlib
import pickle
import MinkowskiEngine as ME
import math

from .conv4d import Conv4d
from .sparse import *

def dense_to_sparse(x):
    """ converts dense tensor x to sparse format """

    indices = torch.nonzero(x)

    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return torch.sparse.FloatTensor(indices,values).coalesce()

def featureL2Norm(feature):
    epsilon = 1e-6
    if isinstance(feature, list):
        result = []
        for f in feature:
            norm = torch.pow(torch.sum(torch.pow(f,2),1)+epsilon,0.5).unsqueeze(1).expand_as(f)
            result.append(torch.div(f, norm))
        return result
    else:
        norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature,norm)

class FeatureExtraction(torch.nn.Module):
    def __init__(self, train_fe=False, feature_extraction_cnn='resnet101', feature_extraction_model_file='', normalization=True, last_layer='', use_cuda=True):
        super(FeatureExtraction, self).__init__()
        self.normalization = normalization
        self.feature_extraction_cnn=feature_extraction_cnn
        # for resnet below
        resnet_feature_layers = ['conv1','bn1','relu','maxpool','layer1','layer2','layer3','layer4']
        if feature_extraction_cnn.startswith('resnet'):
            if feature_extraction_cnn=='resnet101':
                self.model = models.resnet101(pretrained=True)
            elif feature_extraction_cnn=='resnet18':
                self.model = models.resnet18(pretrained=True)
            elif feature_extraction_cnn=='resnet50':
                self.model = models.resnet50(pretrained=True)
            if last_layer=='':
                last_layer = 'layer3'                            
            resnet_module_list = [getattr(self.model,l) for l in resnet_feature_layers]
            last_layer_idx = resnet_feature_layers.index(last_layer)
            self.model = nn.Sequential(*resnet_module_list[:last_layer_idx])
            self.f1 = nn.Sequential(resnet_module_list[last_layer_idx][0], resnet_module_list[last_layer_idx][1])
            self.f2 = nn.Sequential(resnet_module_list[last_layer_idx][2], resnet_module_list[last_layer_idx][3])
            self.f3 = nn.Sequential(resnet_module_list[last_layer_idx][4], resnet_module_list[last_layer_idx][5])

        if train_fe==False:
            # freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.f1.parameters():
                param.requires_grad = False
            for param in self.f2.parameters():
                param.requires_grad = False
            for param in self.f3.parameters():
                param.requires_grad = False
        # move to GPU
        if use_cuda:
            self.model = self.model.cuda()
            self.f1 = self.f1.cuda()
            self.f2 = self.f2.cuda()
            self.f3 = self.f3.cuda()
        
    def forward(self, image_batch):
        features = self.model(image_batch)
        f1 = self.f1(features)
        f2 = self.f2(f1)
        f3 = self.f3(f2)
        features = [f1,f2,f3]
        if self.normalization and not self.feature_extraction_cnn=='resnet101fpn':
            features = featureL2Norm(features)
        return features
    
    def change_stride(self):
        print('Changing FeatureExtraction stride')
        self.model[-1][0].conv1.stride=(1,1)
        self.model[-1][0].conv2.stride=(1,1)
        self.model[-1][0].downsample[0].stride=(1,1)
    
def corr_dense(feature_A, feature_B):        
    b,c,hA,wA = feature_A.size()
    b,c,hB,wB = feature_B.size()
    # reshape features for matrix multiplication
    feature_A = feature_A.view(b,c,hA*wA).transpose(1,2) # size [b,c,h*w]
    feature_B = feature_B.view(b,c,hB*wB) # size [b,c,h*w]
    # perform matrix mult.
    feature_mul = torch.bmm(feature_A,feature_B)
    # indexed [batch,row_A,col_A,row_B,col_B]
    correlation_tensor = feature_mul.view(b,hA,wA,hB,wB).unsqueeze(1)
        
    return correlation_tensor

class GeometricSparseNeighConsensus(torch.nn.Module):
    def __init__(self, use_cuda=True, kernel_sizes=[3], channels=[1], symmetric_mode=True, bn=False, dimension=6):
        super(GeometricSparseNeighConsensus, self).__init__()
        self.symmetric_mode = symmetric_mode
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        num_layers = len(kernel_sizes)
        nn_modules = list()
        ch_in = 1
        ch_out = channels[0]
        k_size = kernel_sizes[0]
        nn_modules.append(ME.MinkowskiReLU(inplace=True))
        nn_modules.append(ME.MinkowskiConvolution(1,1,kernel_size=k_size,bias=False,dimension=dimension))

        #nn_modules.append(ME.MinkowskiConvolution(8,1,kernel_size=k_size,bias=False,dimension=dimension))
        #nn_modules.append(ME.MinkowskiSigmoid())
        nn_modules.append(ME.MinkowskiSigmoid())
        self.conv = nn.Sequential(*nn_modules) 
        
        if use_cuda:
            self.conv.cuda()

    def forward(self, x):
        if self.symmetric_mode:
            # apply network on the input and its "transpose" (swapping A-B to B-A ordering of the correlation tensor),
            # this second result is "transposed back" to the A-B ordering to match the first result and be able to add together

            x1 = me_to_torch(self.conv(x))
            x = x1 + transpose_torch_6d(me_to_torch(self.conv(transpose_me_6d(x))))
            x = x.coalesce().to_dense()
            #x = me_to_torch(x).coalesce().to_dense()
            x = x.squeeze(-1)
            b, s, s, dim1, dim2 , dim3, dim4 = x.shape
            x = x.reshape(b,-1,dim1,dim2,dim3,dim4)
            x = x.sum(dim=1)
            x = dense_to_sparse(x)
            x = torch_to_me(x)
        else:
            exit("Not symmetric mode")
        
        return x

class SparseNeighConsensus(torch.nn.Module):
    def __init__(self, use_cuda=True, kernel_sizes=[3,3,3], channels=[10,10,1], symmetric_mode=True, bn=False):
        super(SparseNeighConsensus, self).__init__()
        self.symmetric_mode = symmetric_mode
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers):
            if i==0:
                nn_modules.append(ME.MinkowskiReLU(inplace=True))
                ch_in = 1
            else:
                ch_in = channels[i-1]
            ch_out = channels[i]
            k_size = kernel_sizes[i]
            if ch_out==1 or bn==False:
                nn_modules.append(ME.MinkowskiConvolution(ch_in,ch_out,kernel_size=k_size,bias=True,dimension=4))
            elif bn==True:
                nn_modules.append(torch.nn.Sequential(
                    ME.MinkowskiConvolution(ch_in,ch_out,kernel_size=k_size,bias=True,dimension=4),
                    ME.MinkowskiBatchNorm(ch_out)))
            nn_modules.append(ME.MinkowskiReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules) 
        
        if use_cuda:
            self.conv.cuda()

    def forward(self, x):
        if self.symmetric_mode:
            # apply network on the input and its "transpose" (swapping A-B to B-A ordering of the correlation tensor),
            # this second result is "transposed back" to the A-B ordering to match the first result and be able to add together
            x = me_to_torch(self.conv(x)) + transpose_torch(me_to_torch(self.conv(transpose_me(x))))
            x = x.coalesce()
        else:
            x = me_to_torch(self.conv(x))
        return x

class DenseNeighConsensus(torch.nn.Module):
    def __init__(self, use_cuda=True, kernel_sizes=[3,3,3], channels=[10,10,1], symmetric_mode=True):
        super(DenseNeighConsensus, self).__init__()
        self.symmetric_mode = symmetric_mode
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers):
            if i==0:
                ch_in = 1
            else:
                ch_in = channels[i-1]
            ch_out = channels[i]
            k_size = kernel_sizes[i]
            nn_modules.append(Conv4d(in_channels=ch_in,out_channels=ch_out,kernel_size=k_size,bias=True))
            nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)        
        if use_cuda:
            self.conv.cuda()

    def forward(self, x):
        if self.symmetric_mode:
            # apply network on the input and its "transpose" (swapping A-B to B-A ordering of the correlation tensor),
            # this second result is "transposed back" to the A-B ordering to match the first result and be able to add together
            x = self.conv(x)+self.conv(x.permute(0,1,4,5,2,3)).permute(0,1,4,5,2,3)
        else:
            x = me_to_torch(self.conv(x))
        return x

def MutualMatching(corr4d):
    # mutual matching
    batch_size,ch,fs1,fs2,fs3,fs4 = corr4d.size()

    corr4d_B=corr4d.view(batch_size,fs1*fs2,fs3,fs4) # [batch_idx,k_A,i_B,j_B]
    corr4d_A=corr4d.view(batch_size,fs1,fs2,fs3*fs4)

    # get max
    corr4d_B_max,_=torch.max(corr4d_B,dim=1,keepdim=True)
    corr4d_A_max,_=torch.max(corr4d_A,dim=3,keepdim=True)

    eps = 1e-5
    corr4d_B=corr4d_B/(corr4d_B_max+eps)
    corr4d_A=corr4d_A/(corr4d_A_max+eps)

    corr4d_B=corr4d_B.view(batch_size,1,fs1,fs2,fs3,fs4)
    corr4d_A=corr4d_A.view(batch_size,1,fs1,fs2,fs3,fs4)

    corr4d=corr4d*(corr4d_A*corr4d_B) # parenthesis are important for symmetric output 
        
    return corr4d

def maxpool4d(corr4d_hres,k_size=4):
    slices=[]
    for i in range(k_size):
        for j in range(k_size):
            for k in range(k_size):
                for l in range(k_size):
                    slices.append(corr4d_hres[:,0,i::k_size,j::k_size,k::k_size,l::k_size].unsqueeze(0))
    slices=torch.cat(tuple(slices),dim=1)
    corr4d,max_idx=torch.max(slices,dim=1,keepdim=True)
    max_l=torch.fmod(max_idx,k_size)
    max_k=torch.fmod(max_idx.sub(max_l).div(k_size),k_size)
    max_j=torch.fmod(max_idx.sub(max_l).div(k_size).sub(max_k).div(k_size),k_size)
    max_i=max_idx.sub(max_l).div(k_size).sub(max_k).div(k_size).sub(max_j).div(k_size)
    # i,j,k,l represent the *relative* coords of the max point in the box of size k_size*k_size*k_size*k_size
    return (corr4d,max_i,max_j,max_k,max_l)

class ImMatchNet(nn.Module):
    def __init__(self, 
                 feature_extraction_cnn='resnet101', 
                 feature_extraction_last_layer='',
                 feature_extraction_model_file='',
                 return_correlation=False,  
                 ncons_kernel_sizes=[3,3,3],
                 ncons_channels=[10,10,1],
                 normalize_features=True,
                 train_fe=False,
                 use_cuda=True,
                 relocalization_k_size=0,
                 half_precision=False,
                 checkpoint=None,
                 sparse=False,
                 symmetric_mode=True,
                 k = 10,
                 bn=False,
                 return_fs=False,
                 change_stride=False
                 ):
        
        super(ImMatchNet, self).__init__()
        # Load checkpoint
        if checkpoint is not None and checkpoint is not '':
            ncons_channels, ncons_kernel_sizes, checkpoint = self.get_checkpoint_parameters(checkpoint)

        self.use_cuda = use_cuda
        self.normalize_features = normalize_features
        self.return_correlation = return_correlation
        self.relocalization_k_size = relocalization_k_size
        self.half_precision = half_precision
        self.sparse = sparse
        self.k = k
        self.d2 = feature_extraction_cnn=='d2'
        self.return_fs = return_fs
        self.Npts = None
        
        self.FeatureExtraction = FeatureExtraction(train_fe=train_fe,
                                                   feature_extraction_cnn=feature_extraction_cnn,
                                                   feature_extraction_model_file=feature_extraction_model_file,
                                                   last_layer=feature_extraction_last_layer,
                                                   normalization=normalize_features,
                                                   use_cuda=self.use_cuda)
        self.FeatureExtraction.eval()
        
        if sparse:
            self.NeighConsensus = SparseNeighConsensus(use_cuda=self.use_cuda,
                                             kernel_sizes=ncons_kernel_sizes,
                                             channels=ncons_channels,
                                             symmetric_mode=symmetric_mode,
                                             bn = bn)
        else:
            self.NeighConsensus = DenseNeighConsensus(use_cuda=self.use_cuda,
                                             kernel_sizes=ncons_kernel_sizes,
                                             channels=ncons_channels,
                                             symmetric_mode=symmetric_mode)

        if checkpoint is not None and checkpoint is not '': self.load_weights(checkpoint)
        if self.half_precision: self.set_half_precision()
        if change_stride: self.FeatureExtraction.change_stride()

    def get_checkpoint_parameters(self, checkpoint):
        print('Loading checkpoint...')
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        checkpoint['state_dict'] = OrderedDict([(k.replace('vgg', 'model'), v) for k, v in checkpoint['state_dict'].items()])
        # override relevant parameters
        print('Using checkpoint parameters: ')
        ncons_channels=checkpoint['args'].ncons_channels
        print('  ncons_channels: '+str(ncons_channels))
        ncons_kernel_sizes=checkpoint['args'].ncons_kernel_sizes
        print('  ncons_kernel_sizes: '+str(ncons_kernel_sizes)) 
        return ncons_channels, ncons_kernel_sizes, checkpoint
    
    def load_weights(self, checkpoint):
        # Load weights
        print('Copying weights...')
        for name, param in self.FeatureExtraction.state_dict().items():
            if 'num_batches_tracked' not in name:
                self.FeatureExtraction.state_dict()[name].copy_(checkpoint['state_dict']['FeatureExtraction.' + name])    
        for name, param in self.NeighConsensus.state_dict().items():
            self.NeighConsensus.state_dict()[name].copy_(checkpoint['state_dict']['NeighConsensus.' + name])
        print('Done!')
        
    def set_half_precision(self):
        for p in self.NeighConsensus.parameters():
            p.data=p.data.half()
        for l in self.NeighConsensus.conv:
            if isinstance(l,Conv4d):
                l.use_half=True
                    
    def forward(self, tnf_batch): 
        # feature extraction
        feature_A = self.FeatureExtraction(tnf_batch['source_image'])
        feature_B = self.FeatureExtraction(tnf_batch['target_image'])

        if self.sparse:
            return self.process_sparse(feature_A, feature_B)
        else:
            return self.process_dense(feature_A, feature_B)
            
    def process_sparse(self, feature_A, feature_B):
        corr4d = corr_and_add(feature_A, feature_B, k = self.k, Npts = self.Npts)
        corr4d = self.NeighConsensus(corr4d)
        if self.return_fs:
            fs1, fs2 = feature_A.shape[-2:]
            fs3, fs4 = feature_B.shape[-2:]
            return corr4d, fs1, fs2, fs3, fs4
        else:
            return corr4d
        
    def process_dense(self, feature_A, feature_B):
        if self.half_precision:
            feature_A=feature_A.half()
            feature_B=feature_B.half()
        corr4d = corr_dense(feature_A,feature_B)
        if self.relocalization_k_size>1:
            corr4d,max_i,max_j,max_k,max_l=maxpool4d(corr4d,k_size=self.relocalization_k_size)
        corr4d = MutualMatching(corr4d)
        corr4d = self.NeighConsensus(corr4d)            
        corr4d = MutualMatching(corr4d)        
        if self.relocalization_k_size>1:
            delta4d = (max_i,max_j,max_k,max_l)
            return (corr4d, delta4d)
        else:
            return corr4d


#TODO: To_implement
class TransformNet(nn.Module):
    def __init__(self, 
                 feature_extraction_cnn='resnet101', 
                 feature_extraction_last_layer='',
                 feature_extraction_model_file='',
                 return_correlation=False,
                 ncons_kernel_sizes=[3],
                 ncons_channels=[1],
                 normalize_features=True,
                 train_fe=False,
                 use_cuda=True,
                 relocalization_k_size=0,
                 half_precision=False,
                 checkpoint=None,
                 sparse=False,
                 symmetric_mode=True,
                 k = 10,
                 bn=False,
                 return_fs=False,
                 change_stride=False,
                 scale=3
                 ):
        
        super(TransformNet, self).__init__()
        # Load checkpoint
        #if checkpoint is not None and checkpoint is not '':
        #    ncons_channels, ncons_kernel_sizes, checkpoint = self.get_checkpoint_parameters(checkpoint)

        self.use_cuda = use_cuda
        self.normalize_features = normalize_features
        self.return_correlation = return_correlation
        self.relocalization_k_size = relocalization_k_size
        self.half_precision = half_precision
        self.sparse = sparse
        self.k = k
        self.d2 = feature_extraction_cnn=='d2'
        self.return_fs = return_fs
        self.Npts = None
        
        self.FeatureExtraction = FeatureExtraction(train_fe=train_fe,
                                                   feature_extraction_cnn=feature_extraction_cnn,
                                                   feature_extraction_model_file=feature_extraction_model_file,
                                                   last_layer=feature_extraction_last_layer,
                                                   normalization=normalize_features,
                                                   use_cuda=self.use_cuda)
        self.FeatureExtraction.eval()
        
        self.NeighConsensus = GeometricSparseNeighConsensus(use_cuda=self.use_cuda,
                                            kernel_sizes=ncons_kernel_sizes,
                                            channels=ncons_channels,
                                            symmetric_mode=symmetric_mode,
                                            bn = bn)

        self.refineNeighConsensus = SparseNeighConsensus(use_cuda=self.use_cuda,
                                            kernel_sizes=[3,3],
                                            channels=[16,1],
                                            symmetric_mode=symmetric_mode,
                                            bn = bn)

        #if checkpoint is not None and checkpoint is not '': self.load_weights(checkpoint)
        if self.half_precision: self.set_half_precision()
        if change_stride: self.FeatureExtraction.change_stride()

        # for 6d purposes 
        if scale == 3:
            self.scales = [0.5, 1, 2]
        elif scale == 5:
            self.scales = [0.5, 0.75, 1, 1.5, 2]

        self.conv2ds = []
        self.conv2ds.append(nn.ModuleList(
                    [nn.Conv2d(1024, 1024 // 4, kernel_size=3, stride=1, padding=1, bias=False).cuda() for _ in self.scales]))
        self.conv2ds = nn.ModuleList(self.conv2ds)

        self.sigmoid = nn.Sigmoid()

    def get_checkpoint_parameters(self, checkpoint):
        print('Loading checkpoint...')
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        checkpoint['state_dict'] = OrderedDict([(k.replace('vgg', 'model'), v) for k, v in checkpoint['state_dict'].items()])
        # override relevant parameters
        print('Using checkpoint parameters: ')
        ncons_channels=checkpoint['args'].ncons_channels
        print('  ncons_channels: '+str(ncons_channels))
        ncons_kernel_sizes=checkpoint['args'].ncons_kernel_sizes
        print('  ncons_kernel_sizes: '+str(ncons_kernel_sizes)) 
        return ncons_channels, ncons_kernel_sizes, checkpoint
    
    def load_weights(self, checkpoint):
        # Load weights
        print('Copying weights...')
        for name, param in self.FeatureExtraction.state_dict().items():
            if 'num_batches_tracked' not in name:
                self.FeatureExtraction.state_dict()[name].copy_(checkpoint['state_dict']['FeatureExtraction.' + name])    
        for name, param in self.NeighConsensus.state_dict().items():
            self.NeighConsensus.state_dict()[name].copy_(checkpoint['state_dict']['NeighConsensus.' + name])
        for name, param in self.refineNeighConsensus.state_dict().items():
            self.refineNeighConsensus.state_dict()[name].copy_(checkpoint['state_dict']['refineNeighConsensus.' + name])
        for name, param in self.conv2ds.state_dict().items():
            self.conv2ds.state_dict()[name].copy_(checkpoint['state_dict']['conv2ds.' + name])
        print('Done!')
        
    def set_half_precision(self):
        for p in self.NeighConsensus.parameters():
            p.data=p.data.half()
        for l in self.NeighConsensus.conv:
            if isinstance(l,Conv4d):
                l.use_half=True
                    
    def forward(self, tnf_batch): 
        # feature extraction
        feature_A = self.FeatureExtraction(tnf_batch['source_image'])
        feature_B = self.FeatureExtraction(tnf_batch['target_image'])
        return self.process_sparse(feature_A, feature_B)
            
    def process_sparse(self, feature_A, feature_B):
        # create a 6D tensor
        #import pdb; pdb.set_trace()
        correlations = self.scalespace_correlation(feature_A, feature_B, self.scales, self.conv2ds)
        corr4d = self.NeighConsensus(correlations)
        corr4d = self.refineNeighConsensus(corr4d)

        if self.return_fs:
            fs1, fs2 = feature_A[0].shape[-2:]
            fs3, fs4 = feature_B[0].shape[-2:]
            return corr4d, fs1, fs2, fs3, fs4

        return corr4d

    def scalespace_correlation(self, src_feats, trg_feats, scales, conv2d):
        r""" *_feats: list of features from intermediate layers
             [(bsz, ch_1, h, w), (bsz, ch_2, h, w), ..., (bsz, ch_n, h, w)]
        """

        bsz, ch, side1, side2 = src_feats[0].size()
        bsz, ch, side3, side4 = trg_feats[0].size()

        # new_src_feats = []
        # new_trg_feats = []
        # for scale, conv in zip(scales, conv2d[0]):
        #     new_side = round(side * math.sqrt(scale))
        #     new_src_feat = F.interpolate(src_feat, (new_side, new_side), mode='bilinear', align_corners=True)
        #     new_trg_feat = F.interpolate(trg_feat, (new_side, new_side), mode='bilinear', align_corners=True)
        #     new_src_feats.append(conv(new_src_feat))
        #     new_trg_feats.append(conv(new_trg_feat))

        # correlations = []

        length1 = bsz * side1 * side2 * self.k
        length2 = bsz * side3 * side4 * self.k
        #import pdb; pdb.set_trace()
        scalewise_feat_st = torch.zeros(length1 * len(scales) * len(scales), 1, device=torch.device('cuda'))
        scalewise_coord_st = torch.zeros(length1 * len(scales) * len(scales), 7, device=torch.device('cuda'), dtype=torch.int)
        scalewise_feat_ts = torch.zeros(length2 * len(scales) * len(scales), 1, device=torch.device('cuda'))
        scalewise_coord_ts = torch.zeros(length2 * len(scales) * len(scales), 7, device=torch.device('cuda'), dtype=torch.int)
        for s_1, src_feat in enumerate(src_feats):
            new_ch = src_feat.size(1)

            #src_feat = F.interpolate(src_feat, (side, side),mode='bilinear', align_corners=True)
            #src_feat = src_feat.view(bsz, new_ch, -1)

            for s_2, trg_feat in enumerate(trg_feats):
                #trg_feat = F.interpolate(trg_feat, (side, side),mode='bilinear', align_corners=True)
                #trg_feat = trg_feat.view(bsz, new_ch, -1)

                corr_coords_st = sparse_corr_6d([s_1,s_2], src_feat, trg_feat, k=self.k, ratio=False, sparse_type='raw')
                corr_coords_ts = sparse_corr_6d([s_1,s_2], trg_feat, src_feat, k=self.k, ratio=False, sparse_type='raw', reverse=True)

                start1 = (s_2 + s_1 * len(scales)) * length1
                end1 = start1 + length1
                
                start2 = (s_2 + s_1 * len(scales)) * length2
                end2 = start2 + length2

                scalewise_feat_st[start1:end1] = corr_coords_st[0]
                scalewise_coord_st[start1:end1] = corr_coords_st[1]
                scalewise_feat_ts[start2:end2] = corr_coords_ts[0]
                scalewise_coord_ts[start2:end2] = corr_coords_ts[1]

        #import pdb; pdb.set_trace()
        s2t = ME.SparseTensor(scalewise_feat_st, scalewise_coord_st)
        t2s = ME.SparseTensor(scalewise_feat_ts, scalewise_coord_ts, coordinate_manager=s2t.coordinate_manager)
        correlations = ME.MinkowskiUnion()(s2t, t2s)         

        return correlations


