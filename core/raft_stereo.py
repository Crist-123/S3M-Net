import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicMultiUpdateBlock
from core.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from core.corr import CorrBlock1D, PytorchAlternateCorrBlock1D, CorrBlockFast1D, AlternateCorrBlock
from core.utils.utils import coords_grid, upflow8
import torchvision

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass
'''
self.cnet为一个编码器网络，用于提取特征。
self.update_block是用于更新隐藏状态的模块，它使用参数args.hidden_dims来定义隐藏层的维度。
self.context_zqr_convs是一系列卷积层，用于处理上下文特征。
self.conv2或self.fnet是网络的附加部分，取决于是否使用共享的骨干网络（backbone）。
'''
def create_conv1x1(inplanes, outplanes):
    return nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1)
def create_conv3x3(inplanes,outplanes,k=3,s=2,p=1):
    return      nn.Sequential(
                nn.Conv2d(inplanes,outplanes,kernel_size=3,stride=2,padding=1),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(inplace=True))

class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        #这里后续既然使用bn层，bias可以设为false
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)
        return output

class upsample_layer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(upsample_layer, self).__init__()
        #采用双线性插值 上采样两倍 但这里目的等下看看哦 前面的卷积好像没有改变图像大小啊
        #回答上一个问题：这里的上采样是要和上一层融合 所以需要上采样加倍 而且同时还需要改变通道数与上层相同才能后续做cat拼接
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.bn1(x)
        output = self.activation(x)
        return output
class RAFTStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        context_dims = args.hidden_dims
        '''
        cnet是特征提取的编码器网络（也是context_net）
        update_block是迭代更新模块
        context_zqr_convs是处理上下文特征的一系列卷积块

        PS: 下采样倍数downsample默认是3
        '''
        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn=args.context_norm, downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        #如果share_backbone 则context_net (cnet) 和 feature_ner(fent)相同，否则不同
        if args.shared_backbone:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, 'instance', stride=1),
                nn.Conv2d(128, 256, 3, padding=1))
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)
        
        resnet_raw_model1 = torchvision.models.resnet152(pretrained=True)
        filters = [64, 256, 512, 1024, 2048]
        self.encoder_another_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_another_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1), dim=1)
        self.encoder_another_bn1 = resnet_raw_model1.bn1
        self.encoder_another_relu = resnet_raw_model1.relu
        self.encoder_another_maxpool = resnet_raw_model1.maxpool
        self.encoder_another_layer1 = resnet_raw_model1.layer1
        self.encoder_another_layer2 = resnet_raw_model1.layer2
        self.encoder_another_layer3 = resnet_raw_model1.layer3
        self.encoder_another_layer4 = resnet_raw_model1.layer4
        ###  decoder  ###

        self.conv1_1 = conv_block_nested(filters[0]*2, filters[0], filters[0])
        self.conv2_1 = conv_block_nested(filters[1]*2, filters[1], filters[1])
        self.conv3_1 = conv_block_nested(filters[2]*2, filters[2], filters[2])
        self.conv4_1 = conv_block_nested(filters[3]*2, filters[3], filters[3])

        self.conv1_2 = conv_block_nested(filters[0]*3, filters[0], filters[0])
        self.conv2_2 = conv_block_nested(filters[1]*3, filters[1], filters[1])
        self.conv3_2 = conv_block_nested(filters[2]*3, filters[2], filters[2])

        self.conv1_3 = conv_block_nested(filters[0]*4, filters[0], filters[0])
        self.conv2_3 = conv_block_nested(filters[1]*4, filters[1], filters[1])

        self.conv1_4 = conv_block_nested(filters[0]*5, filters[0], filters[0])

        #upsample_layer(in_channel,out_channel) in_channel是这一层的filters out用上一层的filters
        self.up2_0 = upsample_layer(filters[1], filters[0])
        self.up2_1 = upsample_layer(filters[1], filters[0])
        self.up2_2 = upsample_layer(filters[1], filters[0])
        self.up2_3 = upsample_layer(filters[1], filters[0])

        self.up3_0 = upsample_layer(filters[2], filters[1])
        self.up3_1 = upsample_layer(filters[2], filters[1])
        self.up3_2 = upsample_layer(filters[2], filters[1])

        self.up4_0 = upsample_layer(filters[3], filters[2])
        self.up4_1 = upsample_layer(filters[3], filters[2])

        self.up5_0 = upsample_layer(filters[4], filters[3])
        #最终的融合是将深层特征融合到第一层，然后再做个上采样回到原分辨率，输出通道为分割标签类别数
        self.final = upsample_layer(filters[0],34)

        ### layers without pretrained model need to be initialized ###
        self.need_initialization = [self.conv1_1, self.conv2_1, self.conv3_1, self.conv4_1, self.conv1_2,
                                    self.conv2_2, self.conv3_2, self.conv1_3, self.conv2_3, self.conv1_4,
                                    self.up2_0, self.up2_1, self.up2_2, self.up2_3, self.up3_0, self.up3_1,
                                    self.up3_2, self.up4_0, self.up4_1, self.up5_0, self.final]

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    '''
    这里就是初始化视差
    '''
    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1
    '''
    低分辨率到高分辨率
    factor表示下采样的阶次 也是上采样回原分辨率的阶次
    '''
    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = flow.shape
        '''
        N 是批次大小，D 是流场维度（在立体匹配中通常是2，对应于水平和垂直位移）
        H 和 W 分别是高度和宽度
        factor 是上采样因子，由 2 的 n_downsample 次幂计算得到，表示流场的下采样倍数

        '''
        factor = 2 ** self.args.n_downsample
        '''
        mask 从形状 (N, 1, H, W) 变换为 (N, 1, 9, factor, factor, H, W)。
        掩码的形状变换反映了卷积展开的大小和上采样因子。9 对应于 3x3 的卷积核。
        上采样过程需要在每个低分辨率像素周围考虑一个 3x3 的邻域，以便在这个邻域内进行插值。
        目标掩码的维度应该反映这个 3x3 邻域
        同时考虑到上采样因子。因此，目标掩码维度将是 (N, 1, 9, factor, factor, H, W)
        '''
        '''
        mask 是一个在上采样过程中使用的权重!
        它是由网络的一个部分学习得到的，用于对低分辨率流场进行插值以产生高分辨率的流场！！
        '''
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor*H, factor*W)
    
    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """ Estimate optical flow between pair of frames """
    #归一化到[-1.1]
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            if self.args.shared_backbone:
                *cnet_list, F1,F2,F3,F4,F5,x = self.cnet(torch.cat((image1, image2), dim=0), dual_inp=True, num_layers=self.args.n_gru_layers)
                print("cnet_list length:", len(cnet_list))
                if len(cnet_list) > 0:
                    print("First element size:", len(cnet_list[0]))
                fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0]//2)
            else:
                cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.fnet([image1, image2])
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]

            # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning 
            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]

        if self.args.corr_implementation == "reg": # Default
            corr_block = CorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "alt": # More memory efficient than reg
            corr_block = PytorchAlternateCorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "reg_cuda": # Faster version of reg
            corr_block = CorrBlockFast1D
        elif self.args.corr_implementation == "alt_cuda": # Faster version of alt
            corr_block = AlternateCorrBlock
        corr_fn = corr_block(fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels)

        coords0, coords1 = self.initialize_flow(net_list[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume
            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru: # Update low-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:# Update low-res GRU and mid-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=self.args.n_gru_layers==3, iter16=True, iter08=False, update=False)
                net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, corr, flow, iter32=self.args.n_gru_layers==3, iter16=self.args.n_gru_layers>=2)

            # in stereo mode, project flow onto epipolar 视差只在水平方向变化
            delta_flow[:,1] = 0.0

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < iters-1:
                continue

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = flow_up[:,:1]

            #将每次迭代后上采样的流保存到flow_predictions列表中
            flow_predictions.append(flow_up)


        #视差估计完毕后 进行语义分割估计 融合sne_model
        another=flow_up

        convA1=create_conv3x3(inplanes=64,outplanes=64)
        convA1 = convA1.cuda()
        A1=convA1(F1.float())

        another = self.encoder_another_conv1(another)
        another = self.encoder_another_bn1(another)
        another = self.encoder_another_relu(another)
        F1F=A1+another
        x1_0=F1F
        ###
        convA2=create_conv1x1(inplanes=128,outplanes=256)
        convA2 = convA2.cuda()
        A2=convA2(F3.float())

        another=self.encoder_another_maxpool(another)
        another = self.encoder_another_layer1(another)
        F2F=A2+another
        x2_0=F2F

        ###
        convA3=create_conv1x1(inplanes=128,outplanes=512)
        convA3=convA3.cuda()
        A3=convA3(F4.float())
        another=self.encoder_another_layer2(another)
        F3F=A3+another
        x3_0 = F3F

        ###
        convA4=create_conv3x3(inplanes=512,outplanes=1024)
        convA4=convA4.cuda()
        A4=convA4(F3F.float())
        another = self.encoder_another_layer3(another)
        F4F=A4+another
        x4_0 = F4F
        
        ###
        convA5=create_conv3x3(inplanes=1024,outplanes=2048)
        convA5=convA5.cuda()
        A5=convA5(F4F.float())
        another = self.encoder_another_layer4(another)
        F5F=A5+another
        x5_0 = F5F

        # decoder
        '''
          逐层上采样 上采样的过程也有下一层的特征和上一层的特征融合 也就是下面的在channel维度进行cat拼接 拼接完成后维度变成2c
          这里也正是由于下层要与上层融合 因此upsampling需要完成两件事情： 1. 双线性插值提高分辨率与上层相同 2.卷积+bn+relu将通道数恢复（降低）
        回上一层 也就是从c（i）到c（i-1）
          这里注意使用跳跃连接 比如x1_2 融合 x1_0 x1_1 up(x2_1) 
        '''
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], dim=1))
        x4_1 = self.conv4_1(torch.cat([x4_0, self.up5_0(x5_0)], dim=1))

        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], dim=1))
        x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, self.up4_1(x4_1)], dim=1))

        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], dim=1))
        x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, self.up3_2(x3_2)], dim=1))

        x1_4 = self.conv1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, self.up2_3(x2_3)], dim=1))
        out = self.final(x1_4)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions,out

