# coding:utf-8
# By Zhen Feng, Feb. 2, 2023
# Email: zfeng94@outlook.com

import torch
import torch.nn as nn 
import torchvision.models as models 
from torch import nn, einsum
from einops import rearrange, repeat
import torch.nn.functional as F

class Student_model(nn.Module):

    def __init__(self, n_class):
        super(Student_model, self).__init__()

        self.num_resnet_layers = 50

        if self.num_resnet_layers == 18:
            resnet_raw_model1 = models.resnet18(pretrained=True)
            self.inplanes = 512
        elif self.num_resnet_layers == 34:
            resnet_raw_model1 = models.resnet34(pretrained=True)
            self.inplanes = 512
        elif self.num_resnet_layers == 50:
            resnet_raw_model1 = models.resnet50(pretrained=True)
            self.inplanes = 2048
        elif self.num_resnet_layers == 101:
            resnet_raw_model1 = models.resnet101(pretrained=True)
            self.inplanes = 2048
        elif self.num_resnet_layers == 152:
            resnet_raw_model1 = models.resnet152(pretrained=True)
            self.inplanes = 2048

        ########  Thermal ENCODER  ########
 
        self.encoder_thermal_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        self.encoder_thermal_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1), dim=1)
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool
        self.encoder_thermal_layer1 = resnet_raw_model1.layer1
        self.encoder_thermal_layer2 = resnet_raw_model1.layer2
        self.encoder_thermal_layer3 = resnet_raw_model1.layer3
        self.encoder_thermal_layer4 = BottleStack(dim=1024,fmap_size=(30,40),dim_out=2048,proj_factor = 4,num_layers=3,heads=4,dim_head=512)

        self.createMask=createMask(enlightening=2)

        ########  DECODER  ########

        self.deconv1 = self._make_transpose_layer(TransBottleneck, self.inplanes//2, 2, stride=2) # using // for python 3.6
        self.deconv2 = self._make_transpose_layer(TransBottleneck, self.inplanes//2, 2, stride=2) # using // for python 3.6
        self.deconv3 = self._make_transpose_layer(TransBottleneck, self.inplanes//2, 2, stride=2) # using // for python 3.6
        self.deconv4 = self._make_transpose_layer(TransBottleneck, self.inplanes//2, 2, stride=2) # using // for python 3.6
        self.deconv5 = self._make_transpose_layer(TransBottleneck, n_class, 2, stride=2)
 

        self.skip_tranform = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)


    def _make_transpose_layer(self, block, planes, blocks, stride=1):

        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes, kernel_size=2, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes),
            ) 
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes),
            ) 
 
        for m in upsample.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)
 
    def forward(self, thermal):


        #thermal = input[:,3:]


        mask2x,mask4x,mask8x,mask16x,mask32x = self.createMask(thermal)


        verbose = False

        # encoder

        ######################################################################

        if verbose: print("thermal.size() original: ", thermal.size()) # (480, 640)

        ######################################################################

        thermal = self.encoder_thermal_conv1(thermal)
        if verbose: print("thermal.size() after conv1: ", thermal.size()) # (240, 320)
        thermal = self.encoder_thermal_bn1(thermal)
        if verbose: print("thermal.size() after bn1: ", thermal.size()) # (240, 320)
        thermal = self.encoder_thermal_relu(thermal)
        if verbose: print("thermal.size() after relu: ", thermal.size())  # (240, 320)


        skip1 = thermal



        thermal = self.encoder_thermal_maxpool(thermal)
        if verbose: print("thermal.size() after maxpool: ", thermal.size()) # (120, 160)

        ######################################################################

        thermal = self.encoder_thermal_layer1(thermal)
        if verbose: print("thermal.size() after layer1: ", thermal.size()) # (120, 160)

        skip2 = thermal

        ######################################################################
 

        thermal = self.encoder_thermal_layer2(thermal)
        if verbose: print("thermal.size() after layer2: ", thermal.size()) # (60, 80)


        skip3 = thermal

        ######################################################################


        thermal = self.encoder_thermal_layer3(thermal)
        if verbose: print("thermal.size() after layer3: ", thermal.size()) # (30, 40)

        skip4 = thermal

        ######################################################################


        thermal = self.encoder_thermal_layer4(thermal)
        if verbose: print("thermal.size() after layer4: ", thermal.size()) # (15, 20)

        skip5 = thermal
        
        ######################################################################

        # decoder
        fuse = skip5+mask32x
        fuse = self.deconv1(fuse)
        fuse = fuse+skip4+mask16x
        if verbose: print("fuse after deconv1: ", fuse.size()) # (30, 40)
        fuse = self.deconv2(fuse)
        fuse = fuse+skip3+mask8x
        if verbose: print("fuse after deconv2: ", fuse.size()) # (60, 80)
        fuse = self.deconv3(fuse)
        hint = fuse+skip2
        fuse = hint+mask4x
        if verbose: print("fuse after deconv3: ", fuse.size()) # (120, 160)
        fuse = self.deconv4(fuse)
        skip1 = self.skip_tranform(skip1)
        fuse = fuse+skip1+mask2x
        if verbose: print("fuse after deconv4: ", fuse.size()) # (240, 320)
        fuse = self.deconv5(fuse)
        if verbose: print("fuse after deconv5: ", fuse.size()) # (480, 640)

        return skip5,fuse,hint
  
class TransBottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(TransBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)  
        self.bn2 = nn.BatchNorm2d(planes)

        if upsample is not None and stride != 1:
            self.conv3 = nn.ConvTranspose2d(planes, planes, kernel_size=2, stride=stride, padding=0, bias=False)  
        else:
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)  

        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        heads = 4,
        dim_head = 128,
        rel_pos_emb = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        #rel_pos_class = AbsPosEmb if not rel_pos_emb else RelPosEmb
        #rel_pos_class = AbsPosEmb
        #self.pos_emb = rel_pos_class(fmap_size, dim_head)
        self.pos_emb = AbsPosEmb(fmap_size, dim_head)
        

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        q, k, v = self.to_qkv(fmap).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        sim += self.pos_emb(q)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)

        return out

class AbsPosEmb(nn.Module):
    def __init__(
        self,
        fmap_size,
        dim_head
    ):
        super().__init__()
        scale = dim_head ** -0.5
        self.scale = scale
        self.height = nn.Parameter(torch.randn(fmap_size[0], dim_head) * scale)
        self.width = nn.Parameter(torch.randn(fmap_size[1], dim_head) * scale)

    def forward(self, q):
        emb = rearrange(self.height, 'h d -> h () d') + rearrange(self.width, 'w d -> () w d')
        emb = rearrange(emb, ' h w d -> (h w) d')
        logits = einsum('b h i d, j d -> b h i j', q, emb) * self.scale
        return logits

class BottleBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        dim_out,
        proj_factor,
        downsample,
        heads = 4,
        dim_head = 128,
        rel_pos_emb = False,
        activation = nn.ReLU()
    ):
        super().__init__()

        # shortcut

        if dim != dim_out or downsample:  #di yi bian de shi hou zhi xing
            kernel_size, stride, padding = (3, 2, 1) if downsample else (1, 1, 0)

            self.shortcut = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size, stride = stride, padding = padding, bias = False),
                nn.BatchNorm2d(dim_out),
                activation
            )
        else:
            self.shortcut = nn.Identity()

        # contraction and expansion

        attention_dim = dim_out // proj_factor



        self.net = nn.Sequential(
            nn.Conv2d(dim, attention_dim, 1, bias = False),
            nn.BatchNorm2d(attention_dim),
            activation,
            Attention(
                dim = attention_dim,
                fmap_size = fmap_size,
                heads = heads,
                dim_head = dim_head,
                rel_pos_emb = rel_pos_emb
            ),
            nn.AvgPool2d((2, 2)) if downsample else nn.Identity(),
            nn.BatchNorm2d(heads*dim_head),
            activation,
            nn.Conv2d(heads*dim_head, dim_out, 1, bias = False),
            nn.BatchNorm2d(dim_out)
        )

        # init last batch norm gamma to zero

        nn.init.zeros_(self.net[-1].weight)

        # final activation

        self.activation = activation

    def forward(self, x):

        
        shortcut = self.shortcut(x)


        x = self.net(x)



        x += shortcut
        return self.activation(x)

# main bottle stack

class BottleStack(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        dim_out = 2048,
        proj_factor = 4,
        num_layers = 3,
        heads = 4,
        dim_head = 128,
        downsample = True,
        rel_pos_emb = False,
        activation = nn.ReLU()
    ):
        super().__init__()
        self.dim = dim
        self.fmap_size = fmap_size

        layers = []

        for i in range(num_layers):
            is_first = i == 0
            dim = (dim if is_first else dim_out)
            layer_downsample = is_first and downsample
            #layer_fmap_size = fmap_size
            layer_fmap_size = (fmap_size[0] // (2 if downsample and not is_first else 1),fmap_size[1] // (2 if downsample and not is_first else 1))
            #layer_fmap_size = fmap_size[1] // (2 if downsample and not is_first else 1)
            layers.append(BottleBlock(
                dim = dim,
                fmap_size = layer_fmap_size,
                dim_out = dim_out,
                proj_factor = proj_factor,
                heads = heads,
                dim_head = dim_head,
                downsample = layer_downsample,
                rel_pos_emb = rel_pos_emb,
                activation = activation
            ))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        _, c, h, w = x.shape
        assert c == self.dim, f'channels of feature map {c} must match channels given at init {self.dim}'
        assert h == self.fmap_size[0] and w == self.fmap_size[1], f'height and width of feature map must match the fmap_size given at init {self.fmap_size}'
        return self.net(x)


class expWeightV2(nn.Module):
    def __init__(self):
        super(expWeightV2,self).__init__()

        self.maxpool1 = nn.MaxPool2d(kernel_size=4,stride=4)

        self.conv1 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=1)
        self.relu1 = nn.ReLU()

        self.maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv2 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=2,padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=1)
        self.relu2 = nn.ReLU()

        self.maxpool3 = nn.MaxPool2d(kernel_size=2,stride=2,padding=(1,0))

        self.fc1 = nn.Linear(in_features=80,out_features=20)
        self.bn3 = nn.BatchNorm1d(num_features=20)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=20,out_features=1)


    def forward(self,input):
        b,c,h,w = input.size()
        x = self.maxpool1(input)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.maxpool2(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.maxpool3(x)

        x = x.reshape(b,-1)

        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x

class createMask(nn.Module):
    def __init__(self,enlightening):
        super(createMask,self).__init__()
        self.enlightening = enlightening
        self.expweight = expWeightV2()

    def forward(self,input):
        #print(input.size())
        w = self.expweight(input)+self.enlightening
        w = w.unsqueeze(1)
        w = w.unsqueeze(1)

        input = torch.pow(input,w)
        #input = input.pow(w)

        mask2x = F.interpolate(input, scale_factor=0.5)
        mask4x = F.interpolate(input, scale_factor=0.25)
        mask8x = F.interpolate(input, scale_factor=0.125)
        mask16x = F.interpolate(input, scale_factor=0.0625)
        mask32x = F.interpolate(input, scale_factor=0.03125)

        return mask2x,mask4x,mask8x,mask16x,mask32x


def unit_test():
    num_minibatch = 2
    rgb = torch.randn(num_minibatch, 3, 480, 640).cuda(0)
    thermal = torch.randn(num_minibatch, 1, 480, 640).cuda(0)
    rtf_net = Student_model(9).cuda(0)
    input = torch.cat((rgb, thermal), dim=1)
    rtf_net(input)
    #print('The model: ', rtf_net.modules)

if __name__ == '__main__':
    unit_test()
