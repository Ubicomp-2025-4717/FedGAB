import torch
import torch.nn as nn
# from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def Unfreeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = True

class MultiModalDetector(nn.Module):
    def __init__(self, features_1,features_2,features_3,num_classes=1000, init_weights=True):
        super(MultiModalDetector, self).__init__()
        self.feature_extractor_1 = features_1
        self.feature_extractor_2 = features_2
        self.feature_extractor_3 = features_3
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self.classifier_1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

        self.classifier_2 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

        self.classifier_3 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )


        if init_weights:
            self._initialize_weights()

    def forward(self,x):
        x_1,x_2,x_3=x
        x_1 = self.feature_extractor_1(x_1)
        x_1 = self.avgpool(x_1)
        x_1 = torch.flatten(x_1,1)

        x_2 = self.feature_extractor_2(x_2)
        x_2 = self.avgpool(x_2)
        x_2 = torch.flatten(x_2,1)

        x_3 = self.feature_extractor_2(x_3)
        x_3 = self.avgpool(x_3)
        x_3 = torch.flatten(x_3,1)

        x= x_1+x_2+x_3

        outputs_1 = self.classifier_1(x_1)
        outputs_2 = self.classifier_2(x_2)
        outputs_3 = self.classifier_3(x_3)

        x = self.classifier(x)

        return x, outputs_1,outputs_2,outputs_3

        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def freeze_backbone(self):

        for param in self.feature_extractor_1.parameters():
            param.requires_grad = False

        for param in self.feature_extractor_2.parameters():
            param.requires_grad = False

        for param in self.feature_extractor_3.parameters():
            param.requires_grad = False

    def freeze_backbone_r(self):

        for param in self.feature_extractor_1.parameters():
            param.requires_grad = False

    def freeze_backbone_i(self):

        for param in self.feature_extractor_2.parameters():
            param.requires_grad = False

    def freeze_backbone_a(self):

        for param in self.feature_extractor_3.parameters():
            param.requires_grad = False


    def Unfreeze_backbone(self):

        for param in self.feature_extractor_1.parameters():
            param.requires_grad = True

        for param in self.feature_extractor_2.parameters():
            param.requires_grad = True


class MultiModalDetectorATTDOT(nn.Module):
    def __init__(self, features_1, features_2, num_classes=1000, init_weights=True):
        super(MultiModalDetectorATTDOT, self).__init__()
        self.linear_t = nn.Linear(512 * 7 * 7, 512)
        self.linear_v = nn.Linear(512 * 7 * 7, 512)
        self.feature_extractor_1 = features_1
        self.feature_extractor_2 = features_2

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7+512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self.classifier_t = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2),
        )
        self.classifier_v = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # print("x",x.size())
        x_1, x_2 = x
        # print("x_1",x_1.size())
        x_1 = self.feature_extractor_1(x_1)
        x_1 = self.avgpool(x_1)
        x_1 = torch.flatten(x_1, 1)

        x_t = F.relu(self.linear_t(x_1))
        # print(x_t.size())

        x_2 = self.feature_extractor_2(x_2)
        x_2 = self.avgpool(x_2)
        x_2 = torch.flatten(x_2, 1)

        x_v = F.relu(self.linear_v(x_2))
        # print(x_v.size())

        # Compute attention weights using dot product
        # Expand dimensions to make shapes (batch_size, feature_dim, 1) and (batch_size, 1, feature_dim)
        feature_t_expanded = x_t.unsqueeze(2)  # (batch_size, feature_dim, 1)
        feature_v_expanded = x_v.unsqueeze(1)  # (batch_size, 1, feature_dim)

        # Compute dot product
        attention_weights = torch.bmm(feature_t_expanded, feature_v_expanded)  # (batch_size, feature_dim, feature_dim)

        # Normalize attention weights
        attention_weights = F.softmax(attention_weights.view(-1,512), dim=1).view(-1,512,512)

        # Compute attended features
        C_t = torch.bmm(attention_weights, x_v.unsqueeze(2)).squeeze(2)  # (batch_size, feature_dim)
        C_v = torch.bmm(attention_weights.transpose(1, 2), x_t.unsqueeze(2)).squeeze(2)  # (batch_size, feature_dim)

        c = C_t+C_v

        x = x_1 + x_2

        x=torch.cat((x,c),dim=1)

        outputs_t=self.classifier_t(x_1)
        outputs_v = self.classifier_v(x_2)

        x = self.classifier(x)
        return x,outputs_t,outputs_v

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def freeze_backbone(self):

        for param in self.feature_extractor_1.parameters():
            param.requires_grad = False

        for param in self.feature_extractor_2.parameters():
            param.requires_grad = False

    def Unfreeze_backbone(self):

        for param in self.feature_extractor_1.parameters():
            param.requires_grad = True

        for param in self.feature_extractor_2.parameters():
            param.requires_grad = True

class MultiModalDetectorATTADD(nn.Module):
    def __init__(self, features_1, features_2, num_classes=1000, init_weights=True):
        super(MultiModalDetectorATTADD, self).__init__()
        self.linear_t = nn.Linear(512 * 7 * 7, 512)
        self.linear_v = nn.Linear(512 * 7 * 7, 512)
        self.v = nn.Parameter(torch.randn(512))

        self.feature_extractor_1 = features_1
        self.feature_extractor_2 = features_2

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7+512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # print("x",x.size())
        x_1, x_2 = x
        # print("x_1",x_1.size())
        x_1 = self.feature_extractor_1(x_1)
        x_1 = self.avgpool(x_1)
        x_1 = torch.flatten(x_1, 1)

        x_t = F.relu(self.linear_t(x_1))
        # print(x_t.size())

        x_2 = self.feature_extractor_2(x_2)
        x_2 = self.avgpool(x_2)
        x_2 = torch.flatten(x_2, 1)

        x_v = F.relu(self.linear_v(x_2))
        # print(x_v.size())


        # Additive attention scores
        scores = torch.tanh(x_t + x_v)  # (batch_size, hidden_dim)

        # Compute attention weights
        attention_weights = torch.matmul(scores, self.v)  # (batch_size,)
        attention_weights = F.softmax(attention_weights, dim=0)  # (batch_size,)

        # Weighted sum of features
        C_t = attention_weights.unsqueeze(1) * x_1  # (batch_size, feature_dim)
        C_v = attention_weights.unsqueeze(1) * x_2 # (batch_size, feature_dim)

        x = x_1+ C_v + C_t + x_2

        x = self.classifier(x)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def freeze_backbone(self):

        for param in self.feature_extractor_1.parameters():
            param.requires_grad = False

        for param in self.feature_extractor_2.parameters():
            param.requires_grad = False

    def Unfreeze_backbone(self):

        for param in self.feature_extractor_1.parameters():
            param.requires_grad = True

        for param in self.feature_extractor_2.parameters():
            param.requires_grad = True

class MultiModalDetectorATTHEAD(nn.Module):
    def __init__(self, features_1, features_2, num_classes=1000, init_weights=True):
        super(MultiModalDetectorATTHEAD, self).__init__()

        self.feature_dim = 512  # 特征维度
        self.num_heads = 8      # 注意力头的数量
        self.head_dim = 512 // 8  # 每个注意力头的维度
        self.linear_t = nn.Linear(512 * 7 * 7, 512)
        self.linear_v = nn.Linear(512 * 7 * 7, 512)

        # 确保特征维度可以被注意力头数量整除
        assert self.head_dim * 8 == 512, "feature_dim must be divisible by num_heads"

        # 定义线性变换层
        self.q_linear = nn.Linear(512, 512)
        self.k_linear = nn.Linear(512, 512)
        self.v_linear = nn.Linear(512, 512)
        self.fc = nn.Linear(512, 512*7*7)  # 最后的线性层


        self.feature_extractor_1 = features_1
        self.feature_extractor_2 = features_2

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7+512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # print("x",x.size())
        x_1, x_2 = x
        # print("x_1",x_1.size())
        x_1 = self.feature_extractor_1(x_1)
        x_1 = self.avgpool(x_1)
        x_1 = torch.flatten(x_1, 1)

        x_t = F.relu(self.linear_t(x_1))
        # print(x_t.size())

        x_2 = self.feature_extractor_2(x_2)
        x_2 = self.avgpool(x_2)
        x_2 = torch.flatten(x_2, 1)

        x_v = F.relu(self.linear_v(x_2))
        # print(x_v.size())

        batch_size = x_t.size(0)  # 批量大小

        # 线性变换：Q、K、V
        Q = self.q_linear(x_t).view(batch_size, self.num_heads, self.head_dim)
        K = self.k_linear(x_v).view(batch_size, self.num_heads, self.head_dim)
        V = self.v_linear(x_v).view(batch_size, self.num_heads, self.head_dim)

        # 计算注意力分数
        scores = torch.einsum("bhd,bhd->bh", Q, K) / self.head_dim ** 0.5
        attention_weights = F.softmax(scores, dim=-1)  # 对注意力分数进行 softmax 归一化

        # 加权求和
        attended_features = torch.einsum("bh,bhd->bhd", attention_weights, V).contiguous()

        # 合并注意力头
        attended_features = attended_features.view(batch_size, -1)  # 展平注意力头维度

        # 最后的线性变换层
        output = self.fc(attended_features)



        x = x_1+ output + x_2

        x = self.classifier(x)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def freeze_backbone(self):

        for param in self.feature_extractor_1.parameters():
            param.requires_grad = False

        for param in self.feature_extractor_2.parameters():
            param.requires_grad = False

    def Unfreeze_backbone(self):

        for param in self.feature_extractor_1.parameters():
            param.requires_grad = True

        for param in self.feature_extractor_2.parameters():
            param.requires_grad = True




def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# 224,224,3 -> 224,224,64 -> 112,112,64 -> 112,112,128 -> 56,56,128 -> 56,56,256 -> 28,28,256 -> 28,28,512
# 14,14,512 -> 14,14,512 -> 7,7,512
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg11(pretrained=False, progress=True, num_classes=1000):
    model = VGG(make_layers(cfgs['A']))
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['vgg11'], model_dir='./model_data',
                                              progress=progress)
        model.load_state_dict(state_dict,strict=False)

    if num_classes!=1000:
        model.classifier =  nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    return model

def vgg13(pretrained=False, progress=True, num_classes=1000):
    model = VGG(make_layers(cfgs['B']))
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['vgg13'], model_dir='./model_data',
                                              progress=progress)
        model.load_state_dict(state_dict,strict=False)

    if num_classes!=1000:
        model.classifier =  nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    return model

def vgg16(pretrained=False, progress=True, num_classes=1000):
    model = VGG(make_layers(cfgs['D']))
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['vgg16'], model_dir='./model_data',
                                              progress=progress)
        model.load_state_dict(state_dict,strict=False)

    if num_classes!=1000:
        model.classifier =  nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    return model

def vgg11_bn(pretrained=False, progress=True, num_classes=1000):
    model = VGG(make_layers(cfgs['A'], True))
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['vgg11_bn'], model_dir='./model_data',
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)

    if num_classes!=1000:
        model.classifier =  nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    return model

def vgg13_bn(pretrained=False, progress=True, num_classes=1000):
    model = VGG(make_layers(cfgs['B'], True))
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['vgg13_bn'], model_dir='./model_data',
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)

    if num_classes!=1000:
        model.classifier =  nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    return model

def vgg16_bn(pretrained=False, progress=True, num_classes=1000):
    model = VGG(make_layers(cfgs['D'], True))
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['vgg16_bn'], model_dir='./model_data',
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)

    if num_classes!=1000:
        model.classifier =  nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    return model

def MD(pretrained=False, progress=True, num_classes=1000):
    model = MultiModalDetector(make_layers(cfgs['A']),make_layers(cfgs['A']),make_layers(cfgs["A"]))
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['vgg16'], model_dir='./model_data',
                                              progress=progress)
        model.load_state_dict(state_dict,strict=False)

    if num_classes!=1000:
        model.classifier =  nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    return model

def MDATTDOT(pretrained=False, progress=True, num_classes=1000):
    model = MultiModalDetectorATTDOT(make_layers(cfgs['D']),make_layers(cfgs['D']))
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['vgg16'], model_dir='./model_data',
                                              progress=progress)
        model.load_state_dict(state_dict,strict=False)

    if num_classes!=1000:
        model.classifier =  nn.Sequential(
            nn.Linear(512 * 7 * 7+512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    return model

def MDATTADD(pretrained=False, progress=True, num_classes=1000):
    model = MultiModalDetectorATTADD(make_layers(cfgs['D']),make_layers(cfgs['D']))
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['vgg16'], model_dir='./model_data',
                                              progress=progress)
        model.load_state_dict(state_dict,strict=False)

    if num_classes!=1000:
        model.classifier =  nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    return model

def MDATTHEAD(pretrained=False, progress=True, num_classes=1000):
    model = MultiModalDetectorATTADD(make_layers(cfgs['D']),make_layers(cfgs['D']))
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['vgg16'], model_dir='./model_data',
                                              progress=progress)
        model.load_state_dict(state_dict,strict=False)

    if num_classes!=1000:
        model.classifier =  nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    return model