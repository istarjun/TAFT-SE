import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
  'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
  'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',

}


def conv1x1(in_planes, out_planes, stride=1):
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
  expansion = 1

  def __init(self, inplanes, planes, stride=1, downsample=None, groups=1,
             base_width=64, dilation=1, norm_layer=None):
    super(BasicBlock, self).__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    if groups != 1 or base_width != 64:
      raise ValueError('BasicBlock only supports groups=1 and base_width=64')
    if dilation > 1:
      raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)
    out += identity
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
               base_width=64, dilation=1, norm_layer=None):
    super(Bottleneck, self).__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    width = int(planes * (base_width / 64.)) * groups
    self.conv1 = conv1x1(inplanes, width)
    self.bn1 = norm_layer(width)
    self.conv2 = conv3x3(width, width, stride, groups, dilation)
    self.bn2 = norm_layer(width)
    self.conv3 = conv1x1(width, planes * self.expansion)
    self.bn3 = norm_layer(planes * self.expansion)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
               groups=1, width_per_group=64, replace_stride_with_dilation=None,
               norm_layer=None):
    super(ResNet, self).__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    self._norm_layer = norm_layer

    self.inplanes = 64
    self.dilation = 1
    if replace_stride_with_dilation is None:
      replace_stride_with_dilation = [False, False, False]
    if len(replace_stride_with_dilation) != 3:
      raise ValueError("replace_stride_with_dilation should be None "
                       "or a 3-element tuple, got{}".format(replace_stride_with_dilation))
    self.groups = groups
    self.base_width = width_per_group
    self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = norm_layer(self.inplanes)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
    self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilate=replace_stride_with_dilation[2])

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    if zero_init_residual:
      for m in self.modules():
        if isinstance(m, Bottleneck):
          nn.init.constant_(m.bn3.weight, 0)
        elif isinstance(m, BasicBlock):
          nn.init.constant_(m.bn2.weight, 0)

  def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
    norm_layer = self._norm_layer
    downsample = None
    previous_dilation = self.dilation
    if dilate:
      self.dilation *= stride
      stride = 1
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        conv1x1(self.inplanes, planes * block.expansion, stride),
        norm_layer(planes * block.expansion),
      )
    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes, groups=self.groups,
                          base_width=self.base_width, dilation=self.dilation,
                          norm_layer=norm_layer))
    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x2 = self.layer1(x)
    x = self.layer2(x2)
    x = self.layer3(x)
    x = self.layer4(x)
    return x, x2


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
  model = ResNet(block, layers, **kwargs)
  if pretrained:
    pretrain_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
    state_dict = model.state_dict()
    model_dict = {}
    for k, v in pretrain_dict.items():
      if k in state_dict:
        model_dict[k] = v
    model.load_state_dict(model_dict)
  return model


def resnet50(pretrained=False, progress=True, **kwargs):
  return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnet101(pretrained=False, progress=True, **kwargs):
  return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


class _ASPPModule(nn.Module):
  def __init__(self, inplanes, planes, kernel_size, padding, dilation):
    super(_ASPPModule, self).__init__()
    self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                 stride=1, padding=padding, dilation=dilation, bias=False)
    self.bn = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU()

    self._init_weight()

  def forward(self, x):
    x = self.atrous_conv(x)
    x = self.bn(x)

    return self.relu(x)

  def _init_weight(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class ASPP(nn.Module):
  def __init__(self):
    super(ASPP, self).__init__()
    inplanes = 2048
    dilations = [1, 4, 7, 11]

    self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
    self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1])
    self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2])
    self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3])

    self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                         nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU())
    self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
    self.bn1 = nn.BatchNorm2d(256)
    self.relu = nn.ReLU()
    self._init_weight()

  def forward(self, x):
    x1 = self.aspp1(x)
    x2 = self.aspp2(x)
    x3 = self.aspp3(x)
    x4 = self.aspp4(x)
    x5 = self.global_avg_pool(x)
    x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
    x = torch.cat((x1, x2, x3, x4, x5), dim=1)

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    return x

  def _init_weight(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class Decoder(nn.Module):
  def __init__(self, num_classes):
    super(Decoder, self).__init__()
    low_level_inplanes = 256

    self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
    self.bn1 = nn.BatchNorm2d(48)
    self.relu = nn.ReLU()
    self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(),
                                   nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU())
    self._init_weight()

  def forward(self, x, low_level_feat):
    low_level_feat = self.conv1(low_level_feat)
    low_level_feat = self.bn1(low_level_feat)
    low_level_feat = self.relu(low_level_feat)

    x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
    x = torch.cat((x, low_level_feat), dim=1)
    x = self.last_conv(x)

    return x

  def _init_weight(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
      # elif isinstance(m, SynchronizedBatchNorm2d):
      #    m.weight.data.fill_(1)
      #    m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class ScaledDotProductAttention(nn.Module):
  ''' Scaled Dot-Product Attention '''

  def __init__(self, dim, attn_dropout=0.1):
    super().__init__()
    self.temperature = np.power(dim, 0.5)
    self.dropout = nn.Dropout(attn_dropout)
    self.softmax = nn.Softmax(dim=2)

  def forward(self, q, k, v):
    attn = torch.bmm(q, k.transpose(1, 2))
    attn = attn / self.temperature
    attn = self.softmax(attn)
    attn = self.dropout(attn)
    output = torch.bmm(attn, v)
    return output


class MultiHeadAttention(nn.Module):
  ''' Multi-Head Attention module '''

  def __init__(self, n_head, dim, dropout=0.1):
    super().__init__()
    self.n_head = n_head
    self.dim = dim
    self.w_qs = nn.Linear(dim, n_head * dim, bias=False)
    self.w_ks = nn.Linear(dim, n_head * dim, bias=False)
    self.w_vs = nn.Linear(dim, n_head * dim, bias=False)
    nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (dim + dim)))
    nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (dim + dim)))
    nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (dim + dim)))

    self.attention = ScaledDotProductAttention(dim=dim)
    self.layer_norm = nn.LayerNorm(dim)

    self.fc = nn.Linear(n_head * dim, dim)
    nn.init.xavier_normal_(self.fc.weight)
    self.dropout = nn.Dropout(dropout)

  def forward(self, input):
    d_k, d_v, n_head = self.dim, self.dim, self.n_head
    sz_b, len_q, _ = input.size()
    sz_b, len_k, _ = input.size()
    sz_b, len_v, _ = input.size()

    residual = input
    q = self.w_qs(input).view(sz_b, len_q, n_head, d_k)
    k = self.w_ks(input).view(sz_b, len_k, n_head, d_k)
    v = self.w_vs(input).view(sz_b, len_v, n_head, d_v)

    q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
    k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
    v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

    output = self.attention(q, k, v)

    output = output.view(n_head, sz_b, len_q, d_v)
    output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

    output = self.dropout(self.fc(output))
    output = self.layer_norm(output + residual)

    return output


class TAFT_SE_Deeplab(nn.Module):
  def __init__(self, encoder, nb_class, n_shot, device, n_head=1,  coco=False):
    """
        Args
            nb_class_train (int): number of classes in a training episode
            nb_class_test (int): number of classes in a test episode
            input_size (int): dimension of input vector
            dimension (int) : dimension of embedding space
            n_shot (int) : number of shots
        """
    super(TAFT_SE_Deeplab, self).__init__()
    self.nb_class = nb_class

    self.n_shot = n_shot
    if encoder == 'resnet50':
      self.encoder = resnet50(pretrained=True)
    elif encoder == 'resnet101':
      self.encoder = resnet101(pretrained=True)
    self.attention = MultiHeadAttention(n_head=n_head, dim=2048)
    self.ASPP = ASPP()
    self.ASPP2 = ASPP()
    self.decoder = Decoder(self.nb_class + 1)
    self.decoder2 = Decoder(self.nb_class + 1)

    self.phi = nn.Linear(2048, nb_class + 1, bias=True)
    self.phi2 = nn.Conv2d(256, self.nb_class + 1, kernel_size=1, stride=1)
    if coco:
      class_num = 61
    else:
      class_num = 16
    self.last_conv = nn.Conv2d(256, class_num, kernel_size=1, stride=1)
    nn.init.kaiming_normal_(self.phi.weight, a=0, mode='fan_in', nonlinearity='linear')
    nn.init.constant_(self.phi.bias, 0)
    self.device = device
    self.phi.to(self.device)
    self.phi2.to(self.device)
    self.last_conv.to(self.device)
    self.encoder.to(self.device)
    self.ASPP.to(self.device)
    self.ASPP2.to(self.device)
    self.decoder.to(self.device)
    self.decoder2.to(self.device)
    self.attention.to(self.device)
    # self.class_loss = cross_entropy_with_probs
    self.class_loss = nn.MSELoss()
    self.seg_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0]).to(self.device))
    self.seg_loss2 = nn.CrossEntropyLoss(ignore_index=255)

    self.temperature = temperature
    self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

  def set_optimizer(self, learning_rate1, learning_rate2, momentum, weight_decay_rate1, weight_decay_rate2, lrstep,
                    decay_rate):
    self.optimizer1 = optim.SGD(
      list(self.ASPP.parameters()) + list(self.ASPP2.parameters()) + list(self.decoder.parameters()) + list(self.decoder2.parameters()) + list(
        self.phi.parameters()) + list(self.attention.parameters()) + list(self.phi2.parameters()) + list(
        self.last_conv.parameters()), lr=learning_rate1, momentum=momentum,
      weight_decay=weight_decay_rate1)
    self.scheduler1 = optim.lr_scheduler.StepLR(self.optimizer1, step_size=lrstep, gamma=decay_rate)

    self.optimizer2 = optim.SGD(list(self.encoder.parameters()), lr=learning_rate2, momentum=momentum,
                                weight_decay=weight_decay_rate2)
    self.scheduler2 = optim.lr_scheduler.StepLR(self.optimizer2, step_size=lrstep, gamma=decay_rate)

  def Transformation_Matrix(self, average_key):
    c_t = average_key
    eps = 1e-6
    Phi_tmp = self.phi.weight

    # Phi_sum = Phi_tmp.sum(0)
    # Phi = (self.nb_class+1)*(Phi_tmp)-Phi_sum
    Phi = Phi_tmp
    power_Phi = ((Phi * Phi).sum(dim=1, keepdim=True)).sqrt()

    Phi = Phi / (power_Phi + eps)

    power_c = ((c_t * c_t).sum(dim=1, keepdim=True)).sqrt()
    c_tmp = c_t / (power_c + eps)
    # c_tmp = c_t

    P = torch.matmul(torch.pinverse(c_tmp), Phi)
    P = P.permute(1, 0)
    return P

  def generate_label(self, labels, support_set):
    labels_base = labels
    labels_base[labels_base == 255] = 0

    with torch.no_grad():
      labels_base = labels_base.view(labels_base.size(0), 1, labels_base.size(1), labels_base.size(2))
      labels_tmp = F.avg_pool2d(labels_base, kernel_size=16, stride=16, ceil_mode=True)

      labels_tmp2 = 1 - labels_tmp
      labels_gen = torch.cat((labels_tmp2, labels_tmp), dim=1)

    return labels_gen

  def compute_prototypes(self, labels_gen, data_aspp):
    eps = 1e-6
    for i in range(self.nb_class + 1):
      label_tmp = labels_gen[:, i]
      label_tmp = label_tmp.view(label_tmp.size(0), 1, label_tmp.size(1), label_tmp.size(2)).expand(data_aspp.size())
      data_label = data_aspp * label_tmp
      if i == 0:
        prototype = data_label.mean((2, 3))
        label_mean = label_tmp.mean((2, 3))
        prototype = prototype / (label_mean + eps)
        prototype = prototype.mean(0, keepdim=True)
      else:
        prototype_tmp = data_label.mean((2, 3))
        label_mean = label_tmp.mean((2, 3))
        prototype_tmp = prototype_tmp / (label_mean + eps)
        prototype_tmp = prototype_tmp.mean(0, keepdim=True)
        prototype = torch.cat((prototype, prototype_tmp))
    for i in range(self.nb_class + 1):
      if prototype[i].sum() == 0:
        prototype[i] = -prototype.sum(dim=0) / self.nb_class
    return prototype

  def compute_class_loss(self, label_class, r_t):
    label_class_tmp = label_class.permute(0, 2, 3, 1)
    label_class_tmp = label_class_tmp.contiguous().view(-1, label_class_tmp.size(3))
    r_t_tmp = r_t.permute(0, 2, 3, 1)
    r_t_tmp = r_t_tmp.contiguous().view(-1, r_t_tmp.size(3))
    y_t = self.phi(r_t_tmp)
    m = nn.Softmax(dim=1)
    y_t = m(y_t / self.temperature)

    return self.class_loss(y_t, label_class_tmp)

  def compute_seg_loss(self, label_seg, dec_key):
    return self.seg_loss(dec_key, label_seg.long())

  def compute_accuracy(self, t_data, dec_image):
    y_pred = dec_image.long().flatten()
    y_true = t_data.long().flatten()

    pixel_acc = (y_pred == y_true).float().mean()

    return pixel_acc

  def compute_precision(self, t_data, dec_image):
    batchsize = t_data.shape[0]
    y_pred = dec_image.long().view(batchsize, -1)
    y_true = t_data.long().view(batchsize, -1)
    tp = (y_pred & y_true).float().sum()
    fp = (y_pred & (1 - y_true)).float().sum()
    tn = ((1 - y_pred) & (1 - y_true)).float().sum()
    fn = ((1 - y_pred) & y_true).float().sum()

    return tp, fp, tn, fn

  def train(self, images, labels, labels2):
    """
        Train a minibatch of episodes
        """
    images = np.stack(images)
    images = torch.Tensor(images).to(self.device)
    labels = np.stack(labels)
    labels = torch.Tensor(labels).to(self.device)
    labels2 = np.stack(labels2)
    labels2 = torch.Tensor(labels2).to(self.device)
    loss = 0
    self.encoder.train()
    self.ASPP.train()
    self.decoder.train()
    self.attention.train()

    key, low_level_key = self.encoder(images)
    support_set = key[:self.nb_class * self.n_shot]
    query_set = key[self.nb_class * self.n_shot:]

    labels_gen = self.generate_label(labels, support_set)
    labels_support = labels_gen[:self.nb_class * self.n_shot]
    labels_query = labels_gen[self.nb_class * self.n_shot:]

    support_set_tmp = support_set.permute(0, 2, 3, 1)
    support_set_size = support_set_tmp.size()
    support_set_tmp = support_set_tmp.reshape((support_set_tmp.size(0), -1, support_set_tmp.size(3)))
    support_set_SA = self.attention(support_set_tmp)
    support_set = support_set_SA.reshape(support_set_size)
    support_set = support_set.permute(0, 3, 1, 2)
    prototype = self.compute_prototypes(labels_support, support_set)

    query_set_tmp = query_set.permute(0, 2, 3, 1)
    query_set_size = query_set_tmp.size()
    query_set_tmp = query_set_tmp.reshape((query_set_tmp.size(0), -1, query_set_tmp.size(3)))
    query_set_SA = self.attention(query_set_tmp)
    query_set = query_set_SA.reshape(query_set_size)
    query_set = query_set.permute(0, 3, 1, 2)

    P = self.Transformation_Matrix(prototype)
    weight = P.view(P.size(0), P.size(1), 1, 1)
    r_t = F.conv2d(query_set, weight)
    loss_class = self.compute_class_loss(labels_query, r_t)

    r_t = self.ASPP(r_t)
    r_t = self.decoder(r_t, low_level_key[self.nb_class * self.n_shot:])

    dec_key = self.phi2(r_t)
    dec_image = F.interpolate(dec_key, size=images.size()[2:4], mode='bilinear', align_corners=True)
    loss_seg = self.compute_seg_loss(labels[self.nb_class * self.n_shot:], dec_image)

    r_t2 = self.ASPP2(query_set)
    r_t2 = self.decoder2(r_t2, low_level_key[self.nb_class * self.n_shot:])
    dec_key2 = self.last_conv(r_t2)
    dec_image2 = F.interpolate(dec_key2, size=images.size()[2:4], mode='bilinear', align_corners=True)
    loss_seg2 = self.seg_loss2(dec_image2, labels2[self.nb_class * self.n_shot:].long())

    loss = self.alpha * loss_class + self.beta * loss_seg + loss_seg2
    self.optimizer1.zero_grad()
    self.optimizer2.zero_grad()
    loss.backward()
    self.optimizer1.step()
    self.scheduler1.step()
    self.optimizer2.step()
    self.scheduler2.step()
    return loss.data

  def evaluate(self, images, labels):
    """
            Evaluate accuracy score
            """
    images = np.stack(images)
    images = torch.Tensor(images).to(self.device)
    labels = np.stack(labels)
    labels = torch.Tensor(labels).to(self.device)

    self.encoder.eval()
    self.ASPP.eval()
    self.decoder.eval()
    self.attention.eval()

    with torch.no_grad():
      key, low_level_key = self.encoder(images)
      support_set = key[:self.nb_class * self.n_shot]
      query_set = key[self.nb_class * self.n_shot:]

      labels_gen = self.generate_label(labels, support_set)
      labels_support = labels_gen[:self.nb_class * self.n_shot]

      support_set_tmp = support_set.permute(0, 2, 3, 1)
      support_set_size = support_set_tmp.size()
      support_set_tmp = support_set_tmp.reshape((support_set_tmp.size(0), -1, support_set_tmp.size(3)))
      suppot_set_SA = self.attention(support_set_tmp)
      support_set = suppot_set_SA.reshape(support_set_size)
      support_set = support_set.permute(0, 3, 1, 2)

      prototype = self.compute_prototypes(labels_support, support_set)

      query_set_tmp = query_set.permute(0, 2, 3, 1)
      query_set_size = query_set_tmp.size()
      query_set_tmp = query_set_tmp.reshape((query_set_tmp.size(0), -1, query_set_tmp.size(3)))
      query_set_SA = self.attention(query_set_tmp)
      query_set = query_set_SA.reshape(query_set_size)
      query_set = query_set.permute(0, 3, 1, 2)

      P = self.Transformation_Matrix(prototype)
      weight = P.view(P.size(0), P.size(1), 1, 1)
      r_t = F.conv2d(query_set, weight)

      r_t = self.ASPP(r_t)
      r_t = self.decoder(r_t, low_level_key[self.nb_class * self.n_shot:])

      dec_key = self.phi2(r_t)
      dec_image = F.interpolate(dec_key, size=images.size()[2:4], mode='bilinear', align_corners=True)
      dec_image = torch.argmax(dec_image, dim=1, keepdim=True)
      pixel_acc = self.compute_accuracy(labels[self.nb_class * self.n_shot:], dec_image)
      tp, fp, tn, fn = self.compute_precision(labels[self.nb_class * self.n_shot:], dec_image)

    return pixel_acc, tp, fp, tn, fn

