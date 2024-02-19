import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1
import pytorch_lightning as pl
from modules.utils import accuracy_fn, loss_fn


class CoarseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels):
        super(CoarseBlock, self).__init__()

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.linear(x)
        return x


class HResNet(pl.LightningModule):
    def __init__(
        self,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes: int = 1000,
        learning_rate: float = 1e-3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes[2])

        self.learning_rate = learning_rate
        self.weights = [0.8, 0.1, 0.1]
        self.coarse1_block = CoarseBlock(64*64*64, num_classes[0])
        self.coarse2_block = CoarseBlock(512 * 16 * 16, num_classes[1])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
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
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        c1 = self.coarse1_block(x)
        x = self.layer2(x)
        c2 = self.coarse2_block(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        fine = self.fc(x)

        return c1, c2, fine

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        images, labels = train_batch
        c1, c2, fine = self(images)
        labels_arr = [
            labels[:, :c1.shape[1]],
            labels[:, c1.shape[1]:c1.shape[1]+c2.shape[1]],
            labels[:, c1.shape[1]+c2.shape[1]:]
        ]

        # Compute loss
        loss = loss_fn(self.weights, c1, c2, fine, labels_arr)

        # Compute accuracy
        accuracy = accuracy_fn(fine, labels_arr[-1])

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'accuracy': accuracy}

    def validation_step(self, val_batch, batch_idx):
        images, labels = val_batch
        c1, c2, fine = self(images)
        labels_arr = [
            labels[:, 0:c1.shape[1]],
            labels[:, c1.shape[1]:c1.shape[1]+c2.shape[1]],
            labels[:, c1.shape[1]+c2.shape[1]:]
        ]

        # Compute loss
        loss = loss_fn(self.weights, c1, c2, fine, labels_arr)

        # Compute accuracy
        accuracy = accuracy_fn(fine, labels_arr[-1])

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_accuracy': accuracy}

    def on_train_epoch_end(self) -> None:
        if self.current_epoch == 5:
            self.weights = [0.1, 0.8, 0.1]
        elif self.current_epoch == 10:
            self.weights = [0.1, 0.1, 0.8]
