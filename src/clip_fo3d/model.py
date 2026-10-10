"""Mink16UNet34C backbone implemented with the official MinkowskiEngine API.

The architecture follows the ScanNet ``Res16UNet34C``/``Mink16UNet34C``
definition released with Choy et al.'s SpatioTemporalSegmentation project and
the current MinkowskiEngine ``MinkUNet34C`` example.  CLIP-FO3D replaces the
semantic classifier with a 512-dimensional feature projection.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    import MinkowskiEngine as ME
except ImportError as exc:  # pragma: no cover - environment diagnostic
    raise RuntimeError(
        "MinkowskiEngine 0.5.4 with CUDA support is required for Mink16UNet34C"
    ) from exc


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
        dimension: int = 3,
    ) -> None:
        super().__init__()
        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride,
            dilation=dilation, bias=False, dimension=dimension,
        )
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=0.1)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1,
            dilation=dilation, bias=False, dimension=dimension,
        )
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=0.1)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class Mink16UNet34C(nn.Module):
    """Official ScanNet Res16UNet34C topology with a 512-D output head."""

    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    INIT_DIM = 32

    def __init__(self, in_channels: int = 3, out_channels: int = 512, dimension: int = 3) -> None:
        super().__init__()
        self.dimension = dimension
        self.inplanes = self.INIT_DIM

        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, bias=False, dimension=dimension
        )
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes, momentum=0.1)
        self.relu = ME.MinkowskiReLU(inplace=True)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2,
            bias=False, dimension=dimension,
        )
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes, momentum=0.1)
        self.block1 = self._make_layer(self.PLANES[0], self.LAYERS[0])
        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2,
            bias=False, dimension=dimension,
        )
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes, momentum=0.1)
        self.block2 = self._make_layer(self.PLANES[1], self.LAYERS[1])
        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2,
            bias=False, dimension=dimension,
        )
        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes, momentum=0.1)
        self.block3 = self._make_layer(self.PLANES[2], self.LAYERS[2])
        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2,
            bias=False, dimension=dimension,
        )
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes, momentum=0.1)
        self.block4 = self._make_layer(self.PLANES[3], self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2,
            bias=False, dimension=dimension,
        )
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4], momentum=0.1)
        self.inplanes = self.PLANES[4] + self.PLANES[2]
        self.block5 = self._make_layer(self.PLANES[4], self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2,
            bias=False, dimension=dimension,
        )
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5], momentum=0.1)
        self.inplanes = self.PLANES[5] + self.PLANES[1]
        self.block6 = self._make_layer(self.PLANES[5], self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2,
            bias=False, dimension=dimension,
        )
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6], momentum=0.1)
        self.inplanes = self.PLANES[6] + self.PLANES[0]
        self.block7 = self._make_layer(self.PLANES[6], self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2,
            bias=False, dimension=dimension,
        )
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7], momentum=0.1)
        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.PLANES[7], self.LAYERS[7])

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7], out_channels, kernel_size=1, bias=True, dimension=dimension
        )
        self._initialize_weights()

    def _make_layer(self, planes: int, blocks: int) -> nn.Sequential:
        downsample = None
        if self.inplanes != planes:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes, planes, kernel_size=1,
                    bias=False, dimension=self.dimension,
                ),
                ME.MinkowskiBatchNorm(planes, momentum=0.1),
            )
        layers = [BasicBlock(self.inplanes, planes, downsample=downsample, dimension=self.dimension)]
        self.inplanes = planes
        layers.extend(BasicBlock(planes, planes, dimension=self.dimension) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(module.kernel, mode="fan_out", nonlinearity="relu")
            if isinstance(module, ME.MinkowskiBatchNorm):
                nn.init.ones_(module.bn.weight)
                nn.init.zeros_(module.bn.bias)

    def _forward_sparse(self, x):
        out_p1 = self.relu(self.bn0(self.conv0p1s1(x)))
        out_b1p2 = self.block1(self.relu(self.bn1(self.conv1p1s2(out_p1))))
        out_b2p4 = self.block2(self.relu(self.bn2(self.conv2p2s2(out_b1p2))))
        out_b3p8 = self.block3(self.relu(self.bn3(self.conv3p4s2(out_b2p4))))
        out = self.block4(self.relu(self.bn4(self.conv4p8s2(out_b3p8))))

        out = self.relu(self.bntr4(self.convtr4p16s2(out)))
        out = self.block5(ME.cat(out, out_b3p8))
        out = self.relu(self.bntr5(self.convtr5p8s2(out)))
        out = self.block6(ME.cat(out, out_b2p4))
        out = self.relu(self.bntr6(self.convtr6p4s2(out)))
        out = self.block7(ME.cat(out, out_b1p2))
        out = self.relu(self.bntr7(self.convtr7p2s2(out)))
        out = self.block8(ME.cat(out, out_p1))
        return self.final(out)

    def forward(
        self, grid_coord: torch.Tensor, features: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        coordinates = torch.cat((batch[:, None].int(), grid_coord.int()), dim=1).contiguous()
        sparse_input = ME.SparseTensor(features=features, coordinates=coordinates)
        output = self._forward_sparse(sparse_input)
        if output.F.shape[0] != features.shape[0]:
            raise RuntimeError(
                f"Mink16UNet34C changed the number of stride-1 voxels: "
                f"{features.shape[0]} -> {output.F.shape[0]}"
            )
        # A stride-1 sparse output normally reuses the input coordinate map.
        # Keep a safe fallback so features always align with dataset targets,
        # even if a MinkowskiEngine build returns the coordinate map reordered.
        if output.coordinate_map_key == sparse_input.coordinate_map_key:
            return output.F
        return output.features_at_coordinates(coordinates.to(output.F.device).float())


def build_model(config: dict) -> Mink16UNet34C:
    architecture = config.get("architecture", "Mink16UNet34C")
    if architecture not in {"Mink16UNet34C", "Res16UNet34C", "MinkUNet34C"}:
        raise ValueError(f"Unsupported paper-backbone mapping: {architecture}")
    expected_channels = list(Mink16UNet34C.PLANES)
    expected_layers = list(Mink16UNet34C.LAYERS)
    if int(config.get("base_channels", Mink16UNet34C.INIT_DIM)) != Mink16UNet34C.INIT_DIM:
        raise ValueError(f"Mink16UNet34C base_channels must be {Mink16UNet34C.INIT_DIM}")
    if "channels" in config and list(config["channels"]) != expected_channels:
        raise ValueError(f"Mink16UNet34C channels must be {expected_channels}")
    if "layers" in config and list(config["layers"]) != expected_layers:
        raise ValueError(f"Mink16UNet34C layers must be {expected_layers}")
    return Mink16UNet34C(
        in_channels=int(config["in_channels"]),
        out_channels=int(config["out_channels"]),
        dimension=3,
    )
