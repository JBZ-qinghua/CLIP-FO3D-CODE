from clip_fo3d.model import BasicBlock, Mink16UNet34C, build_model


def test_mink16unet34c_topology_and_parameter_count():
    model = build_model(
        {
            "architecture": "Mink16UNet34C",
            "in_channels": 3,
            "out_channels": 512,
            "base_channels": 32,
            "channels": [32, 64, 128, 256, 256, 128, 96, 96],
            "layers": [2, 3, 4, 6, 2, 2, 2, 2],
        }
    )
    assert model.PLANES == (32, 64, 128, 256, 256, 128, 96, 96)
    assert model.LAYERS == (2, 3, 4, 6, 2, 2, 2, 2)
    assert sum(isinstance(module, BasicBlock) for module in model.modules()) == 23
    assert sum(parameter.numel() for parameter in model.parameters()) == 37_903_776
    assert isinstance(model, Mink16UNet34C)
