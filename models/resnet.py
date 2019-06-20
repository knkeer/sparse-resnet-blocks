from models.resnet_blocks import conv_3x3


def make_block_sequence(block, in_channels, out_channels, stride, num_blocks, sparse=False):
    if in_channels <= out_channels:
        layers = [block(in_channels, out_channels, stride=stride, sparse=sparse)]
        for _ in range(num_blocks - 1):
            layers.append(block(out_channels, out_channels, sparse=sparse))
    else:
        layers = [block(in_channels, in_channels, sparse=sparse) for _ in range(num_blocks - 1)]
        layers.append(block(in_channels, out_channels, stride=stride, sparse=sparse))
    return layers


def make_resnet(block, num_blocks=1, channels=(16, 64, 128, 256), stride=(1, 2, 2), sparse=False):
    layers = [conv_3x3(3, channels[0], sparse=sparse)]
    layers.extend(make_block_sequence(block, channels[0], channels[1], stride[0], num_blocks, sparse))
    layers.extend(make_block_sequence(block, channels[1], channels[2], stride[1], num_blocks, sparse))
    layers.extend(make_block_sequence(block, channels[2], channels[3], stride[2], num_blocks, sparse))
    return layers
