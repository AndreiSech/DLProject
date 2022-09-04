import torch.nn as nn

def vgg_block(num_convs, input_channels, num_channels):

    block = nn.Sequential(
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )

    for i in range(num_convs - 1):
        block.add_module("conv{}".format(i),
                         nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
                         )
        block.add_module("relu{}".format(i),
                         nn.ReLU()
                         )

    block.add_module("pool", nn.MaxPool2d(2, stride=2))

    return block

conv_arch = ((2, 3, 64), (2, 64, 128), (3, 128, 256), (3, 256, 512), (3, 512, 512))


def vgg(conv_arch, NUM_CLASSES):
    net = nn.Sequential()

    for i, (num_convs, input_ch, num_channels) in enumerate(conv_arch):
        net.add_module("block{}".format(i), vgg_block(num_convs, input_ch, num_channels))

    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(25088, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, NUM_CLASSES))

    net.add_module('classifier', classifier)
    return net

class VGG16(nn.Module):

    def __init__(self):
        self._current_object = None

    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        self.net = vgg(conv_arch, num_classes)

    def forward(self, x):
        return self.net(x)