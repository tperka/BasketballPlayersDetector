vgg_cfg = {
    'vgg': [32, 32, 'M', 64, 64, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M'],
}

resnet_cfg = {
    "bpd": [32, (32, False), (32, False), (64, True), (64, False), (128, True), (256, True)],
}

# for explanation please see: https://python.plainenglish.io/implementing-efficientnet-in-pytorch-part-1-an-overview-1830935e0c8b
efc_cfg = {
    'efc': {'widths': [(32, 16), (16, 24), (24, 40), (40, 80), (80, 112), (112, 192)],
            'depths': [1, 1, 2, 2, 2, 3],
            'kernel_sizes': [3, 3, 5, 3, 5, 5],
            'strides': [1, 2, 2, 2, 1, 2],
            'ps': [0, 0.029, 0.057, 0.086, 0.114, 0.143],
            'rs': [4, 24, 24, 24, 24, 24],
            'expansion_factors': [1, 6, 6, 6, 6, 6],
            'w_factor': 1,
            'd_factor': 1
            }
}
