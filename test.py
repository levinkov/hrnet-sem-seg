from hrnet import get_hrnet


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = get_hrnet('big', 19)

    print(f'Number of trainable parameters: {count_parameters(model)}')
