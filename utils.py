def count_parameters(net):
    n = 0
    for param in net.parameters():
        n += param.numel()
    return n
