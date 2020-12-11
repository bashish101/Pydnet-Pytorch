import torch

def time_torch_model(model, input, use_cuda=True, print_time=False):
    """
    Times the forward pass of a nn.Module with the given input.
    Inputs:
        model: nn-Module
        input: torch.Tensor
        use_cuda: bool       To use CUDA GPU or not (default: True)
        print_time: bool     Print the time?        (default: False)
    Output:
        total_time_ms
    License: Zeeshan Khan Suri
    """
    model.eval()
    torch.cuda.synchronize()
    with torch.no_grad():
        with torch.autograd.profiler.profile(use_cuda=use_cuda) as prof:
            model(input)
    total_time_ms = sum([item.cuda_time for item in prof.function_events])/1000
    if print_time:
        print("{:.3f} ms".format(total_time_ms))
    return total_time_ms