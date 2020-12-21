from matplotlib import pyplot as plt
import numpy as np
import torch

def time_torch_model(model, input, use_cuda=True, print_time=False):
    """
    Times the forward pass of a nn.Module with the given input.
    Inputs:
    ---------
        model: nn-Module
        input: torch.Tensor
        use_cuda: bool       To use CUDA GPU or not (default: True)
        print_time: bool     Print the time?        (default: False)
    Output:
    ---------
        total_time_ms
    Author: Zeeshan Khan Suri
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

def plot_input_output(img,output, print_output_shape=True):
    """
    Plots two images side-by-side.
    Inputs:
    -----------
        image1: PIL image or numpy array
        output: torch.Tensor or numpy array
    Author: Zeeshan Khan Suri
    """
    if isinstance(output, torch.Tensor):
        output = output.to('cpu').detach().numpy()
    output = output.squeeze()
    
    if print_output_shape: print(output.shape)

    fig, axes = plt.subplots(1,2, figsize=(20,3))
    axes[0].imshow(img)
    depthmap=axes[1].imshow(output,vmax=np.percentile(output, 99))
    fig.colorbar(depthmap);
    
def scale_disp(disp, min_depth, max_depth):
    """
    Scale disparity such that depth lies in the range [min_depth, max_depth]
    Inputs
    -----------
    disp: 
        Disparity image
    min_depth:
        minimum depth
    max_depth:
        maximum depth
    Outputs
    -----------
    scaled_disp:
        Scaled dispartiy
    depth:
        1/scaled_disparity
    """
    min_disp, max_disp = 1 / max_depth, 1/min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    return scaled_disp, 1/scaled_disp