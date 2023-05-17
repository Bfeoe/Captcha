import torch

if_cuda = torch.cuda.is_available()
print("if_cuda=", if_cuda)
gpu_count = torch.cuda.device_count()
print("gpu_count=", gpu_count)

# import torch
# torch.__version__
# torch.cuda.is_available()
# torch.cuda.device(0)
# torch.cuda.device_count()
# torch.cuda.get_device_name(0)
