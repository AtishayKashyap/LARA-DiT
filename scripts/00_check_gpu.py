import torch,sys
print('PyTorch:',torch.__version__); print('CUDA:',torch.version.cuda); print('CUDA available:',torch.cuda.is_available())
if not torch.cuda.is_available(): sys.exit('CUDA is unavailable')
for i in range(torch.cuda.device_count()):
 p=torch.cuda.get_device_properties(i); print(f'GPU {i}: {p.name} | {p.total_memory/2**30:.2f} GiB | CC {p.major}.{p.minor}')
print('BF16:',torch.cuda.is_bf16_supported())
if 'H100' not in torch.cuda.get_device_name(0): print('WARNING: this project is tuned for H100/MIG 3g.40gb')
