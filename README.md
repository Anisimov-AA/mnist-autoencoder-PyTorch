## Setup

**Requirements:** Python 3.11+, NVIDIA GPU with CUDA drivers

1.  create virtual environment

```
py -m venv venv
```

2.  activate it

```
venv\Scripts\activate
```

3.  install dependencies

```
pip install -r requirements.txt
```

4. verify GPU:

```
python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
```
