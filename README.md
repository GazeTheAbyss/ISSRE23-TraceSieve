# ISSRE23-TraceSieve
VGAE model in PyTorch.

## Usage
### Train
```
python3 -m scripts.train_vgae -D dataset_name -g xxx -t xxx --flag
```

### Test
```
python3 -m tracegnn.models.vgae.test evaluate-nll -D dataset_name -M model_path --device cuda/cpu --flag
```
