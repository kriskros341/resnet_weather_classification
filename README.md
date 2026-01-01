Kroki:

dataset:
https://www.kaggle.com/datasets/jehanbhathena/weather-dataset?resource=download

Sprawdzenie sterowników GPU

```
nvidia-smi
```

bierzemy CUDA Version.

W repozytorium użyto managera uv.

```
uv init
```

```
uv venv
```

```
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu{CUDA Version}
```

U mnie konieczny był też restart

```
uv run python check-cuda.py
```


