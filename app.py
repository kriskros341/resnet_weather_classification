# app.py
import os
import io
from pathlib import Path
from typing import Optional, List, Tuple

import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import pandas as pd


from model import ResNet
import sys


st.write("Python:", sys.executable)
st.write("Torch version:", torch.__version__)



try:
    from test_transforms import DataLoaderCompatibleTransform, ManualTransform
except Exception:
    DataLoaderCompatibleTransform = None
    ManualTransform = None

try:
    from transforms import test_transform, train_transform
except Exception:
    test_transform = None
    train_transform = None


st.set_page_config(page_title="Weather recognition UI", layout="wide")


#######################
# Utilities
#######################

def get_device(force: str = "auto"):
    """force: 'auto'|'cpu'|'cuda'"""
    if force == "cpu":
        return torch.device("cpu")
    if force == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_dataset(dataset_path: str, transform=None) -> Optional[ImageFolder]:
    if not os.path.isdir(dataset_path):
        return None
    try:
        ds = ImageFolder(root=dataset_path, transform=transform)
        return ds
    except Exception as e:
        st.warning(f"Nie udało się wczytać dataset: {e}")
        return None


def infer_single_image(model: torch.nn.Module, device: torch.device, img_tensor: torch.Tensor, classes: List[str]):
    model.eval()
    with torch.no_grad():
        x = img_tensor.unsqueeze(0).to(device)
        out = model(x)
        probs = F.softmax(out, dim=1).cpu().squeeze(0).numpy()
        topk_idx = probs.argsort()[::-1][:5]
        results = [(classes[i], float(probs[i])) for i in topk_idx]
    return results


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())



def instantiate_model(num_classes: int, in_channels: int = 3):
    return ResNet(in_channels, num_classes)



def try_load_state_dict(model: torch.nn.Module, path: str, map_location="cpu"):
    sd = torch.load(path, map_location=map_location)
    new_sd = {}
    for k, v in sd.items():
        new_sd[k.replace("_orig_mod.", "")] = v
    model.load_state_dict(new_sd)
    return model


#######################
# Layout
#######################
st.title("Weather recognition — Streamlit UI")
st.markdown("Interfejs do wczytywania modelu, testowania i eksportu ONNX. ")


st.sidebar.header("Ustawienia")
device_choice = st.sidebar.radio("Urządzenie", options=["auto", "cuda", "cpu"], index=0)
device = get_device(device_choice)

st.sidebar.markdown("**Ścieżki**")
dataset_path = st.sidebar.text_input("Dataset folder", value="dataset")
default_model_path = "best.pth"
model_path = st.sidebar.text_input("Plik modelu (.pth)", value=default_model_path)

st.sidebar.markdown("---")
st.sidebar.markdown("**ONNX / export**")
onnx_name = st.sidebar.text_input("Nazwa pliku ONNX", value="model_app.onnx")

#######################
# Top info / device check
#######################
st.subheader("Informacje o urządzeniu")
col1, col2 = st.columns([1, 2])
with col1:
    if st.button("Sprawdź CUDA"):
        st.write("torch.cuda.is_available():", torch.cuda.is_available())
        if torch.cuda.is_available():
            st.write("Aktualne GPU:", torch.cuda.get_device_name(0))
        else:
            st.write("Brak GPU lub CUDA niedostępne.")
with col2:
    st.write(f"Wybrane urządzenie: **{device}**")


#######################
# Dataset and classes
#######################
st.subheader("Dataset i klasy")

dataset = load_dataset(dataset_path, transform=None)
if dataset is None:
    st.warning(f"Brak katalogu dataset w: {dataset_path}. Niektóre funkcje będą niedostępne.")
    classes = []
else:
    classes = dataset.classes
    st.write(f"Znaleziono {len(dataset)} obrazów w {len(classes)} klasach.")
    st.write("Klasy:", classes)

#######################
# Model load & info
#######################
st.subheader("Model")

num_classes_guess = len(classes) if len(classes) > 0 else st.sidebar.number_input("Liczba klas (jeśli brak dataset)", min_value=1, value=11)
if st.button("Wczytaj model"):
    try:
        net = instantiate_model(num_classes_guess, in_channels=3)
        net = try_load_state_dict(net, model_path, map_location="cpu")
        net.to(device)
        st.success("Model wczytany poprawnie.")
        st.write("Parametry modelu:", count_params(net))
        st.write("Konfiguracja pierwszej warstwy (conv1.weight):")
        try:
            st.write(dict(net.conv1.weight.shape._asdict()) if hasattr(net.conv1.weight.shape, "_asdict") else tuple(net.conv1.weight.shape))
        except Exception:
            st.write(str(net.conv1.weight.shape))
        # cache the loaded model in session_state for later use
        st.session_state['net'] = net
    except Exception as e:
        st.error(f"Nie udało się wczytać modelu: {e}\nUpewnij się, że plik .pth jest kompatybilny z ResNet(3, num_classes).")


if 'net' in st.session_state:
    net = st.session_state['net']
    st.write(net)  # print structure (Streamlit will render)
else:
    st.info("Wczytaj model aby zobaczyć szczegóły.")


#######################
# Single image inference
#######################
st.subheader("Inference — pojedynczy obraz")

uploaded_file = st.file_uploader("Prześlij obraz (jpg/png)", type=["jpg", "jpeg", "png"])
use_manual_transform = st.checkbox("Użyć DataLoaderCompatibleTransform (jeśli dostępne)", value=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Przesłany obraz", width=256)


    if DataLoaderCompatibleTransform is not None and use_manual_transform:
        transformer = DataLoaderCompatibleTransform()
        img_tensor = transformer(image)  
    elif test_transform is not None:
        img_tensor = test_transform(image)
    else:
        img_tensor = torchvision.transforms.ToTensor()(image)
        img_tensor = torchvision.transforms.functional.normalize(img_tensor, mean=(0.5,)*3, std=(0.5,)*3)

    if 'net' not in st.session_state:
        st.warning("Wczytaj najpierw model z sekcji 'Model'.")
    else:
        with st.spinner("Robię predykcję..."):
            net = st.session_state['net']
            results = infer_single_image(net, device, img_tensor, classes if classes else [str(i) for i in range(num_classes_guess)])
        st.markdown("**Top predictions:**")
        for cls, prob in results[:5]:
            st.write(f"{cls} — {prob*100:.2f} %")


#######################
# Batch evaluation (quick)
#######################
st.subheader("Szybka ocena na części dataset (batch)")

if dataset is None:
    st.info("Aby użyć tej opcji wymagana jest struktura dataset (ImageFolder).")
else:
    n_samples = st.number_input("Ile próbek do oceny (losowo, maksymalnie)", min_value=1, max_value=len(dataset), value=min(200, len(dataset)))
    run_eval = st.button("Uruchom szybką ocenę (eval)")

    if run_eval:
        if 'net' not in st.session_state:
            st.error("Wczytaj model najpierw.")
        else:
            if DataLoaderCompatibleTransform is not None:
                ds_transform = DataLoaderCompatibleTransform()
                dataset.transform = ds_transform
            elif test_transform is not None:
                dataset.transform = test_transform
            else:
                dataset.transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(280),
                    torchvision.transforms.CenterCrop((256,256)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                ])

            import random
            idxs = random.sample(range(len(dataset)), n_samples)
            subset = Subset(dataset, idxs)
            use_cuda = torch.cuda.is_available()
            loader = DataLoader(subset, batch_size=16, shuffle=False, num_workers=2, pin_memory=use_cuda)

            net = st.session_state['net']
            net.eval()
            correct = 0
            total = 0
            per_class_correct = {c: 0 for c in classes}
            per_class_total = {c: 0 for c in classes}

            with st.spinner("Evaluacja..."):
                with torch.no_grad():
                    for xb, yb in loader:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        logits = net(xb)
                        preds = torch.argmax(logits, dim=1)
                        total += yb.size(0)
                        correct += (preds == yb).sum().item()
                        for gt, p in zip(yb.cpu().numpy(), preds.cpu().numpy()):
                            per_class_total[classes[gt]] += 1
                            if gt == p:
                                per_class_correct[classes[gt]] += 1

            overall_acc = 100.0 * correct / total if total > 0 else 0.0
            st.success(f"Skończono. Accuracy (na {total} próbkach): {overall_acc:.2f}%")


            rows = []
            for c in classes:
                t = per_class_total[c]
                if t > 0:
                    acc = 100.0 * per_class_correct[c] / t
                else:
                    acc = None
                rows.append({"class": c, "total": t, "correct": per_class_correct[c], "acc_%": acc})
            df = pd.DataFrame(rows)
            st.dataframe(df.sort_values("total", ascending=False))


#######################
# Export to ONNX
#######################
st.subheader("Export do ONNX")

if st.button("Eksportuj model do ONNX (z transformacją, jeśli dostępna)"):
    if 'net' not in st.session_state:
        st.error("Wczytaj model najpierw.")
    else:
        net = st.session_state['net']
        net.eval()
        if ManualTransform is not None:
            wrapper = torch.nn.Sequential(ManualTransform(), net)
            dummy = torch.randn(1, 3, 256, 256)
            dynamic_axes = {'input': {0: 'batch_size', 2: 'height', 3: 'width'}, 'output': {0: 'batch_size'}}
        else:
            wrapper = net
            dummy = torch.randn(1, 3, 256, 256)
            dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

        try:
            torch.onnx.export(
                wrapper,
                dummy,
                onnx_name,
                export_params=True,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                opset_version=13
            )
            st.success(f"Wyeksportowano ONNX -> {onnx_name}")
            with open(onnx_name, "rb") as f:
                btn = st.download_button(
                    label="Pobierz ONNX",
                    data=f.read(),
                    file_name=onnx_name,
                    mime="application/octet-stream"
                )
        except Exception as e:
            st.error(f"Błąd eksportu ONNX: {e}")


#######################
# Show some random images from dataset
#######################
st.subheader("Przykładowe obrazy z dataset")
if dataset is not None:
    sample_n = st.slider("Ile obrazków pokazać", min_value=1, max_value=16, value=8)
    import random
    idxs = random.sample(range(len(dataset)), sample_n)
    cols = st.columns(min(sample_n, 4))
    for i, idx in enumerate(idxs):
        img, lbl = dataset[idx]
        path, target = dataset.samples[idx]
        pil = Image.open(path).convert("RGB")
        cols[i % len(cols)].image(pil, caption=f"{dataset.classes[target]}")

st.markdown("---")
st.write("Uwagi: jeśli masz problemy z pamięcią GPU lub speedem, ustaw `device` na CPU w lewym panelu.")
