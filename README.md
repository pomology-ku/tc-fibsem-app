<!--
Copyright (c) MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# tc-FIBSEM App

A lightweight extension of the [MONAI Bundle App](https://github.com/Project-MONAI/MONAILabel/tree/main/sample-apps/monaibundle) for 2.5‑D segmentation of persimmon tannin cells in FIB‑SEM stacks.

## Key Features

- **2.5‑D training pipeline** built on `segmentation_models_pytorch` with optional ImageNet‑pretrained encoders.
- **Interactive mode** through MONAI Label + 3D Slicer for annotation and inference.
- **Headless CLI** for repeatable training / inference on any server.

## Citation

> **Developmental dynamics of cellular specialization during proanthocyanidin accumulation in persimmon fruit**  
> *Yosuke Fujiwara, Soichiro Nishiyama, Akane Kusumi, Keiko Okamoto-Furuta, Hisayo Yamane1, Keizo Yonemori and Ryutaro Tao*  
> *in preparation*

## Pre-requisites

```bash
# Python = 3.9
pip install -U monailabel
pip install segmentation-models-pytorch
mkdir app
cd app
git clone https://github.com/pomology-ku/tc-fibsem-app.git
```

## Dataset Preparation & Layout

### Quick Prep

If you intend to **train** a model, open each raw *multitiff* stack in **3D Slicer**, paint a few representative slices with the segmentation tools, and save the mask as a *multitiff* (`.tiff`).  This preserves voxel alignment and works seamlessly with the CLI.

### Folder Structure

```
dat/
 ├── sample_A.tiff        # raw multitiff stack(s)
 ├── sample_B.tiff
 └── labels/
     └── final/
         ├── sample_A.tiff  # label pairs created via “Submit Label”
         └── sample_B.tiff
```

Multiple image/label pairs are supported.

## CLI Usage
### Training

```bash
python tc-fibsem-app/scripts/run_fibsem_seg.py \
  -a tc-fibsem-app/ \
  -s ./dat \
  --encoder efficientnet-b4 \
  train \
  --max-epochs 500
```

- `--encoder` downloads the chosen backbone and saves checkpoints under `models/<encoder>/`.

### Inference

```bash
python tc-fibsem-app/scripts/run_fibsem_seg.py \
  -a tc-fibsem-app/ \
  -s ./dat \
  --encoder efficientnet-b4 \
  infer \
  --image test.regist.tiff
```

- Automatically loads the latest checkpoint for the selected encoder.
- Outputs a `.nrrd` mask alongside the input.

## 3D Slicer Workflow

> **Status:** Slicer‑based training & inference are experimental and not yet fully validated.

### Prerequisites

```bash
pip install monailabel
```

Install **3D Slicer** (≥ 5.6) and enable the **MONAI Label** extension.

### Typical Steps

1. Load a *multitiff* stack via the **Volumes** module.
2. Use **Segment Editor** to paint a few slices or load an existing mask.
3. Start the MONAI Label server:
   ```bash
   monailabel start_server --app apps/tc-fibsem-app/ --studies dir_to_your_study/
   ```
4. In Slicer, click the green **connect** icon in the MONAI Label panel to link to the server.
5. Set **Source volume** and **Label** correctly, then **Submit Label**.
   - The mask is saved to `dir_to_your_study/labels/final/` as a multitiff with label values.
6. Choose **Train** or **Infer** from the same panel as needed.

> If UI actions fail (the code has evolved since early tests), please fall back to the CLI workflow described above.

## Pretrained Models

Trained checkpoint for this project (from efficientnet-b4) can be downloaded from Google Drive and unpack it so the folder structure becomes:

```
tc-fibsem-app/
  models/
    tc-fibsem-seg-efficientnet-b4/
      model.pt
```

### Quick Download & Unzip
```bash
# Requires gdown →  pip install gdown
gdown https://drive.google.com/uc?id=1yj3FG3aoDF1SfGMpoPYDOrUS0QB-Xnr2 -O model.zip
unzip model.zip -d tc-fibsem-app/
```

