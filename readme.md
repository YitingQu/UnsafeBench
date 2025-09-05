# UnsafeBench: A Comprehensive Benchmark for Vision-Language Model Safety

UnsafeBench is a comprehensive evaluation framework for assessing the safety and robustness of Vision-Language Models (VLMs) and image safety classifiers against unsafe content.

## 🚀 Features

- **Multi-modal Safety Assessment**: Evaluate both image safety classifiers and vision-language models
- **Comprehensive Dataset Support**: Built-in support for multiple safety datasets including SMID, NSFWDataset, MultiHeaded_Dataset, Violence_Dataset, and Self-harm_Dataset
- **Adversarial Robustness Testing**: Tools for evaluating model robustness against adversarial attacks
- **Standardized Evaluation**: Consistent evaluation protocols across different model types
- **Extensible Architecture**: Easy to add new models and datasets

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU
- Git
- Conda

### Quick Setup

1. Setup the environment:

```bash
bash setup.sh
conda activate llava
````

2. For InstructBLIP models (optional):

```bash
bash setup_lavis_env.sh
conda activate lavis
```

3. Setup Tokens:

```bash
export HF_TOKEN=hf_xxx   # required
export OPENAI_API_KEY=sk-proj-xxx   # optional
```

## 🎯 UnsafeBench Evaluation

**Attention: due to ethical considerations, the UnsafeBench dataset is provided for research or education purposes only. To use it, please first request access from the Hugging Face: [link](https://huggingface.co/datasets/yiting/UnsafeBench)**

### Effectiveness Evaluation

1. Generate VLM responses using the UnsafeBench dataset:

```bash
python query_vlms.py --model_name llava-v1.5-7b --save_dir outputs/vlm_responses
```

```bash
conda activate lavis
python query_vlms.py --model_name instructblip-7b --save_dir outputs/vlm_responses
```

GPT-4V (`gpt-4-vision-preview`) has been deprecated, so it cannot be queried at the moment.
Nonetheless, we provide the generated responses in the [link](https://huggingface.co/datasets/yiting/unsafebench_vlm_responses).

For reproducible evaluation, this step can be skipped by directly using our responses, which will be automatically downloaded in step 2.

2. Obtain Evaluation Results:

```bash
python assess_effectiveness.py --classifiers "Q16" "SD_Filter" "llava-v1.5-7b" \
    --vlm_response_dir ./outputs/vlm_responses \
    --save_dir ./outputs/effectiveness/UnsafeBench
```

This reproduces the effectiveness result in **Table 3**.

### Robustness Evaluation

1. Conduct adversarial attacks against **conventional classifiers**:

```bash
python assess_robustness.py --classifiers "Q16" "MultiHeaded" "SD_Filter" "NSFW_Detector" "NudeNet" \
    --attack_types "GN" "FGSM" "PGD" "DeepFool" \
    --eps 0.01 \
    --prediction_path ./outputs/effectiveness/UnsafeBench \
    --save_dir ./outputs/robustness
```

Running this will yield the Robust Accuracy of **conventional classifiers**, as shown in **Table 4**.

2. Conduct adversarial attacks against VLM classifiers:

For **LLaVA**:

```bash
python llava_adv_attack.py --model-base liuhaotian/llava-v1.5-7b \
    --attack_types "GN" "FGSM" "PGD" "DeepFool" \
    --eps 0.01 \
    --prediction_path ./outputs/effectiveness/UnsafeBench \
    --save_dir ./outputs/robustness
```

For **InstructBLIP**:

```bash
python lavis_adv_attack.py \
    --attack_types "GN" "FGSM" "PGD" "DeepFool" \
    --eps 0.01 \
    --prediction_path ./outputs/effectiveness/UnsafeBench \
    --save_dir ./outputs/robustness
```

Running these will yield the Robust Accuracy of **LLaVA** and **InstructBLIP**, as shown in **Table 4**.

## 🎯 PerspectiveVision

1. For direct inference:

```bash
cd PerspectiveVision
python eval.py --dataset_names UnsafeBench_test SMID NSFWDataset MultiHeaded_Dataset Violence_Dataset Self-harm_Dataset
```
This reproduces the effectiveness result of PerspectiveVision in **Table 5**.

2. For Training:

```bash
python build_training_set.py
```

Train LLaVA using LoRA:

```bash
bash train_llava.sh
```

## 📊 Supported Models

### Image Safety Classifiers

* **Q16**: CLIP-based multi-category safety classifier
* **NudeNet**: Nudity detection model
* **NSFW\_Detector**: General NSFW content detector
* **MultiHeaded**: Multi-head classification model
* **SD\_Filter**: Stable Diffusion safety filter

### Vision-Language Models

* **LLaVA-v1.5**: Large Language and Vision Assistant
* **InstructBLIP**: Instruction-tuned BLIP model
* **GPT-4V**: OpenAI GPT models

**Usage Example:**

```python
from classifiers import Q16
import torch

classifier = Q16()

image_paths = ["data/test.png"]
images = classifier.preprocess_images(image_paths)
logits = classifier.classify(images)
pred = torch.argmax(logits, dim=-1)
print(f"Prediction: {pred} (0: safe, 1: unsafe)")
```


## 🗂️ Dataset Support

The framework supports multiple safety datasets:

* **UnsafeBench**: Comprehensive multi-category unsafe content dataset
* **SMID**: Safety in Multimodal Intelligence Dataset
* **NSFWDataset**: Not Safe For Work content dataset
* **MultiHeaded\_Dataset**: Multi-head classification dataset
* **Violence\_Dataset**: Violence detection dataset
* **Self-harm\_Dataset**: Self-harm content detection dataset

**Usage Example:**

UnsafeBench:
```python
from datasets import load_dataset
dataset = load_dataset("yiting/UnsafeBench")["train"]
print(dataset[0])

{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1600x1435 at 0x7FB291227D90>,
'safety_label': 'Safe',
'category': 'Hate',
'source': 'Laion5B',
'text': "xxx"}

```

Other External Datasets:
```python
from datasets import load_dataset
import io
import base64
from PIL import Image

dataset_name = "SMID"
    
dataset = load_dataset(f"yiting/{dataset_name}", split="train")

encoded_image = dataset[0]["image"]

def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image

image = decode_base64_to_image(encoded_image)
label = dataset[0]["label"]
print(image, label) # 0 safe, 1 unsafe
```
[Attention!] Do not cite our work when you use these external datasets, cite their original works pls.

## 🛠️ Project Structure

```
├── README.md                # Project documentation
├── pyproject.toml            # Project configuration
├── unsafe_datasets.py        # Dataset loading and management
├── classifiers.py            # Image safety classifiers
├── vlms.py                   # Vision-language models
├── assess_effectiveness.py   # Model effectiveness evaluation
├── assess_robustness.py      # Robustness evaluation
├── query_vlms.py             # VLM querying utilities
├── utils.py                  # Helper functions
├── data/                     # Dataset storage
├── outputs/                  # Evaluation results
├── PerspectiveVision/        # PerspectiveVision
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions, issues, or collaborations, please:

* Open an issue on GitHub
* Reach out via email: \[[yiting.qu@cispa.de](mailto:yiting.qu@cispa.de)]

## 📚 Citation

If you use UnsafeBench in your research, please cite:

```bibtex
@inproceedings{QSWBZZ24,
  author    = {Yiting Qu and Xinyue Shen and Yixin Wu and Michael Backes and Savvas Zannettou and Yang Zhang},
  title     = {{UnsafeBench: Benchmarking Image Safety Classifiers on Real-World and AI-Generated Images}},
  booktitle = {{ACM SIGSAC Conference on Computer and Communications Security (CCS)}},
  publisher = {ACM},
  year      = {2025}
}
```
