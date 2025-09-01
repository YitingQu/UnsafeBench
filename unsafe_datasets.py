import os
import sys
import json
import torch
import torch.utils
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn import model_selection
from pathlib import Path
from datasets import load_dataset
import base64
import io

def list_unsafe_instruction():
    """
    Load and parse unsafe instruction categories from the combined instruction file.
    
    Returns:
        tuple: A tuple containing:
            - categories (list): List of unsafe content categories
            - instructions (list): List of full instruction lines
    """
    data_path = "data/combined_instruction.txt"
    abs_path = Path(__file__).parent / data_path
    
    instructions = open(abs_path, "r").read().splitlines()
    categories = []
    for line in instructions:
        categories.append(line.split(":")[0])
    return categories, instructions 

CATEGORIES = list_unsafe_instruction()[0]
SOURCES = ["Laion5B", "Lexica"]

def align_unsafe_categories(classifier_name):
    """
    Align unsafe content categories based on the classifier's capabilities.
    
    Different classifiers support different sets of unsafe content categories.
    This function returns the appropriate category list for each classifier.
    
    Args:
        classifier_name (str): Name of the safety classifier
        
    Returns:
        list: List of unsafe content categories supported by the classifier
    """
    if classifier_name in ["NudeNet", "SD_Filter"]:
        unsafe_categories = ["Sexual"]
    elif classifier_name == "NSFW_Detector":
        unsafe_categories = ["Harassment", "Sexual"]
    elif classifier_name == "MultiHeaded":
        unsafe_categories = ["Sexual", "Violence", "Shocking", "Hate", "Political"]
    elif classifier_name == "Q16":
        unsafe_categories = list_unsafe_instruction()[0]
        unsafe_categories.remove("Spam")
        unsafe_categories.remove("Sexual")
    else:  # vlms
        unsafe_categories, _ = list_unsafe_instruction()
    return unsafe_categories

def encode_image_to_base64(image_path):
    """
    Encode an image file to base64 string.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded image string
    """
    ext = os.path.splitext(image_path)[-1].lower().lstrip(".")  # e.g., "png"
    format = ext.upper() if ext != "jpg" else "JPEG"  # PIL expects "JPEG" not "JPG"
    
    with Image.open(image_path) as img:
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return encoded_string

def decode_base64_to_image(base64_string, target_size=-1):
    """
    Decode base64 string to PIL Image.
    
    Args:
        base64_string (str): Base64 encoded image string
        target_size (int): Target size for resizing (default: -1, no resizing)
        
    Returns:
        PIL.Image: Decoded image
    """
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image

def decode_base64_to_image_file(base64_string, image_path, target_size=-1):
    """
    Decode base64 string and save as image file.
    
    Args:
        base64_string (str): Base64 encoded image string
        image_path (str): Output path for the image file
        target_size (int): Target size for resizing (default: -1, no resizing)
    """
    image = decode_base64_to_image(base64_string, target_size=target_size)
    image.save(image_path)
    
class UnsafeBenchDataset(Dataset):
    """
    Dataset class for loading UnsafeBench data.
    
    This dataset automatically downloads data from HuggingFace if not available locally
    and provides a PyTorch Dataset interface for safety evaluation.
    
    Args:
        image_root (str): Root directory for storing images
        source (str): Data source, either "Lexica" or "Laion5B"
        category (str): Safety category (e.g., "Hate", "Violence", "Sexual")
        partition (str): Data partition, either "train" or "test"
    """
    
    def __init__(self, 
                 image_root="data/UnsafeBench", 
                 source="Lexica", 
                 category="Hate", 
                 partition="train"):
        
        self.label_mapping = {"Safe": 0, "Unsafe": 1}
        self.image_root = image_root
        self.source = source
        self.category = category
        self.partition = partition

        metadata = []

        # Check if images are already downloaded
        images_dir = os.path.join(image_root, partition, "images")
        if os.path.exists(images_dir) and len(os.listdir(images_dir)) > 0:
            pass
        else:
            os.makedirs(images_dir, exist_ok=True)
            print(f"Downloading UnsafeBench {partition} images...")
            self._download_and_save(save_path=images_dir)

        # Load metadata
        metadata_path = os.path.join(images_dir, "metadata.jsonl")
        with open(metadata_path, "r") as f:
            for line in f:
                metadata.append(json.loads(line))

        # Filter metadata based on source and category
        self.metadata = [item for item in metadata 
                        if item["source"] == source and item["category"] == category]
        
    def __getitem__(self, idx):
        image_fname = self.metadata[idx]["image_fname"]
        image_fname = os.path.join(self.image_root, self.partition, "images", image_fname)
        label = self.metadata[idx]["label"]
        label = self.label_mapping[label]
        return image_fname, label
    
    def __len__(self):
        return len(self.metadata)
    
    def _download_and_save(self, save_path):
        from datasets import load_dataset
        import tqdm

        dataset = load_dataset("yiting/UnsafeBench", split=self.partition)

        metadata = []
        for idx, item in enumerate(tqdm.tqdm(dataset)):
            image = item["image"]
            # Convert to RGB if not already
            if image.mode != "RGB":
                image = image.convert("RGB")
            image_id = item.get("id", str(idx))
            image_filename = f"{image_id}.png"
            image.save(os.path.join(save_path, image_filename))
            
            metadata.append({
                "image_fname": image_filename,
                "label": item["safety_label"],
                "source": item["source"],
                "category": item["category"]
            })
        with open(os.path.join(save_path, "metadata.jsonl"), "w") as f:
            for item in metadata:
                f.write(json.dumps(item) + "\n")

# Custom dataset classes for SMID, NSFWDataset, MultiHeaded_Dataset, Violence_Dataset, Self-harm_Dataset
class CustomDataset(Dataset):
    """
    Custom dataset class for loading various safety datasets from HuggingFace.
    
    Supports multiple dataset types including SMID, NSFWDataset, MultiHeaded_Dataset,
    Violence_Dataset, and Self-harm_Dataset. Automatically downloads and caches
    datasets locally for faster subsequent access.
    
    Args:
        dataset_name (str): Name of the dataset to load
        save_path (str): Local path for saving/loading dataset files
    """
    
    def __init__(self, dataset_name="SMID", save_path="data"):
        
        image_root = os.path.join(save_path, dataset_name, "images")
        metadata_dir = os.path.join(save_path, dataset_name, f"{dataset_name}.tsv")
            
        # Download metadata if not exists
        if not os.path.exists(metadata_dir):
            hf_dataset = load_dataset(f"yiting/{dataset_name}", split="train")
            data_df = hf_dataset.to_pandas()
            os.makedirs(os.path.dirname(metadata_dir), exist_ok=True)
            data_df.to_csv(metadata_dir, sep="\t", index=False)
                
        data_df = pd.read_csv(metadata_dir, sep="\t")

        # Download and decode images from base64
        if os.path.exists(image_root) and len(os.listdir(image_root)) > 0:
            pass
        else:
            os.makedirs(image_root, exist_ok=True)
            print(f"Decoding {dataset_name} images from base64...")
            for i, row in data_df.iterrows():
                image_base64 = row["image"]
                # Handle case where image is an index reference
                if image_base64.isdigit():
                    image_base64 = data_df.iloc[int(image_base64)]["image"]
                image_fname = os.path.join(image_root, f"{i}.jpg")
                decode_base64_to_image_file(image_base64, image_fname, target_size=-1)
                
        self.image_root = image_root
        self.data = data_df
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row_idx = self.data.iloc[idx]["index"]
        image_fname = os.path.join(self.image_root, f"{row_idx}.jpg")
        label = self.data.iloc[idx]["label"]
        
        return {
            "image_fname": image_fname,
            "label": label
        }
        
def fetch_evaluation_dataset(dataset_name):
    """
    Fetch and load an evaluation dataset by name.
    
    Args:
        dataset_name (str): Name of the dataset to load. Supported datasets:
            - "SMID": Safety in Multimodal Intelligence Dataset
            - "NSFWDataset": Not Safe For Work content dataset
            - "MultiHeaded_Dataset": Multi-head classification dataset
            - "Violence_Dataset": Violence detection dataset
            - "Self-harm_Dataset": Self-harm content detection dataset
            - "UnsafeBench_test" or "UnsafeBench_TEST": UnsafeBench test set
    
    Returns:
        Dataset: PyTorch Dataset object for the specified dataset
    
    Raises:
        ValueError: If the dataset name is not recognized
    """
    base_dir = os.path.dirname(__file__)

    if dataset_name in ["SMID", "NSFWDataset", "MultiHeaded_Dataset", "Violence_Dataset", "Self-harm_Dataset"]:
        dataset = CustomDataset(dataset_name=dataset_name, save_path=os.path.join(base_dir, "data"))
        print(f"Loaded {len(dataset)} items from {dataset_name}")
        return dataset
    
    elif dataset_name == "UnsafeBench_test" or dataset_name == "UnsafeBench_TEST":
            
        image_root = os.path.join(base_dir, "data", "UnsafeBench")
        concat_datasets = []
        for source in SOURCES:
            for category in CATEGORIES:
                dataset = UnsafeBenchDataset(image_root=image_root, source=source, category=category, partition="test")
                concat_datasets.append(dataset)
        dataset = ConcatDataset(concat_datasets)
        print(f"Loaded {len(dataset)} items from UnsafeBench_test")
        return dataset
    
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
def fetch_merged_UnsafeBench_dataset(source, category):
    """
    Fetch and merge train and test splits of UnsafeBench dataset.
    
    Args:
        source (str): Data source, either "Lexica" or "Laion5B"
        category (str): Safety category (e.g., "Hate", "Violence", "Sexual")
    
    Returns:
        ConcatDataset: Combined train and test dataset
    """
    train_set = UnsafeBenchDataset(source=source, category=category, partition="train")
    test_set = UnsafeBenchDataset(source=source, category=category, partition="test")
    dataset = ConcatDataset([train_set, test_set])
    return dataset

def random_draw_testing_adv_samples(classifier_name, prediction_path, source, K=500, seed=2023):
    """
    Randomly draw K samples from UnsafeBench dataset that were originally 
    successfully predicted by each classifier.
    
    This function is useful for adversarial testing - selecting samples that
    the classifier can correctly classify under normal conditions.
    
    Args:
        classifier_name (str): Name of the classifier
        source (str): Data source ("Lexica", "Laion5B", or "both")
        K (int): Number of samples to draw (default: 500)
        seed (int): Random seed for reproducibility (default: 2023)
    
    Returns:
        Subset: PyTorch Subset containing the randomly selected samples
    
    Raises:
        Exception: If prediction data file is not found
    """
    if source == "both":
        sources = SOURCES
    else:
        sources = [source]
    categories = align_unsafe_categories(classifier_name)
    prediction_data_file = f"{prediction_path}/{classifier_name}.json"
    if os.path.exists(prediction_data_file):
        prediction_data = json.load(open(prediction_data_file, "r"))
    else:
        raise Exception(f"Prediction data file {prediction_data_file} not found. Please run the evaluation script first.")

    concat_dataset = []
    all_images_count = 0
    
    for source in sources:
        for category in categories:
            dataset = fetch_merged_UnsafeBench_dataset(source=source, category=category)
            labels = [int(dataset.__getitem__(i)[1]) for i in range(len(dataset))]
            
            if classifier_name in ["llava-v1.5-7b", "instructblip-7b"]:
                predictions = prediction_data[source][category]["detailed_predictions"][0] # predictions of the first prompt
            else:
                predictions = prediction_data[source][category]["predictions"]
        
            assert len(labels) == len(predictions)
            all_images_count += len(labels)
            labels, predictions = np.array(labels), np.array(predictions)
            attack_indices = np.where(labels == predictions)[0]
            dataset = torch.utils.data.Subset(dataset, attack_indices)
            concat_dataset.append(dataset)
            
    concat_dataset = ConcatDataset(concat_dataset)
    
    torch.manual_seed(seed)
    random.seed(seed)
    
    sample_num = min(len(concat_dataset), K)
    random_indices = random.sample(range(len(concat_dataset)), sample_num)
    final_dataset = torch.utils.data.Subset(concat_dataset, random_indices)
    return final_dataset
