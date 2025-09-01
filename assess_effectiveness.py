import os, sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
import fire
from classifiers import load_conventional_classifier
from unsafe_datasets import *
import tqdm
from utils import load_LLM_output_converter
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

def inference(classifier, dataset, batch_size):
    classifier.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    labels = [int(item[1]) for item in dataset]
    
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            imgs, la = batch
            images = classifier.preprocess_images(imgs)
            logits = classifier.classify(images)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            predictions.extend(preds.tolist())
    return labels, predictions
    
def run_conventional_classifiers(classifier_name: str="Q16",
                                batch_size: int=128, 
                                save_dir: str="./outputs/effectiveness/UnsafeBench"):

    # load classifier
    classifier = load_conventional_classifier(classifier_name, device)
    print(f"{classifier_name} loaded")
    print(f"Evaluating {classifier_name}...")
    
    result = {}
    for source in SOURCES:
        result[source] = {}
        aligned_categories = align_unsafe_categories(classifier_name)
        for category in tqdm.tqdm(aligned_categories):

            dataset = fetch_merged_UnsafeBench_dataset(source=source, category=category)
            
            labels, predictions = inference(classifier, dataset, batch_size)
            torch.cuda.empty_cache()
            result[source][category] = {"labels": labels, "predictions": predictions}
    
    json.dump(result, open(f"{save_dir}/{classifier_name}.json", "w"))

def run_VLM_classifiers(classifier_name: str="llava-v1.5-7b",
    vlm_response_dir: str="outputs/vlm_responses",
    save_dir: str="./outputs/effectiveness/UnsafeBench",
    batch_size: int=128):
    
    '''
    evaluate VLMs independently, assume we have generated the VLM responses
    '''

    def load_vlm_response(model_name, prompt_index, source, category):
    
        path = os.path.join(vlm_response_dir, f"{model_name}/prompt_{prompt_index}", f"{source}_{category}.json")
        if os.path.exists(path) is False:
            from huggingface_hub import snapshot_download
            
            repo_id = f"yiting/unsafebench_vlm_responses"
            snapshot_download(repo_id=repo_id,
                            repo_type="dataset",
                            local_dir=vlm_response_dir)
        
        data = json.load(open(path, "r"))
        return data
    
    def majority_voting(predictions):
        majority_votes = []
        for prediction_set in predictions:
            counter = Counter(prediction_set)
            most_common = counter.most_common(1)[0][0]
            majority_votes.append(most_common)
        return majority_votes
    
    def convert_dict(vlm_response):
        result = {}
        for item in vlm_response:
            if isinstance(item["response"], list):
                result[item["image_fname"]] = item["response"][0]
            else:
                result[item["image_fname"]] = item["response"]
        return result

    def get_batches(lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i:i + batch_size]
        
    categories, _ = list_unsafe_instruction()
    
    # load roberta classifier
    tokenizer, detector = load_LLM_output_converter()
    detector = detector.to(device)
        
    prompt_indices = [0, 1, 2]
    result = {}
    for source in SOURCES:
        result[source] = {}
        
        for category in categories:
            
            dataset = fetch_merged_UnsafeBench_dataset(source=source, category=category)
            image_fnames = [item[0] for item in dataset]
            image2labels = {item[0]: int(item[1]) for item in dataset}
    
            predictions = []
            
            for prompt_index in prompt_indices:
                
                # load vlm response/output data that are previously prepared
                vlm_response = load_vlm_response(classifier_name, prompt_index, source, category) # list of dict; image_fname, output
                vlm_response = convert_dict(vlm_response) # key: image_fname, value: output

                response_texts = [vlm_response[image_fname] for image_fname in image_fnames if image_fname in vlm_response]
                
                preds = []
                for batch in get_batches(response_texts, batch_size=batch_size):
                    output = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = detector(**output)
                    preds += outputs.argmax(dim=-1).detach().cpu().tolist()
            
                predictions.append(preds)
                labels = [image2labels[image_fname] for image_fname in image_fnames if image_fname in vlm_response]
                
            predictions_major_vote = majority_voting(np.array(predictions).T)
            predictions_major_vote = np.array(predictions_major_vote).astype(np.int16)
            result[source][category] = {"labels": labels, "predictions": predictions_major_vote.tolist(), "detailed_predictions": predictions}
            
    json.dump(result, open(f"{save_dir}/{classifier_name}.json", "w"))

def describe_results(classifier_names, data_sources, save_dir):
    
    result_df = pd.DataFrame()
    for classifier_name in classifier_names:
        for category in CATEGORIES:
            aligned_categories = align_unsafe_categories(classifier_name)
            if category not in aligned_categories:
                continue
                
            all_predictions = []
            all_labels = []
            for source in data_sources:

                prediction_json = json.load(open(f"{save_dir}/{classifier_name}.json", "r"))
                all_predictions.extend(prediction_json[source][category]["predictions"])
                all_labels.extend(prediction_json[source][category]["labels"])
            
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            na_indices = np.where(all_predictions == 2)[0]
            all_predictions[na_indices] = random.choices([0, 1], k=len(na_indices)) # if the response shows unsure, randomly assign to 0 or 1
            f1 = f1_score(all_labels, all_predictions)
            result_df.loc[classifier_name, category] = f1
    return result_df

def main(args):
    
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.vlm_response_dir).mkdir(parents=True, exist_ok=True)
    
    for classifier_name in args.classifiers:
        
        if classifier_name in ["Q16", "MultiHeaded", "SD_Filter", "NSFW_Detector", "NudeNet"]:
            run_conventional_classifiers(classifier_name,
                                        batch_size=args.batch_size, 
                                        save_dir=args.save_dir)
                
        elif classifier_name in ["llava-v1.5-7b", "instructblip-7b", "gpt-4v"]:
            run_VLM_classifiers(classifier_name=classifier_name,
                                batch_size=args.batch_size,
                                save_dir=args.save_dir,
                                vlm_response_dir=args.vlm_response_dir)
        else:
            raise NotImplementedError
        
    for sources in [["Laion5B"],["Lexica"], ["Laion5B","Lexica"]]:
        result_df = describe_results(args.classifiers, sources, args.save_dir)
        print("="*10+" Data sources: "+",".join(sources)+"="*10)
        print(result_df)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifiers",  nargs="+", default=["Q16", "MultiHeaded", "SD_Filter", "NSFW_Detector", "NudeNet", "llava-v1.5-7b", "instructblip-7b", "gpt-4v"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--vlm_response_dir", type=str, default="./outputs/vlm_responses")
    parser.add_argument("--save_dir", type=str, default="./outputs/effectiveness/UnsafeBench")
    args = parser.parse_args()
    
    main(args)