import os, sys
import torch
from PIL import Image
import open_clip
import pickle
from torch.nn import functional as F
import copy
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from huggingface_hub import snapshot_download

class Q16(torch.nn.Module):
    def __init__(self, checkpoint_dir="./checkpoints/Q16/prompts.p", device="cuda"):
        super(Q16, self).__init__()
        
        self.device = device
        model_name, pretrained = "ViT-L-14", "openai"
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained, quick_gelu=True)
        self.clip_model.to(torch.float32).to(self.device)
        
        # download checkpoints if not exists
        if not os.path.exists(checkpoint_dir):
            os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)
            print(f"Downloading Q16 checkpoints from Hugging Face Hub to {checkpoint_dir}...")
            snapshot_download(repo_id="yiting/Q16",
                    repo_type="model",
                    local_dir=os.path.dirname(checkpoint_dir))
        
        embeddings = torch.FloatTensor(pickle.load(open(checkpoint_dir, 'rb')))
        self.prompts = torch.nn.Parameter(embeddings).to(self.device)
    
    def preprocess_images(self, image_paths):
        images = [self.preprocess(Image.open(img_path)) for img_path in image_paths]
        images = torch.stack(images)
        return images
    
    @ torch.no_grad()
    def classify(self, x):
        x = x.to(self.device)
        text_feat = self.prompts / self.prompts.norm(dim=-1, keepdim=True)
        image_feat = self.clip_model.encode_image(x)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_feat @ text_feat.T)
        return similarity.squeeze()
    
    def forward(self, x):
        x = x.to(self.device)
        text_feat = self.prompts / self.prompts.norm(dim=-1, keepdim=True)
        image_feat = self.clip_model.encode_image(x)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_feat @ text_feat.T)
        
        logits = F.softmax(similarity, dim=-1) # add this to adapt adversarial attack
        return logits
    
    @ torch.no_grad()
    def forward_head_only(self, e): # input is image features: [bs, embed_len]
        e /= e.norm(dim=-1, keepdim=True)
        e = e.to(self.device)
        text_feat = self.prompts / self.prompts.norm(dim=-1, keepdim=True)
        similarity = (100.0 * e @ text_feat.T)
        logits = F.softmax(similarity, dim=-1)
        return logits
        

class MultiHeadedClassifier(torch.nn.Module):
    def __init__(self, checkpoint_dir="./checkpoints/MultiHeaded", device="cuda"):
        super(MultiHeadedClassifier, self).__init__()
        self.device = device
        model_name, pretrained = "ViT-L-14", "openai"
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained, quick_gelu=True)
        self.clip_model.to(torch.float32).to(self.device)
        
        self.category_heads = torch.nn.ModuleList()
        
        # download checkpoints if not exists
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Downloading MultiHeaded checkpoints from Hugging Face Hub to {checkpoint_dir}...")
            snapshot_download(repo_id="yiting/MultiHeaded",
                    repo_type="model",
                    local_dir=checkpoint_dir)
            
        unsafe_contents = ["sexual", "violent", "disturbing", "hateful", "political"]
        for head in unsafe_contents:
            checkpoint_path = os.path.join(checkpoint_dir, head+".pt")
            mlp_head = self._init_mlp_head()
            new_mlp_head = self._load_pretrained_weights(mlp_head, checkpoint_path)
            self.category_heads.append(new_mlp_head)
    
        self.category_heads.to(self.device)

    def _init_mlp_head(self):
        return torch.nn.Sequential(
            torch.nn.Linear(768, 384),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.BatchNorm1d(384),
            torch.nn.Linear(384, 1)
            )
        
    def _load_pretrained_weights(self, mlp_head, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        new_mlp_head = copy.deepcopy(mlp_head)
        new_mlp_head.load_state_dict(checkpoint)
        return new_mlp_head
                
    def preprocess_images(self, image_paths):
        images = [self.preprocess(Image.open(img_path)) for img_path in image_paths]
        images = torch.stack(images)
        return images
    
    @ torch.no_grad()
    def classify(self, x):
        return self.forward(x)
    
    # to adapt the adversarial attack
    def forward(self, x):
        x = x.to(self.device)
        x = self.clip_model.encode_image(x)
        # nsfw score
        out = torch.stack([torch.nn.Sigmoid()(head(x)) for head in self.category_heads], dim=0).permute(1, 0, 2) # [b_s, 5, 1]
        
        # this is for adversarial attack
        out = torch.max(out, dim=1)[0] # [b_s, 1]
        sfw_score = torch.ones_like(out) - out
        logits = torch.cat([sfw_score, out], dim=-1)
        logits = F.softmax(logits, dim=-1) # [b_s, 2]
        
        return logits
    
    @ torch.no_grad()
    def forward_head_only(self, e):
        e = e.to(self.device)
        # nsfw score
        out = torch.stack([torch.nn.Sigmoid()(head(e)) for head in self.category_heads], dim=0).permute(1, 0, 2) # [b_s, 5, 1]
        max_values, max_indices = torch.max(out, dim=1) # [b_s, 1]
        
        embed_list = []
        for idx, max_idx in enumerate(max_indices):
            head = self.category_heads[max_idx]
            embed_block = torch.nn.Sequential(*list(head.children())[:1])
            embed_list.append(embed_block(e[idx].unsqueeze(0)))
            
        embed = torch.cat(embed_list, dim=0)
        sfw_score = torch.ones_like(max_values) - max_values
        logits = torch.cat([sfw_score, max_values], dim=-1)
        logits = F.softmax(logits, dim=-1) # [b_s, 2]
        
        return embed, logits
    
    
from transformers import CLIPConfig, CLIPVisionModel, PreTrainedModel
from torch.nn import functional as F


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = torch.nn.functional.normalize(image_embeds)
    normalized_text_embeds = torch.nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())


class StableDiffusionSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPConfig):
        super().__init__(config)
        safety_model_id = "CompVis/stable-diffusion-safety-checker"
        self.safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
        # print("Image encoder of SD Filter:", config.vision_config)
        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = torch.nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

        self.concept_embeds = torch.nn.Parameter(torch.ones(17, config.projection_dim), requires_grad=False)
        self.special_care_embeds = torch.nn.Parameter(torch.ones(3, config.projection_dim), requires_grad=False)

        self.concept_embeds_weights = torch.nn.Parameter(torch.ones(17), requires_grad=False)
        self.special_care_embeds_weights = torch.nn.Parameter(torch.ones(3), requires_grad=False)

    def preprocess_images(self, image_paths):
        images = [Image.open(img_path) for img_path in image_paths]
        safety_checker_input = self.safety_feature_extractor(images, return_tensors="pt") # pixel_values: [1, 3, 224, 224]
        images = safety_checker_input.pixel_values
        return images
    
    @torch.no_grad()
    def classify(self, clip_input):
        clip_input = clip_input.to(self.vision_model.device)
        
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds).cpu().float().numpy()
        cos_dist = cosine_distance(image_embeds, self.concept_embeds).cpu().float().numpy()

        result = []
        batch_size = image_embeds.shape[0]
        for i in range(batch_size):
            result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": []}

            # increase this value to create a stronger `nfsw` filter
            # at the cost of increasing the possibility of filtering benign images
            adjustment = 0.0

            for concept_idx in range(len(special_cos_dist[0])):
                concept_cos = special_cos_dist[i][concept_idx]
                concept_threshold = self.special_care_embeds_weights[concept_idx].item()
                result_img["special_scores"][concept_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["special_scores"][concept_idx] > 0:
                    result_img["special_care"].append({concept_idx, result_img["special_scores"][concept_idx]})
                    adjustment = 0.01

            for concept_idx in range(len(cos_dist[0])):
                concept_cos = cos_dist[i][concept_idx]
                concept_threshold = self.concept_embeds_weights[concept_idx].item()
                result_img["concept_scores"][concept_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["concept_scores"][concept_idx] > 0:
                    result_img["bad_concepts"].append(concept_idx)

            result.append(result_img)

        has_nsfw_concepts = [len(res["bad_concepts"]) > 0 for res in result]
        
        # use logits
        one_hot_list = [[0, 1] if x else [1, 0] for x in has_nsfw_concepts]
        logits = torch.tensor(one_hot_list)
        return logits
    
    # prepare for adversarial attack
    def forward(self, clip_input):
        # Forward pass through the vision model
        clip_input = clip_input.to(self.vision_model.device)
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        # Calculate cosine similarities (not distances as the names suggest)
        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds)
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)

        # Adjustment parameters
        adjustment = 0.0

        # Special care scores calculation
        special_scores = special_cos_dist - self.special_care_embeds_weights + adjustment
        special_care = torch.any(special_scores > 0, dim=1)
        special_adjustment = special_care * 0.01
        special_adjustment = special_adjustment.unsqueeze(1).expand(-1, cos_dist.shape[1])

        # Concept scores calculation
        concept_scores = (cos_dist - self.concept_embeds_weights) + special_adjustment

        # Now, derive safe and unsafe logits
        # Unsafe score is the maximum score among all concepts (indicating the strongest match to an "unsafe" concept)
        unsafe_scores = torch.max(concept_scores, dim=1).values
        
        # Safe score can be the complement of the unsafe score
        safe_scores = -unsafe_scores
        
        logits = torch.stack([safe_scores, unsafe_scores], dim=1)
        logits = F.softmax(logits, dim=-1)
        return logits

def load_torch_model(checkpoint_dir):
    
    class Normalization(torch.nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.register_buffer('mean', torch.zeros(shape))
            self.register_buffer('variance', torch.ones(shape))

        def forward(self, x):
            return (x - self.mean) / self.variance.sqrt()

    class NSFWModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = Normalization([768])
            self.linear_1 = torch.nn.Linear(768, 64)
            self.linear_2 = torch.nn.Linear(64, 512)
            self.linear_3 = torch.nn.Linear(512, 256)
            self.linear_4 = torch.nn.Linear(256, 1)
            self.act = torch.nn.ReLU()
            self.act_out = torch.nn.Sigmoid()

        def forward(self, x):
            x = self.norm(x)
            x = self.act(self.linear_1(x))
            x = self.act(self.linear_2(x))
            x = self.act(self.linear_3(x))
            x = self.act_out(self.linear_4(x))
            return x
    
    model = NSFWModel()
    model.load_state_dict(torch.load(checkpoint_dir))
    return model

class NSFWDetector(torch.nn.Module):
    def __init__(self, checkpoint_dir="./checkpoints/NSFWDetector/clip_autokeras_binary_nsfw.pth", device="cuda"):
        super(NSFWDetector, self).__init__()
        
        self.device = device
        model_name, pretrained = "ViT-L-14", "openai"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained, quick_gelu=True)
        # self.safety_head = NSFW_detector.load_safety_model() # this is tensorflow model
        
        # download checkpoints if not exists
        if not os.path.exists(checkpoint_dir):
            os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)
            print(f"Downloading NSFWDetector checkpoints from Hugging Face Hub to {checkpoint_dir}...")
            snapshot_download(repo_id="yiting/NSFWDetector",
                    repo_type="model",
                    local_dir=os.path.dirname(checkpoint_dir))
        
        self.safety_head = load_torch_model(checkpoint_dir) # convert the model to torch
        
        self.model = self.model.to(self.device)
        self.safety_head = self.safety_head.to(self.device)
        
    def preprocess_images(self, image_paths):
        images = [self.preprocess(Image.open(img_path)) for img_path in image_paths]
        images = torch.stack(images)
        return images
    
    @torch.no_grad()
    def classify(self, image_tensor):
        return self.forward(image_tensor)
        
    def forward(self, image_tensor):
        image_tensor = image_tensor.to(self.device)
        embeddings = self.model.encode_image(image_tensor)
        embeddings = embeddings / torch.norm(embeddings, keepdim=True, dim=-1)
        nsfw_values = self.safety_head(embeddings)
        sfw_values = torch.ones_like(nsfw_values) - nsfw_values
        logits = torch.cat([sfw_values, nsfw_values], dim=1)
        logits = F.softmax(logits, dim=-1)
        return logits
    
    @torch.no_grad()
    def forward_head_only(self, e):
        e = e.to(self.device)
        e = e/e.norm(keepdim=True, dim=-1)
        x = self.safety_head.norm(e)
        x_embed = self.safety_head.act(self.safety_head.linear_1(x))
        x_embed = self.safety_head.linear_2(x_embed)
        # x_embed = x
        
        nsfw_values = self.safety_head(e)
        sfw_values = torch.ones_like(nsfw_values) - nsfw_values
        logits = torch.cat([sfw_values, nsfw_values], dim=1)
        logits = F.softmax(logits, dim=-1)
        return x_embed, logits

import os
import keras
import keras.utils as ku
import pydload
import numpy as np
import onnx
from onnx2pytorch import ConvertModel
import tf2onnx
import tensorflow as tf
import torch

def load_images(image_paths, image_size):
    '''
    Function for loading images into numpy arrays for passing to model.predict
    inputs:
        image_paths: list of image paths to load
        image_size: size into which images should be resized
    
    outputs:
        loaded_images: loaded images on which keras model can run predictions
        loaded_image_indexes: paths of images which the function is able to process
    
    '''
    loaded_images = []
    loaded_image_paths = []

    for i, img_path in enumerate(image_paths):
        try:
            # image = keras.preprocessing.image.load_img(img_path, target_size = image_size)
            # image = keras.preprocessing.image.img_to_array(image)
            image = ku.load_img(img_path, target_size = image_size)
            image = ku.img_to_array(image)
            image /= 255
            loaded_images.append(image)
            loaded_image_paths.append(img_path)
        except Exception as ex:
            print(i, img_path, ex)
    image_tensor = np.asarray(loaded_images)
    return image_tensor

class NudeNet():
    '''
        Class for loading model and running predictions.
        For example on how to use take a look the if __name__ == '__main__' part.
    '''
    nsfw_model = None

    def __init__(self, model_path = "./checkpoints/NudeNet/classifier_model"):
        '''
            model = Classifier()
        '''
        
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print(f"Downloading NudeNet checkpoints from Hugging Face Hub to {model_path}...")
            snapshot_download(repo_id="yiting/NudeNet",
                    repo_type="model",
                    local_dir=os.path.dirname(model_path))

        NudeNet.nsfw_model = keras.models.load_model(model_path)
        self.training = False
        # self.eval = False

    def preprocess_images(self, image_paths, image_size = (256, 256)):
        
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        loaded_images = []
        for i, img_path in enumerate(image_paths):
            try:
                # image = keras.preprocessing.image.load_img(img_path, target_size = image_size)
                # image = keras.preprocessing.image.img_to_array(image)
                image = ku.load_img(img_path, target_size = image_size)
                image = ku.img_to_array(image)
                image /= 255
                loaded_images.append(image)
            except Exception as ex:
                print(i, img_path, ex)
        image_tensor = np.asarray(loaded_images)
        image_tensor = tf.convert_to_tensor(image_tensor, dtype=tf.float32)
        # image_tensor = torch.from_numpy(image_tensor)
        return image_tensor
    
    
    def classify(self, image_tensor, batch_size = 32, image_size = (256, 256), categories = ['unsafe', 'safe']):
        '''
            inputs:
                image_paths: list of image paths or can be a string too (for single image)
                batch_size: batch_size for running predictions
                image_size: size to which the image needs to be resized
                categories: since the model predicts numbers, categories is the list of actual names of categories
        '''
        batch_size = image_tensor.shape[0]
        image_tensor = tf.convert_to_tensor(image_tensor, dtype=tf.float32)
        model_preds = NudeNet.nsfw_model.predict(image_tensor, batch_size = batch_size) # output of foolbox attack
        model_preds = torch.tensor(model_preds)
        
        # it originally use unsafe as 0, now we change it to  1
        model_preds_swapped = model_preds.clone()
        model_preds_swapped[:, [0, 1]] = model_preds[:, [1, 0]]
        return model_preds_swapped
    
    
    def train(self):
        pass
    
    def eval(self):
        pass
    
    def __call__(self, *args):
        return self.classify(*args)

def load_conventional_classifier(classifier_name, device):
    def _freeze_torch_module(m):
        for p in m.parameters():
            p.requires_grad = False

    if classifier_name == "Q16":
        classifier = Q16(device=device)
    elif classifier_name == "MultiHeaded":
        classifier = MultiHeadedClassifier(device=device)
    elif classifier_name == "SD_Filter":
        safety_model_id = "CompVis/stable-diffusion-safety-checker"
        classifier = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)
    elif classifier_name == "NSFW_Detector":
        classifier = NSFWDetector(device=device)
    elif classifier_name == "NudeNet":
        classifier = NudeNet()
        # Keras model: mark not trainable
        try:
            NudeNet.nsfw_model.trainable = False
        except Exception:
            pass
        return classifier

    # Move to device, set eval mode and freeze grads for torch modules
    if isinstance(classifier, torch.nn.Module):
        try:
            classifier = classifier.to(device)
        except Exception:
            pass
        classifier.eval()
        _freeze_torch_module(classifier)

    return classifier