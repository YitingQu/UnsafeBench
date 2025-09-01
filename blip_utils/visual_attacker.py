import torch
from tqdm import tqdm
import random
from torchvision.utils import save_image
import matplotlib.pyplot as plt

def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images

class Attacker:

    def __init__(self, args, model, roberta, roberta_tokenizer, device='cuda:0', is_rtp=False, bounds=(-3, 3)):

        self.args = args
        self.model = model
        self.device = device
        self.roberta = roberta.to(device)
        self.roberta_tokenizer = roberta_tokenizer
        self.is_rtp = is_rtp
        self.bounds = bounds

        self.loss_buffer = []

        # freeze and set to eval model:
        self.model.eval()
        self.model.requires_grad_(False)

        self.roberta.eval()
        self.roberta.requires_grad_(False)
    
    @torch.no_grad()
    def classify_response(self, response):
        _out = self.roberta_tokenizer(response, return_tensors="pt")
        input_ids = _out.input_ids.to(self.device)
        attention_mask = _out.attention_mask.to(self.device)
        logits = self.roberta(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=-1)
        return preds
    
    
    def attack_GN(self, text_prompt, img, targets, epsilon=128/255):
        batch_size = len(targets)
        assert batch_size == len(targets) == img.shape[0]
  
        x = img.clone().cuda()
        x.requires_grad = True  # Ensure gradients can be calculated for x

        batch_targets = targets
        # Generate Gaussian noise with zero mean and unit variance
        adv_noise = torch.randn_like(x).cuda()
        
        # Calculate the L2 norm of the noise
        adv_noise_norm = torch.norm(adv_noise.view(adv_noise.size(0), -1), dim=-1, keepdim=True).view(-1, 1, 1, 1)
        adv_noise = adv_noise / adv_noise_norm * epsilon
        adv_noise.data = adv_noise.data.clamp(-epsilon, epsilon)
        
        x_adv = x + adv_noise
        x_adv = x_adv.clamp(self.bounds[0], self.bounds[1])  # Ensure x_adv remains a valid image

        with torch.no_grad():
            response = self.model.generate({"image": x_adv, "prompt": text_prompt},
                                              num_beams=1, top_p=None, temperature=0)
            # print('>>> Sample Outputs', response)
            
        response = response * batch_size
        # use roberta to check if the generated response is the same as the target
        preds = self.classify_response(response).detach().cpu().numpy()
        is_adv = []
        for pred, tar in zip(preds, targets):
            tar = 0 if tar == "safe" else 1
            is_adv.append(pred == tar)
        
        # print('>>>', response, " >>> ", targets, "is_adv: ", is_adv)

        return x_adv.detach().cpu(), is_adv
    
    def attack_FGSM(self, text_prompt, img, targets, epsilon=128/255):
        batch_size = len(targets)
        assert batch_size == len(targets) == img.shape[0]
  
        x = img.clone().cuda()
        x.requires_grad = True  # Ensure gradients can be calculated for x

        batch_targets = targets
        # x_adv = normalize(x)
        x_adv = x
        
        samples = {
                'image': x_adv,
                'text_input': [text_prompt] * batch_size,
                'text_output': batch_targets
            }
        
        target_loss = self.model(samples)['loss']
        target_loss.backward(retain_graph=True)

        x_adv = x - epsilon * x.grad.sign() # subtracting here as we want to descrease the loss of model generate targets 
        x_adv = x_adv.clamp(self.bounds[0], self.bounds[1])  # Ensure x_adv remains a valid image
        
        self.loss_buffer.append(target_loss.item())
        
        with torch.no_grad():
            response = self.model.generate({"image": x_adv, "prompt": text_prompt},
                                              num_beams=1, top_p=None, temperature=0)
            # print('>>> Sample Outputs', response)
            
        response = response * batch_size
        # use roberta to check if the generated response is the same as the target
        preds = self.classify_response(response).detach().cpu().numpy()
        is_adv = []
        for pred, tar in zip(preds, targets):
            tar = 0 if tar == "safe" else 1
            is_adv.append(pred == tar)
        
        # print('>>>', response, " >>> ", targets, "is_adv: ", is_adv)

        return x_adv.detach().cpu(), is_adv
        
    def attack_PGD(self, text_prompt, img, targets, num_iter=2000, alpha=1/255, epsilon = 128/255 ):

        batch_size = len(targets)
        assert batch_size == len(targets) == img.shape[0]
        
        adv_noise = torch.rand_like(img, requires_grad=True).to(self.device) * 2 * epsilon - epsilon
        x = img.clone().to(self.device)
        adv_noise.data = (adv_noise.data + x.data).clamp(self.bounds[0], self.bounds[1]) - x.data

        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()

        for t in range(num_iter + 1):

            batch_targets = random.sample(targets, batch_size)

            x_adv = x + adv_noise
            x_adv = x_adv.repeat(batch_size, 1, 1, 1)

            samples = {
                'image': x_adv,
                'text_input': [text_prompt] * batch_size,
                'text_output': batch_targets
            }

            target_loss = self.model(samples)['loss']
            target_loss.backward(retain_graph=True)

            adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(-epsilon, epsilon)
            adv_noise.data = (adv_noise.data + x.data).clamp(self.bounds[0], self.bounds[1]) - x.data
            adv_noise.grad.zero_()
            self.model.zero_grad()

            self.loss_buffer.append(target_loss.item())

            # print("target_loss: %f" % (
            #     target_loss.item())
            #       )

            if t % 100 == 0:
                    
                # print('######### Output - Iter = %d ##########' % t)
                x_adv = x + adv_noise
                
                with torch.no_grad():
                    response = self.model.generate({"image": x_adv, "prompt": text_prompt},
                                              num_beams=1, top_p=None, temperature=0)
     
                response = response * batch_size
                preds = self.classify_response(response).detach().cpu().numpy()
                is_adv = []
                for pred, tar in zip(preds, targets):
                    tar = 0 if tar == "safe" else 1
                    is_adv.append(pred == tar)
                
                # print('>>>', response, " >>> ", targets, "is_adv: ", is_adv)
                    
                if sum(is_adv) == len(is_adv):
                    # print('adversarial examples found at iteration %d' % t)
                    return x_adv.detach().cpu(), is_adv
                  
        return x_adv.detach().cpu(), is_adv

    def attack_DeepFool(self, text_prompt, img, targets, num_iter=2000, overshoot=0.02, epsilon=128/255):
        batch_size = len(targets)
        assert batch_size == len(targets) == img.shape[0]
        
        # Initialize perturbation as zero
        adv_noise = torch.zeros_like(img, requires_grad=True).cuda()
        x = img.clone().cuda()
        adv_noise.retain_grad()
        
        for t in range(num_iter + 1):
            batch_targets = random.sample(targets, batch_size)

            x_adv = x + adv_noise
            x_adv = x_adv.repeat(batch_size, 1, 1, 1)
            
            samples = {
                'image': x_adv,
                'text_input': [text_prompt] * batch_size,
                'text_output': batch_targets
            }

            target_loss = self.model(samples)['loss']
            target_loss.backward(retain_graph=True)

            grad = adv_noise.grad.detach()
            distance = torch.abs(target_loss) / (torch.norm(grad.view(batch_size, -1), p=1) + 1e-8)
            adv_noise.data = adv_noise.data - grad.sign() * distance.view(-1, 1, 1, 1)

            adv_noise.data = (1 + overshoot) * adv_noise.data # original DeepFool doesn't clamp epsilon here
            adv_noise.data = (adv_noise.data + x.data).clamp(self.bounds[0], self.bounds[1]) - x.data
   
            adv_noise.grad.zero_()
            self.model.zero_grad()

            self.loss_buffer.append(target_loss.item())

            # print("target_loss: %f" % (target_loss.item()))

            if t % 100 == 0:
                # print('######### Output - Iter = %d ##########' % t)
                adv_noise.data = adv_noise.data.clamp(-epsilon, epsilon)
                x_adv = x + adv_noise
                
                # generate response  
                with torch.no_grad():
                    response = self.model.generate({"image": x_adv, "prompt": text_prompt},
                                              num_beams=1, top_p=None, temperature=0)
     
                response = response * batch_size
                preds = self.classify_response(response).detach().cpu().numpy()
                is_adv = []
                for pred, tar in zip(preds, targets):
                    tar = 0 if tar == "safe" else 1
                    is_adv.append(pred == tar)
                
                # print("current response:", response, "target: ", targets, "is_adv: ", is_adv)
                    
                if sum(is_adv) == len(is_adv):
                    # print('adversarial examples found at iteration %d' % t)
                    return x_adv.detach().cpu(), is_adv
                
        return x_adv.detach().cpu(), is_adv