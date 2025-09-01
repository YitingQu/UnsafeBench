import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

def prepare_text_prompt(user_prompt):
    
    if DEFAULT_IMAGE_TOKEN not in user_prompt:

        qs = DEFAULT_IMAGE_TOKEN + '\n' + user_prompt

    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    return prompt

# support batch implementation
class Prompt:
    # tokenization
    # turn to embeddings

    # padding? wait until targets have been appended
    # prepare labels? need to wait for targets

    def __init__(self, model, tokenizer, text_prompts=None, device='cuda:0'):

        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.text_prompts = text_prompts
        self.context_length = []
        self.input_ids = []
        self.do_tokenization(self.text_prompts)


    def do_tokenization(self, text_prompts):

        if text_prompts is None:
            self.input_ids = []
            self.context_length = []
            return

        input_ids = tokenizer_image_token(text_prompts, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        self.input_ids = [input_ids]
        self.context_length = [input_ids.shape[1]]