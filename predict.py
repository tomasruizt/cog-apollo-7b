# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
from transformers import AutoModelForCausalLM
from utils.mm_utils import (
    KeywordsStoppingCriteria,
    tokenizer_mm_token,
    ApolloMMLoader
)
from utils.conversation import conv_templates, SeparatorStyle
from utils.openai_conversation import Message

MODEL_CACHE = "checkpoints"
# Apollo-LMMs/Apollo-3B-t32 weights
# MODEL_URL = "https://weights.replicate.delivery/default/Apollo-LMMs/Apollo-3B-t32/3b.tar"
MODEL_URL = "https://weights.replicate.delivery/default/Apollo-LMMs/Apollo-7B-t32/7b.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # Initialize device and model
        start = time.time()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        attn_implementation="sdpa" if torch.__version__ > "2.1.2" else "eager"

        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_CACHE,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
        ).to(device=self.device, dtype=torch.bfloat16)
        
        # Set up tokenizer and other components
        self.tokenizer = self.model.tokenizer
        self.vision_processors = self.model.vision_tower.vision_processor
        self.config = self.model.config
        
        # Configure multimedia processor
        num_repeat_token = self.config.mm_connector_cfg['num_output_tokens']
        self.mm_processor = ApolloMMLoader(
            self.vision_processors,
            self.config.clip_duration,
            frames_per_clip=4,
            clip_sampling_ratio=0.65,
            model_max_length=self.config.model_max_length,
            device=self.device,
            num_repeat_token=num_repeat_token
        )
        print("Setup took:", time.time() - start)

    @torch.inference_mode()
    def predict(
        self,
        video: Path = Input(description="Input video file"),
        messages: str = Input(
            description="Question or prompt about the video",
            default='[{"role": "user", "content": "Describe this video in detail"}]',
        ),
        temperature: float = Input(
            description="Sampling temperature",
            default=0.4,
            ge=0.1,
            le=2.0,
        ),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate",
            default=256,
            ge=32,
            le=1024,
        ),
        top_p: float = Input(
            description="Top-p sampling probability",
            default=0.7,
            ge=0.0,
            le=1.0,
        ),
    ) -> str:
        """Run a single prediction on the model"""
        # Process video
        mm_data, replace_string = self.mm_processor.load_video(str(video))
        
        # Prepare conversation template
        conv = conv_templates["qwen_2"].copy()
        user, assistant = conv.roles
        messages: list[Message] = Message.parse_messages(messages)
        found_fst_user_msg = False
        for msg in messages:
            role = user if msg.role == "user" else assistant
            content = msg.content
            if role == user and not found_fst_user_msg:
                content = replace_string + "\n\n" + content
                found_fst_user_msg = True
            conv.append_message(role, content)
        conv.append_message(assistant, None)  # Add generation prompt
        
        # Tokenize input
        prompt = conv.get_prompt()
        print(f"Full prompt: '{prompt}'")

        input_ids = tokenizer_mm_token(
            prompt,
            self.tokenizer,
            return_tensors="pt"
        ).unsqueeze(0).to(self.device)
        
        # Set up stopping criteria
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria(
            [stop_str],
            self.tokenizer,
            input_ids
        )
        
        output_ids = self.model.generate(
            input_ids,
            vision_input=[mm_data],
            data_types=['video'],
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            use_cache=True,
            num_beams=1,
            stopping_criteria=[stopping_criteria]
        )
        
        # Decode and return prediction
        prediction = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0].strip()

        return prediction
