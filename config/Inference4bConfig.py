import os
from typing import List

class Inference4bConfig:
    """Config holder for LLaMA 4bit inference
    """
    def __init__(self,
                 llama_q4_config_dir: str, llama_q4_model: str,
                 lora_apply_dir: str,
                 chat_mode: bool,
                 rope_max_position_embeddings: int,
                 rope_theta: float,
                 rope_scaling: dict,
                 rope_ntk_a: float,
                 offloading: bool,
                 offload_folder: str,
                 config_file_path: str,
                 device_map : str,
                 max_memory : dict,
                 search_max_tokens: bool,
                 prompt_template_file: str,
                 prompt_input_file: str,
                 system_message: str,
                 prompt_fmt: str,
                 repetition_penalty: float,
                 max_new_tokens: int,
                 temperature: float,
                 do_sample_off: bool,
                 top_p: float,
                 top_k: int,
                 num_beams: int,
                 early_stopping: bool,
                 no_streamer: bool
                 ):
        """
        Args:
            llama_q4_config_dir (str): Path to the config.json, tokenizer_config.json, etc
            llama_q4_model (str): Path to the quantized model in huggingface format
            lora_apply_dir (str): Path to directory from which LoRA has to be applied before training
            chat_mode (bool): Start inference in chat mode
            rope_max_position_embeddings (int) : The maximum sequence length that this model might ever be used with
            rope_theta (float): The base period of the RoPE embeddings
            rope_scaling (dict): Dictionary containing the scaling configuration for the RoPE embeddings
            rope_ntk_a (float): RoPE NTK-scaled parameter *a* - https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/?rdt=47731
            https://colab.research.google.com/drive/1VI2nhlyKvd5cw4-zHvAIk00cAVj2lCCC#scrollTo=d2ceb547
            offloading (bool): Use offloading
            offload_folder (str): Offloading disk folder
            config_file_path (str): path to config file used
            search_max_tokens (bool): start in search for maximmum tokens mode
            prompt_template_file (str): Prompt template text
            prompt_input_file (str): Prompt text
            system_message (str): Prompt system message text
            prompt_fmt (str): Prompt format to use
            repetition_penalty (float): Repetition penalty
            max_new_tokens (int): Maximum new tokens to generate
            temperature (float): Temperature
            do_sample_off (bool): Sampling turn-off
            top_p (float): Top P
            top_k (int): Top K
            num_beams (int): search beams
            early_stopping (bool): set stopping conditions for beam-based methods
            no_streamer (bool): turn off text streamer
        """
        self.llama_q4_config_dir = llama_q4_config_dir
        self.llama_q4_model = llama_q4_model
        self.lora_apply_dir = lora_apply_dir
        self.chat_mode = chat_mode
        self.rope_max_position_embeddings = rope_max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rope_ntk_a = rope_ntk_a
        self.offloading = offloading
        self.offload_folder = offload_folder
        self.config_file_path = config_file_path
        self.device_map = device_map if device_map is not None else "auto"
        self.max_memory = None
        if max_memory is not None:
            delattr(self, "max_memory")
            setattr(self, "max_memory", dict())
            for k, v in max_memory.items():
                try:
                    self.max_memory[int(k)] = v
                except:
                    self.max_memory[k] = v
        self.search_max_tokens = search_max_tokens
        self.prompt_template_file = prompt_template_file
        self.prompt_input_file = prompt_input_file
        self.system_message = system_message
        self.prompt_fmt = prompt_fmt
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample_off
        self.top_p = top_p
        self.top_k = top_k
        self.num_beams = num_beams
        self.early_stopping = early_stopping
        self.no_streamer = no_streamer

        self.rope_validation()


    def rope_validation(self):
        # Disable other RoPE types if NTK-aware RoPE is requested
        if self.rope_ntk_a is not None:
            self.rope_ntk_a = 1 if self.rope_ntk_a < 1 else self.rope_ntk_a
            self.rope_max_position_embeddings = 4096 if self.rope_max_position_embeddings is None else self.rope_max_position_embeddings
            self.rope_theta = 10000 if (self.rope_theta is None or self.rope_theta < 10000) else self.rope_theta
            self.rope_scaling = None
            self.__ntk_scaled_rope_patch(self.rope_max_position_embeddings, self.rope_theta,
                                self.rope_ntk_a)
            
        if self.rope_scaling is not None:
            self.rope_scaling = 2 if self.rope_scaling < 1 else self.rope_scaling
            if isinstance(self.rope_scaling, (int, float)):
                self.rope_scaling = {"type": "dynamic", "factor": self.rope_scaling}

        if self.rope_max_position_embeddings is not None:
            self.rope_max_position_embeddings = 2048 if self.rope_max_position_embeddings < 2048 else self.rope_max_position_embeddings


    # Apply NTK-aware RoPE scaling patch
    def __ntk_scaled_rope_patch(self, max_position_embeddings:int = 16384, base:int = 10000, ntk_a: int = 8):
        import transformers
        old_init = transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__

        # Patch
        def __ntk_scaled_init(self, dim, max_position_embeddings = max_position_embeddings, base = base, device=None):
            base = base * ntk_a ** (dim / (dim - 2))
            
            old_init(self, dim, max_position_embeddings, base, device)
            
        transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__=__ntk_scaled_init


    def __str__(self) -> str:
        s = f"\nParameters:\n{'Inference':-^20}\n" +\
        f"{self.llama_q4_config_dir=}\n{self.llama_q4_model=}\n{self.lora_apply_dir=}\n" +\
        f"{self.chat_mode=}\n" +\
        f"{self.rope_max_position_embeddings=}\n" +\
        f"{self.rope_theta=}\n" +\
        f"{self.rope_scaling=}\n" +\
        f"{self.rope_ntk_a=}\n" +\
        f"{self.offloading=}\n" +\
        f"{self.offload_folder=}\n" +\
        f"{self.config_file_path=}\n" +\
        f"{self.repetition_penalty=}\n" +\
        f"{self.max_new_tokens=}\n" +\
        f"{self.temperature=}\n" +\
        f"{self.do_sample=}\n" +\
        f"{self.top_p=}\n" +\
        f"{self.top_k=}\n" +\
        f"{self.num_beams=}\n" +\
        f"{self.early_stopping=}\n" +\
        f"{self.no_streamer=}\n" +\
        f"{self.prompt_input_file=}\n" +\
        f"{self.prompt_template_file=}\n" +\
        f"{self.prompt_fmt=}\n" +\
        f"{self.system_message=}\n"
        return s.replace("self.", "")
