from colorama import init, Style, Fore, Back
import time
from datetime import datetime as dt
import torch
from llmchat import LLMChat

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer

import config
config.WORK_MODE = config.EWorkModes.INFERENCE

from config.arg_parser import get_config

from prompt_builders import AutoPrompt

# ! Suppress warnings from safetensors
import warnings
warnings.filterwarnings(action="ignore", category=UserWarning, message="TypedStorage is deprecated")

# Color output
init(autoreset=True)

# ! Config
config = get_config()

config_path = config.llama_q4_config_dir
model_path = config.llama_q4_model
groupsize = config.groupsize

# * Show loaded parameters
print(f"{config}\n")

print(Style.BRIGHT + Fore.CYAN + f"Loading Model ...\nTimestamp: {dt.now()}")
t0 = time.time()

# Convert max_memory to factual AutoAWQ max_memory
def max_vram_to_autoawq_mib(vram: str) -> str:
    vram_suffixes = ["mib", "gib"]
    vmem_suffix = vram[-3:].lower()
    if vmem_suffix == vram_suffixes[0]:
        mem_mb = int(vram[:-3])
    elif vmem_suffix == vram_suffixes[1]:
        mem_mb = 1024 * int(vram[:-3])
    else:
        raise ValueError()
    
    k_convert = 0.55 / 1.2
    
    return f"{int(k_convert * mem_mb)}Mib"

config.max_memory[0] = max_vram_to_autoawq_mib(config.max_memory[0])

tokenizer = AutoTokenizer.from_pretrained(config_path)
txt_streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


model = AutoAWQForCausalLM.from_quantized(config_path, 
                                          model_path, 
                                          fuse_layers=False,
                                          safetensors=True
                                          )

print(Style.BRIGHT + Fore.GREEN + f"Loaded the model in {(time.time()-t0):.2f} seconds.\nTimestamp: {dt.now()}")

if not config.offloading:
    print(Fore.LIGHTYELLOW_EX + 'Apply AMP Wrapper ...')
    from amp_wrapper import AMPWrapper
    wrapper = AMPWrapper(model)
    wrapper.apply_generate()


def get_model_response(prompt: str, max_new_tokens:int = 4096, use_txt_streamer:bool = True) -> str:
    """Get response from ML-model

    Args:
        prompt (str): Valid model prompt

    Returns:
        str: raw model response
    """
    batch = tokenizer(prompt, 
                      return_tensors="pt", 
                      add_special_tokens=False).input_ids.cuda()
    print(Fore.LIGHTYELLOW_EX + f"Tokens: {batch.flatten().shape}\nTimestamp: {dt.now()}")

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    with torch.inference_mode():
        generated = model.generate(
            inputs=batch,
            do_sample=config.do_sample,
            use_cache=True,
            early_stopping = config.early_stopping,
            repetition_penalty=config.repetition_penalty,
            max_new_tokens=max_new_tokens if max_new_tokens > 0 else 4096,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            num_beams=config.num_beams,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
            streamer=txt_streamer if use_txt_streamer else None
            )
    result_text = tokenizer.decode(generated['sequences'].cpu().tolist()[0])
    return result_text


def inference():
    chat = LLMChat(get_model_response)
    
    system_msg = chat.get_system_message() if config.system_message is None else config.system_message
    prompt_builder = AutoPrompt(mode=config.prompt_fmt, config_dir=config_path, system_msg=system_msg)
    
    is_onetimer = config.prompt_input_file is not None
    
    while True:
        if config.prompt_input_file is None:
            user_text = chat.get_user_input()
        else:
            with open(config.prompt_input_file, "r") as ifile:
                user_text = "".join(ifile.readlines())
        
        if config.prompt_template_file is not None:
            with open(config.prompt_template_file, "r") as ifile:
                prompt_fmt = "".join(ifile.readlines())
                user_text = prompt_fmt.format(USR_PROMPT_TXT=user_text)
        
        chat.chat_history = chat.chat_history if config.chat_mode else None
        prompt = prompt_builder.make(user_text, chat.chat_history)

        print("Processing ⌛", end="")
        start = time.time()
        chat.chat_history = get_model_response(prompt, config.max_new_tokens, config.num_beams == 1)
        print("\r" + Fore.LIGHTMAGENTA_EX + "AI:>" + prompt_builder.refine_output(chat.chat_history))
        end = time.time()
        print(Fore.LIGHTGREEN_EX + f"Inference time: {end - start:.2f}s\nTimestamp: {dt.now()}")
        
        if is_onetimer:
            break


def search_max_tokens(min_tokens:int = 2048, max_tokens:int = 4096 * 8, eps: float=0.4, try_found = True):
    # ! DEBUG: Search max tokens
    # Llama2-70B
    # 3305-3310
    # Llama2-13B
    # 8532
    # Llama2-7B
    # 10574
    prompt_builder = AutoPrompt("")
    
    _eps = eps
    _min_tokens = min_tokens  #- 21
    _max_tokens = max_tokens  #- 21

    _max_old = _max_tokens

    _less_than_tokens = list()

    while True:
        print(f"{_min_tokens=}\n{_max_tokens=}")
        prompt = int(_max_tokens - 1) * "a " + "a"
        prompt = prompt_builder.make(prompt)
        
        try:  # not is_less(max)
            if _max_tokens in _less_than_tokens:
                raise ValueError()
            
            get_model_response(prompt=prompt, max_new_tokens=1, use_txt_streamer=False)
            print(Fore.LIGHTCYAN_EX + f">={_max_tokens}")
            
            _diff = 0.5 * (_max_tokens - _min_tokens)
            _min_tokens = _min_tokens + _diff
            _max_tokens = _max_old
        except:  # is_less(max)
            if _max_tokens not in _less_than_tokens:
                _less_than_tokens.append(_max_tokens)
            
            print(Fore.LIGHTCYAN_EX + f"<{_max_tokens}")
            
            _diff = 0.5 * (_max_tokens - _min_tokens)
            _max_old = _max_tokens
            _max_tokens = _min_tokens + _diff
            
        if abs(_diff) < _eps:
            print(Fore.LIGHTGREEN_EX + f"Max tokens: {_max_tokens + 21}")
            break
        
    if try_found:
        # ! Test 10574
        # prompt = int(10574 - 1 - 21) * "a " + "a"
        print(Fore.LIGHTCYAN_EX + f"Test {_max_tokens}")
        prompt = int(_max_tokens - 1) * "a " + "a"
        prompt = prompt_builder.make(prompt)
        
        print(get_model_response(prompt=prompt, max_new_tokens=1, use_txt_streamer=False))


# Entry point
if __name__ == "__main__":
    if config.search_max_tokens:
        search_max_tokens()
    else:
        inference()
