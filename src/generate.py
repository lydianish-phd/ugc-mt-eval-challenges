import os, argparse, yaml
from vllm import LLM, SamplingParams

from .constants import GREEDY_CONFIG
from .utils import read_file, read_yaml
from .prompt_templates import (
    get_prompt,
    GUIDELINES
)
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", type=str, required=True, help="Path to the input file.")
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("-m", "--model-dir", type=str, required=True, help="Path to the model directory.")
    parser.add_argument("-c", "--config-file", type=str, default=GREEDY_CONFIG)
    parser.add_argument("--source-lang", type=str, default="English")
    parser.add_argument("--target-lang", type=str, default="French")
    parser.add_argument("-g", "--guidelines", type=str, nargs="+", default=["default"])
    parser.add_argument("--overwrite", help="whether to overwrite existing output files", default=False, action="store_true")    
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16"], default="bfloat16", help="Data type for model weights.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92, help="GPU memory utilization for vLLM.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    file_name = os.path.basename(args.input_file)
    model_name = os.path.basename(args.model_dir)

    config = read_yaml(args.config_file)
    print(f" - Loaded configuration from {args.config_file}:")
    for key, value in config.items():
        print(f"   {key}: {value}")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    llm = LLM(
        model=args.model_dir, 
        max_model_len=config["max_model_len"],
        dtype=dtype,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        n=config["n"],
        temperature=config["temperature"], 
        top_p=config["top_p"],
        max_tokens=config["max_tokens"],
    )
        
    sentences = read_file(args.input_file)

    for guideline in args.guidelines:
        if guideline not in GUIDELINES:
            raise ValueError(f"Invalid guideline: {guideline}, expected one of {GUIDELINES.keys()}.")
        
        output_file = os.path.join(args.output_dir, f"{file_name}.{guideline}.out")
        
        if not args.overwrite and os.path.exists(output_file):
            print(f" - Skipping {output_file}")
            continue
        
        print(f" - Generating translations with the {guideline} guidelines...")

        prompts = [ get_prompt(sentence, args.source_lang, args.target_lang, model_name=model_name, guidelines=guideline) for sentence in sentences ]
        outputs = llm.generate(prompts, sampling_params)

        with open(output_file, "w") as f:
            for output in outputs:
                # generated_text = output.outputs[0].text.split('\n')[0].strip()
                generated_text = output.outputs[0].text.strip().replace("\n", " ")
                f.write(f"{generated_text}\n")

        print(f" - Output translations saved to {output_file}")