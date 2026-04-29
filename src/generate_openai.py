import os, argparse, yaml, time
from openai import OpenAI
from .constants import GPT, DEFAULT
from .prompt_templates import get_prompt

from .utils import read_file

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input-file", type=str)
	parser.add_argument("-l", "--target-lang", type=str)
	parser.add_argument("-m", "--model-name", type=str, default=GPT)
	parser.add_argument("-g", "--guidelines", type=str, default=DEFAULT)
	parser.add_argument("-k", "--api-key", type=str, default=None)
	parser.add_argument("--normalize", action="store_true")
	args = parser.parse_args()
	
	api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
	if not api_key:
		raise ValueError("API key must be provided via --api-key argument or OPENAI_API_KEY environment variable")
	
	client = OpenAI(api_key=api_key)

	sentences = read_file(args.input_file)
	
	file_name = os.path.basename(args.input_file)
	output_file = os.path.join(os.path.dirname(args.input_file), f"gpt.{file_name}")
	
	print(f" - Loaded {len(sentences)} sentences...")
	start_time = time.time()
	n = 0
	with open(output_file, "w") as f:
		for sentence in sentences:
			completion = client.chat.completions.create(
				model=GPT,
				messages=get_prompt(sentence, args.target_lang, args.target_lang, normalization=args.normalize, model_name=args.model_name, guidelines=args.guidelines),
				temperature=0, 
				top_p=1,
				max_tokens=512,
			)
			output = completion.choices[0].message.content.strip().replace("\n", " ")
			f.write(f"{output}\n")
			n += 1
			if n % 10 == 0:
				print(f" - {n} done...")

	print(f" - Normalized sentences saved to {output_file}")
	elapsed = time.time() - start_time
	hours, remainder = divmod(int(elapsed), 3600)
	minutes, seconds = divmod(remainder, 60)
	print(f" - Normalization took {hours}h {minutes}m {seconds}s")
