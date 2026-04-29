import os, argparse
from .prompt_templates import (
    extract_translation
)
from .constants import CORPORA_CONFIG
from .utils import read_config, write_json
import re


def find_usernames_hashtags_urls(text):
    # Regular expression patterns
    url_pattern = r'https?://[^\s]+'              # Matches URLs starting with http or https
    username_pattern = r'@\w+'                    # Matches usernames starting with '@' and followed by word characters
    hashtag_pattern = r'#\w+'                     # Matches hashtags starting with '#' and followed by word characters

    return {
        "urls": re.findall(url_pattern, text), 
        "usernames": re.findall(username_pattern, text), 
        "hashtags": re.findall(hashtag_pattern, text)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Path to the output directory for a given model.")
    parser.add_argument("-d", "--data-dir", type=str, help="Path to the data directory.")
    parser.add_argument("-c", "--config-file", type=str, default=CORPORA_CONFIG)
    args = parser.parse_args()

    config = read_config(args.config_file, args.data_dir)

    # loop through all  subdirectories in the output directory (corresponding to different corpora)
    for corpus in os.listdir(args.output_dir):
        corpus_dir = os.path.join(args.output_dir, corpus)
        if not os.path.isdir(corpus_dir):
            continue
        print(f"- Processing {corpus}...")
        # get the source and target languages for the corpus
        corpus_config = config[corpus]
        source_lang = corpus_config["src_lang"]
        target_lang = corpus_config["tgt_lang"]

        # loop through all the files in the corpus directory ending in ".out"
        for file_name in os.listdir(corpus_dir):
            if not file_name.endswith(".out"):
                continue
            guidelines = file_name.split(".")[-2]
            input_file = os.path.join(corpus_dir, file_name)
            output_file = os.path.join(corpus_dir, f"{file_name}.postproc")

            diff = False
            # read the input file line by line and extract the using the extract_translation function and write to the output file
            with open(input_file, "r") as fin, open(output_file, "w") as fout:
                for llm_output in fin:
                    translation = extract_translation(llm_output, source_lang, target_lang, guidelines)
                    fout.write(translation + "\n")
                    # check if the translation is different from the original output
                    diff = diff or (llm_output.strip() != translation)
            
            # print whether the output is different from the original output
            if diff:
                print(f"  - postprocessed {guidelines} file is different from the original output.")
            else:
                print(f"  - postprocessed {guidelines} file is the same as the original output.")
            
            stats_file = f"{output_file}.stats.json"
            stats = {
                "lines": 0,
                "urls": 0,
                "usernames": 0,
                "hashtags": 0
            }
            with open(output_file, "r") as f:
                for line in f:
                    stats["lines"] += 1
                    found = find_usernames_hashtags_urls(line)
                    stats["urls"] += len(found["urls"])
                    stats["usernames"] += len(found["usernames"])
                    stats["hashtags"] += len(found["hashtags"])
            
            stats["urls_per_line"] = stats["urls"] / stats["lines"]
            stats["usernames_per_line"] = stats["usernames"] / stats["lines"]
            stats["hashtags_per_line"] = stats["hashtags"] / stats["lines"]
            
            # write the stats to a json file
            write_json(stats_file, stats)
            print(f"  - stats written to {stats_file}")
