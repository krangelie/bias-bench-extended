import argparse
import json
import os

import transformers

from bias_bench.benchmark.stereoset import StereoSetRunner
from bias_bench.model import models
from bias_bench.util import generate_experiment_id, _is_generative

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Runs StereoSet benchmark.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data will be stored.",
)
parser.add_argument(
    "--model",
    action="store",
    type=str,
    default="BertForMaskedLM",
    choices=[
        "BertForMaskedLM",
        "AlbertForMaskedLM",
        "RobertaForMaskedLM",
        "GPT2LMHeadModel",
        "LukeForMaskedLM",
        "ColakeForMaskedLM"
    ],
    help="Model to evalute (e.g., BertForMaskedLM). Typically, these correspond to a HuggingFace "
    "class.",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="bert-base-uncased",
    choices=["bert-base-uncased", "albert-base-v2", "roberta-base", "gpt2", "gpt2-medium", "gpt2-m-kelm", "luke-base",
             "colake"],
    help="HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a "
    "model is instantiated.",
)
parser.add_argument(
    "--batch_size",
    action="store",
    type=int,
    default=1,
    help="The batch size to use during StereoSet intrasentence evaluation.",
)
parser.add_argument(
    "--seed",
    action="store",
    type=int,
    default=None,
    help="RNG seed. Used for logging in experiment ID.",
)


if __name__ == "__main__":
    args = parser.parse_args()

    if "kelm" in args.model_name_or_path:
        model_name_or_path = "/media/angelie/Samsung_T5/KELM-GPT2/gpt2-medium/kelm_full"
    elif "colake" in args.model_name_or_path:
        model_name_or_path = "bias_bench/model/colake"
    elif "luke" in args.model_name_or_path:
        model_name_or_path = "studio-ousia/luke-base"
    else:
        model_name_or_path = args.model_name_or_path

    experiment_id = generate_experiment_id(
        name="stereoset",
        model=args.model,
        model_name_or_path=args.model_name_or_path,
        seed=args.seed,
    )

    print("Running StereoSet:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - model: {args.model}")
    print(f" - model_name_or_path: {model_name_or_path}")
    print(f" - batch_size: {args.batch_size}")
    print(f" - seed: {args.seed}")


    model = getattr(models, args.model)(model_name_or_path)
    model.eval()

    if "kelm" in args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2-medium")
    elif "colake" in args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

    runner = StereoSetRunner(
        intrasentence_model=model,
        tokenizer=tokenizer,
        input_file=f"{args.persistent_dir}/data/stereoset/test.json",
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        is_generative=_is_generative(args.model),
    )
    results = runner()

    os.makedirs(f"{args.persistent_dir}/results/stereoset", exist_ok=True)
    with open(
        f"{args.persistent_dir}/results/stereoset/{experiment_id}.json", "w"
    ) as f:
        json.dump(results, f, indent=2)
