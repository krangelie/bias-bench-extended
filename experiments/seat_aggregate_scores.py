import argparse
import pandas as pd
import numpy as np
import json


GENDER_TESTS = [
    "sent-weat6",
    "sent-weat6b",
    "sent-weat7",
    "sent-weat7b",
    "sent-weat8",
    "sent-weat8b",
]

RACE_TESTS = [
    "sent-angry_black_woman_stereotype",
    "sent-angry_black_woman_stereotype_b",
    "sent-weat3",
    "sent-weat3b",
    "sent-weat4",
    "sent-weat5",
    "sent-weat5b",
]

RELIGION_TESTS = [
    "sent-religion1",
    "sent-religion1b",
    "sent-religion2",
    "sent-religion2b",
]

parser = argparse.ArgumentParser(
    description="Scores a set of StereoSet prediction files."
)

parser.add_argument(
    "--predictions_file",
    action="store",
    type=str,
    default=None,
    help="Path to the file containing the model predictions.",
)
parser.add_argument(
    "--output_file",
    action="store",
    type=str,
    default=None,
    help="Path to save the results to.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    print("Evaluating SEAT results:")
    print(f" - predictions_file: {args.predictions_file}")
    print(f" - output_file: {args.output_file}")

    results_dict = json.load(open(args.predictions_file))
    results_df = pd.DataFrame(results_dict)

    # Average absolute SEAT effect size
    avg_eff_size_dict = {}
    for test_list, category in zip([GENDER_TESTS, RELIGION_TESTS, RACE_TESTS], ["gender", "religion",
                                                                                "race"]):
        category_df = results_df[results_df["test"].isin(test_list)]
        avg_eff_size_dict[category] = sum(category_df["effect_size"] \
                                          .map(abs)) / len(category_df)

    avg_eff_size_dict["total_avg"] = sum(avg_eff_size_dict.values()) / 3
    print("Effect size averages:\n", avg_eff_size_dict)

    with open(args.output_file, "w+") as f:
        json.dump(avg_eff_size_dict, f, indent=2)