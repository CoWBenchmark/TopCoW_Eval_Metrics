import json
import pprint

import yaml


def parse_edge_yml(edge_yml) -> dict:
    """
    parse the edge list from yaml file

    input:
        edge_yml: pathlike of the yaml file

    output:
        dictionary of the edge list:
        {
            "anterior": {
                "L-A1": 0,
                "Acom": 1,
                "3rd-A2": 1,
                "R-A1": 0,
            },
            "posterior": {
                "L-Pcom": 1,
                "L-P1": 0,
                "R-P1": 1,
                "R-Pcom": 1,
            },
        }
    """
    print(f"\n--- parse_edge_yml({edge_yml}) ---")

    with open(edge_yml, "r") as file:
        yml_dict = yaml.safe_load(file)

    pprint.pprint(yml_dict, sort_dicts=False)
    print("--- EOF edge_yml ---")

    return yml_dict


def parse_edge_json(edge_json) -> dict:
    """
    similar to parse_edge_yml() but for GC's cow-ant-post-classification.json

    Quick example:

    { "anterior": { "L-A1": 1, "Acom": 1, "3rd-A2": 0, "R-A1": 1 },
    "posterior": { "L-Pcom": 1, "L-P1": 0, "R-P1": 1, "R-Pcom": 0 }}
    """
    print(f"\n--- parse_edge_json({edge_json}) ---")

    with open(edge_json, mode="r", encoding="utf-8") as file:
        json_dict = json.load(file)

    pprint.pprint(json_dict, sort_dicts=False)
    print("--- EOF roi_json ---")

    return json_dict
