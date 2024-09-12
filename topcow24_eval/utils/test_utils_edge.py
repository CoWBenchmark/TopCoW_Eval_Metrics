from pathlib import Path

from utils_edge import parse_edge_json, parse_edge_yml

TESTDIR = Path("test_assets/edg_metrics")


def test_parse_edge_yml():
    edge_yml = TESTDIR / "topcow_ct_003.yml"
    yml_dict = parse_edge_yml(edge_yml)
    assert yml_dict == {
        "anterior": {
            "L-A1": 0,
            "Acom": 1,
            "3rd-A2": 0,
            "R-A1": 1,
        },
        "posterior": {
            "L-Pcom": 1,
            "L-P1": 1,
            "R-P1": 1,
            "R-Pcom": 1,
        },
    }


def test_parse_edge_json():
    edge_json = TESTDIR / "cow-ant-post-classification.json"
    json_dict = parse_edge_json(edge_json)
    assert json_dict == {
        "anterior": {"L-A1": 1, "Acom": 1, "3rd-A2": 0, "R-A1": 1},
        "posterior": {"L-Pcom": 1, "L-P1": 0, "R-P1": 1, "R-Pcom": 0},
    }
