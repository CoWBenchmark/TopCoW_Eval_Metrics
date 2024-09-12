from edge_list_to_variant_str import edge_list_to_variant_str


def test_edge_list_to_variant_str():
    # general test
    assert edge_list_to_variant_str([1, 2, 3, 4], "Ant") == "Var_Ant_1_2_3_4"
    assert edge_list_to_variant_str([2024, 0, 1, 2], "Pos") == "Var_Pos_2024_0_1_2"
    assert edge_list_to_variant_str([0], "Pos") == "Var_Pos_0"
