def edge_list_to_variant_str(edge_list: list, region: str) -> str:
    """
    convert edge list [0,1,0,1] to variant string
    for sklearn classification metrics
    """
    assert region in ("Ant", "Pos"), "invalid region"

    # Convert each element to a string and join them
    edge_str = "_".join(map(str, edge_list))

    prefix = f"Var_{region}_"

    return prefix + edge_str
