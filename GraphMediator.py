import streamlit
from graphviz import Digraph

from ComponentData import ComponentData


def build_graph(component_dict: dict[str, ComponentData], graph: Digraph):
    # Style for nodes
    graph.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fillcolor="lightgrey",
        fontname="Helvetica",
    )

    # Add nodes with additional attributes
    for left_id, left_data in component_dict.items():
        label = f"{left_data.component_name} | Class: {left_data.class_name} | Library: {left_data.library} | Access: {left_data.access_type} | Params: {left_data.params} | Output: {left_data.output_var}"
        graph.node(left_id, label=label)
        print(f"Adding graph.node({left_id}, label={label})")

    # Add edges
    for left_id, left_data in component_dict.items():
        for right_id, right_data in component_dict.items():
            if left_id == right_id:
                continue
            if left_data.output_var in right_data.params.values():
                left_label = f"{left_data.component_name} | Class: {left_data.class_name} | Library: {left_data.library} | Access: {left_data.access_type} | Params: {left_data.params}"
                right_label = f"{right_data.component_name} | Class: {right_data.class_name} | Library: {right_data.library} | Access: {right_data.access_type} | Params: {right_data.params}"
                print(f"left_label is {left_label})")
                print(f"right_label is {right_label})")
                graph.edge(left_id, right_id, label=left_data.output_var)
                print(
                    f"Adding graph.edge({left_id}, {right_id}, label={left_data.output_var})"
                )
    return graph


# Add code to parse graph into strings of Python for setup and inference
