from converter import *


    
def main(model_path, output_xml="", output_weights="../bin/weights.bin"):
    model_data = parse_onnx_model(model_path)
    shapes = infer_tensor_shapes(model_data["model"], model_data["initializers"])

    graph = fill_IRGraph(model_data, shapes)

    graph.print_summary()
    graph.print_actors()
    graph.print_edges()
 
    offset_map = create_weights_file_sectioned(model_data["initializers"], output_weights)
    
    add_weight_fork_actors(graph, model_data, offset_map)

    add_load_weights_actors(graph, offset_map)

    write_xml(graph, model_data, "../bin/output.pi")

    return graph


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "../models/mnist-12.onnx"
    main(model_path)
