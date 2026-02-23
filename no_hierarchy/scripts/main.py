from converter import *
from block_skipping_pass import write_block_skipping_xml, apply_block_skipping_pass
    
def main(model_path, output_xml="", output_weights="../bin/weights.bin"):
    model_data = parse_onnx_model(model_path)
    shapes = infer_tensor_shapes(model_data["model"], model_data["initializers"])

    offset_map = create_weights_file_sectioned(model_data["initializers"], output_weights)

    graph = fill_IRGraph(model_data, shapes, offset_map)

    graph.print_summary()
    graph.print_actors()
    graph.print_edges()

    write_xml(graph, model_data, output_xml) 

    n = apply_block_skipping_pass(graph)               # in-place, returns block count
    print(f"Applied BlockDrop pass: {n} residual blocks detected and transformed.")
    write_block_skipping_xml(graph, model_data, "../output_graphs/resnet_blockdrop_V1.pi")
    return graph


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py <model.onnx> [output.xml] [weights.bin]")
        print("Example: python main.py ../models/mnist-12.onnx ../output_graphs/mnist_12.xml ../bin/weights.bin")
        sys.exit(1)

    model_path = sys.argv[1] 
    output_xml = sys.argv[2] if len(sys.argv) > 2 else "../output_graphs/output.pi"
    output_weights = sys.argv[3] if len(sys.argv) > 3 else "../bin/weights.bin"

    main(model_path, output_xml, output_weights)