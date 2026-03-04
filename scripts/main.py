from converter import *
from pi_generator import *

    
def main(model_path, hierarchial = 0, output_xml="", output_weights="../bin/weights.bin"):
    model_data = parse_onnx_model(model_path)
    shapes = infer_tensor_shapes(model_data["model"], model_data["initializers"])

    offset_map = create_weights_file_sectioned(model_data["initializers"], output_weights)

    graph = fill_IRGraph(model_data, shapes, offset_map, hierarchial)

    graph.print_summary()
    graph.print_actors()
    graph.print_edges()

    write_xml(graph, model_data, output_xml)

    if hierarchial:
        generate_all_pi_files(graph, "../sources/pi")

    return graph


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py <model.onnx> [isHierarchial?] [output.xml] [weights.bin]")
        print("Example: python main.py ../models/mnist-12.onnx 0 ../output_graphs/mnist_12.xml ../bin/weights.bin")
        sys.exit(1)

    model_path = sys.argv[1] 
    hierarchial = bool(int(sys.argv[2])) if len(sys.argv) > 2 else False
    output_xml = sys.argv[3] if len(sys.argv) > 3 else "../output_graphs/output.pi"
    output_weights = sys.argv[4] if len(sys.argv) > 4 else "../bin/weights.bin"

    main(model_path, hierarchial,output_xml, output_weights)
