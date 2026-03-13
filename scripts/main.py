from converter import *
from block_skipping_pass import write_block_skipping_xml, apply_block_skipping_pass
from pi_generator import *
from model_codegen import generate_model_actors
    
def main(model_path, hierarchial = 0, isGeneratedKernels = 1,output_xml="", output_weights="../bin/weights.bin"):
    model_data = parse_onnx_model(model_path)
    shapes = infer_tensor_shapes(model_data["model"], model_data["initializers"])

    offset_map = create_weights_file_sectioned(model_data["initializers"], output_weights)
    graph_name = os.path.splitext(os.path.basename(output_xml))[0]
    graph = fill_IRGraph(model_data, shapes, offset_map, hierarchial, graph_name)

    graph.print_summary()
    graph.print_actors()
    graph.print_edges()

    #  generate runtime-backed actors and get the per-instance fn names,
    #  or fall back to hand-written headers / OPTYPE_TO_LOOP_FN when disabled
    if isGeneratedKernels:
        loop_fn_map = generate_model_actors(graph, graph_name)
    else:
        loop_fn_map = None
    write_xml(graph, model_data, loop_fn_map , output_xml)

    n = apply_block_skipping_pass(graph)               # in-place, returns block count
    print(f"Applied BlockDrop pass: {n} residual blocks detected and transformed.")
    write_block_skipping_xml(graph, model_data, "../output_graphs/dy_resnet_blockdrop.pi")

    if hierarchial:
        generate_all_pi_files(graph, "../sources/pi")

    return graph


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py <model.onnx> [isHierarchial?] [isGeneratedKernels?] [output.xml] [weights.bin] ")
        print("Example: python main.py ../models/mnist-12.onnx 0 1 ../output_graphs/mnist_12.pi ../bin/weights.bin")
        sys.exit(1)

    model_path = sys.argv[1]
    hierarchial = bool(int(sys.argv[2])) if len(sys.argv) > 2 else False
    isGeneratedKernels = bool(int(sys.argv[3])) if len(sys.argv) > 3 else True
    output_xml = sys.argv[4] if len(sys.argv) > 4 else "../output_graphs/output.pi"
    output_weights = sys.argv[5] if len(sys.argv) > 5 else "../bin/weights.bin"

    main(model_path, hierarchial, isGeneratedKernels, output_xml, output_weights)
