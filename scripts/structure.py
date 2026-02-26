from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# =============================================================================
# ENUMS
# =============================================================================

class OpType(Enum):
    # Element-wise
    RELU = auto()
    SIGMOID = auto()
    TANH = auto()
    DROPOUT = auto()
    
    # Add variants (detected from shapes)
    ADD_SAME = auto()
    ADD_BIAS = auto()
    ADD_SCALAR = auto()
    ADD_GENERIC = auto()
    
    # Conv/Pool
    CONV2D = auto()
    CONV2D_BIAS = auto()
    MAXPOOL2D = auto()
    AVGPOOL2D = auto()
    GLOBAL_AVGPOOL = auto()
    
    # Linear
    MATMUL = auto()
    SOFTMAX = auto()
    GEMM = auto()
    
    # Shape
    RESHAPE = auto()
    FLATTEN = auto()
    CONCAT = auto()
    TRANSPOSE = auto()   
   
    SLICE = auto()  
    PAD = auto()
    BATCHNORM = auto()
   # IO (special actors in top-level graph)
    LOAD_INPUT = auto()
    LOAD_WEIGHTS = auto()
    SPLIT_WEIGHTS = auto()
    BROADCAST = auto()
    OUTPUT = auto()

    CONSTANT_FILL = auto()
    RANGE_FILL = auto()  # Generates [start, start+step, ..., start+(size-1)*step]

class PortDir(Enum):
    IN = auto()
    OUT = auto()
    CFG = auto()


# =============================================================================
# MAPPING TABLES
# =============================================================================

ONNX_TO_OPTYPE = {
    "relu": OpType.RELU,
    "sigmoid": OpType.SIGMOID,
    "tanh": OpType.TANH,
    "dropout": OpType.DROPOUT,
    "add": OpType.ADD_SAME,
    "conv": OpType.CONV2D,
    "convolution": OpType.CONV2D,
    "maxpool": OpType.MAXPOOL2D,
    "averagepool": OpType.AVGPOOL2D,
    "globalaveragepool": OpType.GLOBAL_AVGPOOL,
    "matmul": OpType.MATMUL,
    "gemm": OpType.GEMM,
    "softmax": OpType.SOFTMAX,
    "reshape": OpType.RESHAPE,
    "flatten": OpType.FLATTEN,
    "concat": OpType.CONCAT,
    "transpose": OpType.TRANSPOSE,
    "slice": OpType.SLICE,
    "pad": OpType.PAD,
    "batchnormalization": OpType.BATCHNORM,
}

OPTYPE_TO_PI = {
    OpType.RELU: "Algo/relu.pi",
    OpType.SIGMOID: "Algo/sigmoid.pi",
    OpType.TANH: "Algo/tanh.pi",
    OpType.DROPOUT: "Algo/dropout.pi",
    
    OpType.ADD_SAME: "Algo/add_same.pi",
    OpType.ADD_BIAS: "Algo/add_bias.pi",
    OpType.ADD_SCALAR: "Algo/add_scalar.pi",
    OpType.ADD_GENERIC: "Algo/add_generic.pi",
    
    OpType.CONV2D: "Algo/conv2d.pi",
    OpType.CONV2D_BIAS: "Algo/conv2d_bias.pi",
    OpType.MAXPOOL2D: "Algo/maxpool2d.pi",
    OpType.AVGPOOL2D: "Algo/avgpool2d.pi",
    OpType.GLOBAL_AVGPOOL: "Algo/global_avgpool.pi",
    
    OpType.MATMUL: "Algo/matmul.pi",
    OpType.SOFTMAX: "Algo/softmax.pi",
    
    OpType.RESHAPE: "Algo/reshape.pi",
    OpType.FLATTEN: "Algo/flatten.pi",
    OpType.CONCAT: "Algo/concat.pi",
    OpType.TRANSPOSE: "Algo/transpose.pi", 

    OpType.SLICE: "Algo/slice.pi",
    OpType.PAD: "Algo/pad.pi",
    OpType.BATCHNORM: "Algo/batchnorm.pi",
    OpType.GEMM: "Algo/gemm.pi",

    OpType.LOAD_INPUT: "",
    OpType.LOAD_WEIGHTS: "",
    OpType.SPLIT_WEIGHTS: "",
    OpType.BROADCAST: "",
    OpType.OUTPUT: "",
    OpType.CONSTANT_FILL: "",
    OpType.RANGE_FILL: "",
}

OPTYPE_TO_H = {
    OpType.RELU: "Code/include/relu.h",
    OpType.SIGMOID: "Code/include/sigmoid.h",
    OpType.TANH: "Code/include/tanh.h",
    OpType.DROPOUT: "Code/include/dropout.h",
    
    OpType.ADD_SAME: "Code/include/add_same.h",
    OpType.ADD_BIAS: "Code/include/add_bias.h",
    OpType.ADD_SCALAR: "Code/include/add_scalar.h",
    OpType.ADD_GENERIC: "Code/include/add_generic.h",
    
    OpType.CONV2D: "Code/include/conv2d.h",
    OpType.CONV2D_BIAS: "Code/include/conv2d_bias.h",
    OpType.MAXPOOL2D: "Code/include/maxpool2d.h",
    OpType.AVGPOOL2D: "Code/include/avgpool2d.h",
    OpType.GLOBAL_AVGPOOL: "Code/include/global_avgpool.h",
    
    OpType.MATMUL: "Code/include/matmul.h",
    OpType.SOFTMAX: "Code/include/softmax.h",
    
    OpType.RESHAPE: "Code/include/reshape.h",
    OpType.FLATTEN: "Code/include/flatten.h",
    OpType.CONCAT: "Code/include/concat.h",
    OpType.TRANSPOSE: "Code/include/transpose.h",

    OpType.SLICE: "Code/include/slice.h",
    OpType.PAD: "Code/include/pad.h",
    OpType.BATCHNORM: "Code/include/batchnorm.h",
    OpType.GEMM: "Code/include/gemm.h",

    OpType.LOAD_INPUT: "",
    OpType.LOAD_WEIGHTS: "",
    OpType.SPLIT_WEIGHTS: "",
    OpType.BROADCAST: "",
    OpType.OUTPUT: "",
    OpType.CONSTANT_FILL: "Code/include/constant_fill.h",
    OpType.RANGE_FILL: "Code/include/range_fill.h",
}

OPTYPE_TO_LOOP_FN = {
    OpType.RELU:    "relu",
    OpType.SIGMOID: "sigmoid",
    OpType.TANH:    "tanh_op",        # NOT "tanh" — clashes with <math.h>
    OpType.DROPOUT: "dropout",

    OpType.ADD_SAME:    "add_same",
    OpType.ADD_BIAS:    "add_bias",
    OpType.ADD_SCALAR:  "add_scalar",
    OpType.ADD_GENERIC: "add_generic",

    OpType.CONV2D:         "conv2d",
    OpType.CONV2D_BIAS:    "conv2d_bias",
    OpType.MAXPOOL2D:      "maxpool2d",
    OpType.AVGPOOL2D:      "avgpool2d",
    OpType.GLOBAL_AVGPOOL: "global_avgpool",

    OpType.MATMUL:   "matmul",
    OpType.SOFTMAX:  "softmax",
    OpType.GEMM:     "gemm",

    OpType.RESHAPE:   "reshape",
    OpType.FLATTEN:   "flatten",
    OpType.CONCAT:    "concat",
    OpType.TRANSPOSE: "transpose",

    OpType.SLICE:     "slice",
    OpType.PAD:       "pad",
    OpType.BATCHNORM: "batchnorm_spatial",  # NOT "batchnorm"

    OpType.LOAD_INPUT:    "load_cifar_image",  # defined in image_loader.cpp
    OpType.LOAD_WEIGHTS:  "loadWeights",        # defined in weight_loader.cpp
    OpType.OUTPUT:        "print_result",        # defined in output.cpp

    OpType.CONSTANT_FILL: "constant_fill",
    OpType.RANGE_FILL:    "range_fill",

    # Structural actors — no loop function.
    OpType.SPLIT_WEIGHTS: "",
    OpType.BROADCAST:     "",
}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class IRParam:

    name: str
    value: int
    unique_id: str = ""  # Set by graph: "size_6272"
    
    def __str__(self) -> str:
        return self.unique_id


@dataclass
class IRPort:
 
    name: str
    direction: PortDir
    rate: int = 1
    actor: Optional[IRActor] = field(default=None, repr=False)


@dataclass
class IRTensor:
  
    name: str
    shape: list[int]
    dtype: str = "float"
    isWeight : bool = False
    
    # Links
    producer: Optional[IRPort] = field(default=None, repr=False)
    consumers: list[IRPort] = field(default_factory=list, repr=False)

    def declare_weight(self):
        self.isWeight = True
    
    @property
    def size(self) -> int:
        result = 1
        for d in self.shape:
            result *= d
        return result
    
    @property
    def height(self) -> int:
        return self.shape[-2] if len(self.shape) >= 2 else 1
    
    @property
    def width(self) -> int:
        return self.shape[-1] if len(self.shape) >= 1 else 1
    
    @property
    def channels(self) -> int:
        return self.shape[-3] if len(self.shape) >= 3 else 1
'''    
    def __str__(self) -> str:
        return f"{self.name}[{self.size}]"
'''

@dataclass
class IRActor:
    """An actor (computation node)."""
    name: str                   # Original ONNX name
    op_type: OpType
    index: int = -1             # Set by graph
    unique_name: str = ""       # Set by graph: "CONV2D_0"
    attributes: dict = field(default_factory=dict)  

    # Ports linked to tensors/params
    inputs : list[tuple[IRPort, IRTensor]] = field(default_factory=list)
    weights: list[tuple[IRPort, IRTensor]] = field(default_factory=list)
    outputs: list[tuple[IRPort, IRTensor]] = field(default_factory=list)
    params : list[tuple[IRPort, IRParam]] = field(default_factory=list)
    
    # For code generation
    source: str = ""            # .pi or .h file
    parallelism: int = 1
    
    def add_input(self, port_name: str, tensor: IRTensor) -> IRPort:
        port = IRPort(name=port_name, direction=PortDir.IN, rate=tensor.size, actor=self)
        self.inputs.append((port, tensor))
        return port
    
    def add_output(self, port_name: str, tensor: IRTensor) -> IRPort:
        port = IRPort(name=port_name, direction=PortDir.OUT, rate=tensor.size, actor=self)
        self.outputs.append((port, tensor))
        return port
    
    def add_param(self, port_name: str, param: IRParam) -> IRPort:
        port = IRPort(name=port_name, direction=PortDir.CFG, rate=1, actor=self)
        self.params.append((port, param))
        return port
    
    def add_weight(self, port_name: str, tensor: IRTensor) -> IRPort:
        port = IRPort(name=port_name, direction=PortDir.IN, rate=tensor.size, actor=self)
        self.weights.append((port, tensor))
        return port
    
    def get_port(self, name: str) -> Optional[IRPort]:
        for port, _ in self.inputs + self.outputs + self.params + self.weights:
            if port.name == name:
                return port
        return None
    
    @property
    def all_ports(self) -> list[IRPort]:
        return [p for p, _ in self.inputs] + [p for p, _ in self.outputs] + [p for p, _ in self.params]
    
    def __str__(self) -> str:
        return self.unique_name


# =============================================================================
# GRAPH
# =============================================================================

@dataclass
class IRGraph:
    """
    The complete graph.

    """
    name: str = "ONNX_GRAPH"
    
    
    _actors: dict[str, IRActor] = field(default_factory=dict)
    _tensors: dict[str, IRTensor] = field(default_factory=dict)
    _params: dict[str, IRParam] = field(default_factory=dict)
    
    # Auto-increment counters for each OP
    _op_counters: dict[OpType, int] = field(default_factory=dict)
    
    # -------------------------------------------------------------------------
    # Actor
    # -------------------------------------------------------------------------
    
    def create_actor(self, op_type: OpType, onnx_name: str = "") -> IRActor:
        # Get and increment counter
        index = self._op_counters.get(op_type, 0)
        self._op_counters[op_type] = index + 1
        
        # Graph decides naming format
        unique_name = f"{op_type.name}_{index}"
        
        actor = IRActor(
            name=onnx_name,
            op_type=op_type,
            index=index,
            unique_name=unique_name,
            source=OPTYPE_TO_H.get(op_type, "")
        )
        self._actors[unique_name] = actor
        return actor
    
    def get_actor(self, unique_name: str) -> Optional[IRActor]:
        return self._actors.get(unique_name)
    
    @property
    def actors(self) -> list[IRActor]:
        return list(self._actors.values())
    
    # -------------------------------------------------------------------------
    # Tensor
    # -------------------------------------------------------------------------
    
    def create_tensor(self, name: str, shape: list[int], dtype: str = "float") -> IRTensor:
        tensor = IRTensor(name=name, shape=shape, dtype=dtype)
        self._tensors[name] = tensor
        return tensor

    
    def get_tensor(self, name: str) -> Optional[IRTensor]:
        return self._tensors.get(name)
    
    def get_or_create_tensor(self, name: str, shape: list[int], dtype: str = "float") -> IRTensor:
        if name in self._tensors:
            return self._tensors[name]
        return self.create_tensor(name, shape, dtype)
    
    @property
    def tensors(self) -> list[IRTensor]:
        return list(self._tensors.values())
    
    # -------------------------------------------------------------------------
    # Param (deduplicated)
    # -------------------------------------------------------------------------
    
    def get_or_create_param(self, name: str, value: int) -> IRParam:
        # Graph decides naming format
        if value >= 0:
            unique_id = f"{name}_{value}"
        else:
            unique_id = f"{name}_{-value}"
        
        if unique_id not in self._params:
            self._params[unique_id] = IRParam(
                name=name,
                value=value,
                unique_id=unique_id
            )
        return self._params[unique_id]
    
    @property
    def params(self) -> list[IRParam]:
        return list(self._params.values())
    
    # -------------------------------------------------------------------------
    # Linking
    # -------------------------------------------------------------------------
    
    def link(self, tensor: IRTensor, producer_port: IRPort, consumer_port: IRPort):
        if producer_port is not None:
            tensor.producer = producer_port
        if consumer_port is not None and consumer_port not in tensor.consumers:
            tensor.consumers.append(consumer_port)
    
    # -------------------------------------------------------------------------
    # Derived edges (for XML generation)
    # -------------------------------------------------------------------------
    
    def get_fifo_edges(self) -> list[tuple[IRPort, IRPort, str]]:
        """Derive FIFO edges: (src_port, dst_port, dtype)."""
        edges = []
        for tensor in self._tensors.values():
            if tensor.producer:
                for consumer in tensor.consumers:
                    edges.append((tensor.producer, consumer, tensor.dtype))
        return edges
    
    def get_dependency_edges(self) -> list[tuple[IRParam, IRPort]]:
        """Derive dependency edges: (param, cfg_port)."""
        edges = []
        for actor in self._actors.values():
            for port, param in actor.params:
                edges.append((param, port))
        return edges
    
    # -------------------------------------------------------------------------
    # Debug
    # -------------------------------------------------------------------------
    
    def print_summary(self):
        fifo = self.get_fifo_edges()
        deps = self.get_dependency_edges()
        print(f"\n{'='*50}")
        print(f"IRGraph: {self.name}")
        print(f"{'='*50}")
        print(f"Actors:  {len(self._actors)}")
        print(f"Tensors: {len(self._tensors)}")
        print(f"Params:  {len(self._params)}")
        print(f"Edges:   {len(fifo)} FIFO, {len(deps)} dependency")
        print(f"{'='*50}\n")
    
    def print_actors(self):
        for actor in self._actors.values():
            print(f"\n{actor.unique_name} ({actor.name}):")
            print(f"  Inputs:  {[(p.name, t.name) for p, t in actor.inputs]}")
            print(f"  Weights:  {[(p.name, t.name) for p, t in actor.weights]}")
            print(f"  Outputs: {[(p.name, t.name) for p, t in actor.outputs]}")
            print(f"  Params:  {[(p.name, pr.unique_id) for p, pr in actor.params]}")
    
    def print_edges(self):
        print("\nFIFO edges:")
        for src, dst, dtype in self.get_fifo_edges():
            src_actor = src.actor.unique_name if src.actor else "?"
            dst_actor = dst.actor.unique_name if dst.actor else "?"
            print(f"  {src_actor}.{src.name} (rate={src.rate}) -> {dst_actor}.{dst.name} (rate={dst.rate}) [{dtype}]")
    
        print("\nDependency edges:")
        for param, port in self.get_dependency_edges():
            dst_actor = port.actor.unique_name if port.actor else "?"
            print(f"  {param.unique_id} -> {dst_actor}.{port.name}")

   
    


