mod graph;
mod task_graph;

pub use graph::{
    AccessVertexError, AddEdgeStatus, EdgeAccessError, Graph, RemoveEdgeStatus, VertexID,
    VertexIDIterator, VertexIterator,
};

pub use task_graph::{
    build_system_graph, BuildSystemGraphError, SystemDesc, TaskGraph, TaskVertex,
    TaskVertexIterator,
};
