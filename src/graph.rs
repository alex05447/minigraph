use std::fmt::{Display, Formatter};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use super::task_graph::TaskGraph;

use minihandle::IndexManager;
use num_traits::{PrimInt, Unsigned};

/// Graph vertex ID trait - primitive unsigned integer.
pub trait VertexID: PrimInt + Unsigned + Hash {}

impl<T: PrimInt + Unsigned + Hash> VertexID for T {}

type Vertices<VID, T> = HashMap<VID, T>;
type Edges<VID> = HashMap<VID, HashSet<VID>>;

fn num_neighbors_impl<VID: VertexID>(edges: &Edges<VID>, vertex_id: VID) -> usize {
    edges
        .get(&vertex_id)
        .map_or(0, |edges| edges.len())
}

fn neighbors_impl<VID: VertexID>(edges: &Edges<VID>, vertex_id: VID) -> VertexIDIterator<'_, VID> {
    VertexIDIterator::new(
        edges
            .get(&vertex_id)
            .map_or(None, |edges| Some(edges.iter())),
    )
}

/// Represents a digraph.
///
/// `VID` must be a primitive unsigned integer type, used to ID vertices.
/// `T` is arbitrary vertex payload.
#[derive(Clone)]
pub struct Graph<VID: VertexID, T> {
    vertex_ids: IndexManager<VID>,

    vertices: Vertices<VID, T>,

    roots: HashSet<VID>,
    leaves: HashSet<VID>,

    num_edges: usize,
    in_edges: Edges<VID>,
    out_edges: Edges<VID>,
}

#[derive(Clone, PartialEq, Debug)]
pub enum AccessVertexError {
    /// The vertex ID was invalid.
    InvalidVertexID,
}

impl Display for AccessVertexError {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        use AccessVertexError::*;

        match self {
            InvalidVertexID => write!(f, "The vertex ID was invalid."),
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum AddEdgeStatus {
    /// A new directed edge was added to the graph between the two vertices.
    Added,
    /// A directed edge between the two vertices already exists.
    AlreadyExists,
}

impl Display for AddEdgeStatus {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        use AddEdgeStatus::*;

        match self {
            Added => write!(f, "A new directed edge was added to the graph between the two vertices."),
            AlreadyExists => write!(f, "A directed edge between the two vertices already exists."),
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum RemoveEdgeStatus {
    /// A previously existing directed edge between the two vertices was removed.
    Removed,
    /// A directed edge between the two vertices does not exist.
    DoesNotExist,
}

impl Display for RemoveEdgeStatus {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        use RemoveEdgeStatus::*;

        match self {
            Removed => write!(f, "A previously existing directed edge between the two vertices was removed."),
            DoesNotExist => write!(f, "A directed edge between the two vertices does not exist."),
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum AccessEdgeError {
    /// The `from` vertex ID was invalid.
    InvalidFromVertex,
    /// The `to` vertex ID was invalid.
    InvalidToVertex,
    /// Loop edge (`from` and `to` vertices are the same).
    LoopEdge,
}

impl Display for AccessEdgeError {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        use AccessEdgeError::*;

        match self {
            InvalidFromVertex => write!(f, "The `from` vertex ID was invalid."),
            InvalidToVertex => write!(f, "The `to` vertex ID was invalid."),
            LoopEdge => write!(f, "Loop edge (`from` and `to` vertices are the same)."),
        }
    }
}

impl<VID: VertexID, T> Graph<VID, T> {
    pub fn new() -> Self {
        Self {
            vertex_ids: IndexManager::new(),
            vertices: HashMap::new(),
            roots: HashSet::new(),
            leaves: HashSet::new(),
            num_edges: 0,
            in_edges: HashMap::new(),
            out_edges: HashMap::new(),
        }
    }

    /// Adds a vertex payload to the graph, returning its unique vertex ID.
    ///
    /// The vertex has no inbound / outbound edges initially.
    pub fn add_vertex(&mut self, vertex: T) -> VID {
        let vertex_id = self.vertex_ids.create();

        let prev = self.vertices.insert(vertex_id, vertex);
        assert!(prev.is_none(), "Duplicate vertex ID in the graph.");

        self.roots.insert(vertex_id);
        self.leaves.insert(vertex_id);

        vertex_id
    }

    /// If `vertex_id` is valid, removes the vertex payload from the graph and returns it.
    ///
    /// Removes the edges to / from the removed vertex.
    pub fn remove_vertex(&mut self, vertex_id: VID) -> Result<T, AccessVertexError> {
        if let Some(vertex) = self.vertices.remove(&vertex_id) {
            if let Some(in_edges) = self.in_edges.remove(&vertex_id) {
                for in_neighbor in in_edges.iter() {
                    if let Ok(RemoveEdgeStatus::Removed) = self.remove_edge(*in_neighbor, vertex_id)
                    {
                    } else {
                        panic!("Invalid edge state.");
                    }
                }
            }

            if let Some(out_edges) = self.out_edges.remove(&vertex_id) {
                for out_neighbor in out_edges.iter() {
                    if let Ok(RemoveEdgeStatus::Removed) =
                        self.remove_edge(vertex_id, *out_neighbor)
                    {
                    } else {
                        panic!("Invalid edge state.");
                    }
                }
            }

            self.roots.remove(&vertex_id);
            self.leaves.remove(&vertex_id);

            self.vertex_ids.destroy(vertex_id);

            Ok(vertex)
        } else {
            Err(AccessVertexError::InvalidVertexID)
        }
    }

    /// Returns the current number of vertices in the graph.
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// If `vertex_id` is valid, returns the reference to the vertex payload.
    pub fn vertex(&self, vertex_id: VID) -> Result<&T, AccessVertexError> {
        self.vertices
            .get(&vertex_id)
            .ok_or(AccessVertexError::InvalidVertexID)
    }

    /// If `vertex_id` is valid, returns the reference to the vertex payload.
    pub fn vertex_mut(&mut self, vertex_id: VID) -> Result<&mut T, AccessVertexError> {
        self.vertices
            .get_mut(&vertex_id)
            .ok_or(AccessVertexError::InvalidVertexID)
    }

    /// Returns an iterator over all vertices in the graph, in no particular order.
    pub fn vertices(&self) -> VertexIterator<'_, VID, T> {
        Graph::vertices_impl(&self.vertices)
    }

    pub fn vertices_impl(vertices: &Vertices<VID, T>) -> VertexIterator<'_, VID, T> {
        VertexIterator::new(vertices.iter())
    }

    /// If `from` and `to` are valid vertex ID's in the graph, adds a directed edge between them.
    ///
    /// Does nothing if the edge already exists.
    pub fn add_edge(&mut self, from: VID, to: VID) -> Result<AddEdgeStatus, AccessEdgeError> {
        use AccessEdgeError::*;

        if !self.vertices.contains_key(&from) {
            return Err(InvalidFromVertex);
        }

        if !self.vertices.contains_key(&to) {
            return Err(InvalidToVertex);
        }

        if from == to {
            return Err(LoopEdge);
        }

        self.roots.remove(&to);
        self.leaves.remove(&from);

        if self
            .in_edges
            .entry(to)
            .or_insert(HashSet::new())
            .insert(from)
            && self
                .out_edges
                .entry(from)
                .or_insert(HashSet::new())
                .insert(to)
        {
            self.num_edges += 1;

            Ok(AddEdgeStatus::Added)

        } else {
            Ok(AddEdgeStatus::AlreadyExists)
        }
    }

    /// If `from` and `to` are valid vertex ID's in the graph, returns `true` if a directed edge exists in the graph
    /// between vertices `from` and `to`.
    pub fn has_edge(&self, from: VID, to: VID) -> Result<bool, AccessEdgeError> {
        use AccessEdgeError::*;

        if !self.vertices.contains_key(&from) {
            return Err(InvalidFromVertex);
        }

        if !self.vertices.contains_key(&to) {
            return Err(InvalidToVertex);
        }

        if from == to {
            return Err(LoopEdge);
        }

        if let Some(out_edges) = self.out_edges.get(&from) {
            Ok(out_edges.contains(&to))

        } else {
            Ok(false)
        }
    }

    /// If `from` and `to` are valid vertex ID's in the graph, removes a directed edge between them.
    ///
    /// Does nothing if the edge does not exist.
    pub fn remove_edge(&mut self, from: VID, to: VID) -> Result<RemoveEdgeStatus, AccessEdgeError> {
        use AccessEdgeError::*;

        if !self.vertices.contains_key(&from) {
            return Err(InvalidFromVertex);
        }

        if !self.vertices.contains_key(&to) {
            return Err(InvalidToVertex);
        }

        if from == to {
            return Err(LoopEdge);
        }

        Ok(
            Self::remove_edge_impl(
                &mut self.roots,
                &mut self.leaves,
                &mut self.num_edges,
                &mut self.in_edges,
                &mut self.out_edges,
                from,
                to,
            )
        )
    }

    fn remove_edge_impl(
        roots: &mut HashSet<VID>,
        leaves: &mut HashSet<VID>,
        num_edges: &mut usize,
        in_edges: &mut Edges<VID>,
        out_edges: &mut Edges<VID>,

        from: VID,
        to: VID
    ) -> RemoveEdgeStatus {
        if let Some(to_in_edges) = in_edges.get_mut(&to) {
            if !to_in_edges.remove(&from) {
                return RemoveEdgeStatus::DoesNotExist;
            }

            if to_in_edges.is_empty() {
                in_edges.remove(&to);
            }
        } else {
            return RemoveEdgeStatus::DoesNotExist;
        }

        debug_assert!(*num_edges > 0);
        *num_edges -= 1;

        let from_out_edges = out_edges
            .get_mut(&from)
            .expect("In / out edge mismatch.");

        from_out_edges.take(&to).expect("In / out edge mismatch.");

        if from_out_edges.is_empty() {
            out_edges.remove(&from);
        }

        if num_neighbors_impl(in_edges, to) == 0 {
            let was_not_a_root = roots.insert(to);
            assert!(was_not_a_root, "Root vertex had inbound edges.")
        }

        if num_neighbors_impl(out_edges, from) == 0 {
            let was_not_a_leaf = leaves.insert(from);
            assert!(was_not_a_leaf, "Leaf vertex had outbound edges.")
        }

        RemoveEdgeStatus::Removed
    }

    /// Returns the current number of edges in the graph.
    pub fn num_edges(&self) -> usize {
        self.num_edges
    }

    /// Returns the current number of root vertices (with no inbound edges) in the graph.
    pub fn num_roots(&self) -> usize {
        self.roots.len()
    }

    /// Returns an iterator over all root vertices (with no inbound edges) in the graph, in no particular order.
    pub fn roots(&self) -> VertexIDIterator<'_, VID> {
        VertexIDIterator::new(Some(self.roots.iter()))
    }

    /// Returns the current number of leaf vertices (with no outbound edges) in the graph.
    pub fn num_leaves(&self) -> usize {
        self.leaves.len()
    }

    /// Returns an iterator over all leaf vertices (with no outbound edges) in the graph, in no particular order.
    pub fn leaves(&self) -> VertexIDIterator<'_, VID> {
        VertexIDIterator::new(Some(self.leaves.iter()))
    }

    /// If `vertex_id` is valid, returns the number of inbound esges to the vertex.
    pub fn num_in_neighbors(&self, vertex_id: VID) -> Result<usize, AccessVertexError> {
        if !self.vertices.contains_key(&vertex_id) {
            Err(AccessVertexError::InvalidVertexID)
        } else {
            Ok(num_neighbors_impl(&self.in_edges, vertex_id))
        }
    }

    /// If `vertex_id` is valid, returns an iterator over all vertices whith inbound edges to the vertex, in no particular order.
    pub fn in_neighbors(
        &self,
        vertex_id: VID,
    ) -> Result<VertexIDIterator<'_, VID>, AccessVertexError> {
        if !self.vertices.contains_key(&vertex_id) {
            Err(AccessVertexError::InvalidVertexID)
        } else {
            Ok(neighbors_impl(&self.in_edges, vertex_id))
        }
    }

    /// If `vertex_id` is valid, returns the number of outbound esges from the vertex.
    pub fn num_out_neighbors(&self, vertex_id: VID) -> Result<usize, AccessVertexError> {
        if !self.vertices.contains_key(&vertex_id) {
            Err(AccessVertexError::InvalidVertexID)
        } else {
            Ok(num_neighbors_impl(&self.out_edges, vertex_id))
        }
    }

    /// If `vertex_id` is valid, returns an iterator over all vertices whith outbound edges from the vertex, in no particular order.
    pub fn out_neighbors(
        &self,
        vertex_id: VID,
    ) -> Result<VertexIDIterator<'_, VID>, AccessVertexError> {
        if !self.vertices.contains_key(&vertex_id) {
            Err(AccessVertexError::InvalidVertexID)
        } else {
            Ok(neighbors_impl(&self.out_edges, vertex_id))
        }
    }

    /// Returns `true` if the graph contains a cycle.
    pub fn is_cyclic(&self) -> bool {
        if self.num_vertices() > 0 && (self.num_roots() == 0 || self.num_leaves() == 0) {
            return true;
        }

        for root in self.roots() {
            let mut stack = HashSet::new();

            if self.is_cyclic_dfs(*root, &mut stack) {
                return true;
            }
        }

        false
    }

    fn is_cyclic_dfs(&self, vertex: VID, stack: &mut HashSet<VID>) -> bool {
        for child in self.out_neighbors(vertex).unwrap() {
            if stack.contains(child) {
                return true;
            }

            stack.insert(*child);

            if self.is_cyclic_dfs(*child, stack) {
                return true;
            }

            stack.remove(child);
        }

        false
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum TransitiveReductionError {
    /// The graph contains a cycle and the transitive reduction is non-unique.
    CyclicGraph,
}

impl Display for TransitiveReductionError {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        use TransitiveReductionError::*;

        match self {
            CyclicGraph => write!(f, "The graph contains a cycle and the transitive reduction is non-unique."),
        }
    }
}

impl<VID: VertexID, T: Clone> Graph<VID, T> {
    /// Returns a copy of the graph with redundant transitive edges removed.
    ///
    /// The implementation is suboptimal but gets the job done for relatively small graphs.
    ///
    /// https://en.wikipedia.org/wiki/Transitive_reduction
    pub fn transitive_reduction(&self) -> Result<Self, TransitiveReductionError> {
        use TransitiveReductionError::*;

        if self.is_cyclic() {
            return Err(CyclicGraph);
        }

        let mut graph = self.clone();

        let vertices = graph.vertices().map(|(vid, _)| *vid).collect::<Vec<_>>();

        for vertex in vertices.iter() {
            let mut processed_set = HashSet::new();

            let children = graph.out_neighbors(*vertex).unwrap().map(|vid| *vid).collect::<Vec<_>>();

            for child in children.iter() {
                graph.transitive_reduction_process_vertex(
                    *vertex,
                    *child,
                    &mut processed_set,
                );
            }
        }

        Ok(graph)
    }

    fn transitive_reduction_process_vertex(
        &mut self,
        vertex: VID,
        child: VID,
        processed_set: &mut HashSet<VID>,
    ) {
        if processed_set.contains(&child) {
            return;
        }

        let children = self.out_neighbors(child).unwrap().map(|vid| *vid).collect::<Vec<_>>();

        for _child in children.iter() {
            Self::remove_edge_impl(
                &mut self.roots,
                &mut self.leaves,
                &mut self.num_edges,
                &mut self.in_edges,
                &mut self.out_edges,

                vertex,
                *_child,
            );

            self.transitive_reduction_process_vertex(
                vertex,
                *_child,
                processed_set,
            );
        }

        processed_set.insert(child);
    }

    /// Creates a [`TaskGraph`] representation of the graph.
    ///
    /// [`TaskGraph`]: struct.TaskGraph.html
    pub fn task_graph(&self) -> TaskGraph<VID, T> {
        TaskGraph::new(self, &self.vertices, &self.roots, &self.out_edges)
    }
}

/// Iterates over graph vertices, returning their vertex ID.
pub struct VertexIDIterator<'a, VID: VertexID> {
    vertex_ids: Option<std::collections::hash_set::Iter<'a, VID>>,
}

impl<'a, VID: VertexID> VertexIDIterator<'a, VID> {
    pub(super) fn new(vertex_ids: Option<std::collections::hash_set::Iter<'a, VID>>) -> Self {
        Self { vertex_ids }
    }
}

impl<'a, VID: VertexID> std::iter::Iterator for VertexIDIterator<'a, VID> {
    type Item = &'a VID;

    fn next(&mut self) -> Option<Self::Item> {
        self.vertex_ids
            .as_mut()
            .map_or(None, |vertex_ids| vertex_ids.next())
    }
}

/// Iterates over graph vertices, returning their vertex ID and payload.
pub struct VertexIterator<'a, VID: VertexID, T> {
    vertices: std::collections::hash_map::Iter<'a, VID, T>,
}

impl<'a, VID: VertexID, T> VertexIterator<'a, VID, T> {
    pub(super) fn new(vertices: std::collections::hash_map::Iter<'a, VID, T>) -> Self {
        Self { vertices }
    }
}

impl<'a, VID: VertexID, T> std::iter::Iterator for VertexIterator<'a, VID, T> {
    type Item = (&'a VID, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        self.vertices.next()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn graph() {
        // Empty graph.
        let mut graph = Graph::<u32, i32>::new();

        assert_eq!(graph.num_vertices(), 0);
        assert_eq!(graph.num_edges(), 0);
        assert_eq!(graph.num_roots(), 0);
        assert_eq!(graph.num_leaves(), 0);

        // One vertex.
        let v0 = graph.add_vertex(-1);

        assert_eq!(*graph.vertex(v0).unwrap(), -1);

        assert_eq!(graph.num_vertices(), 1);
        assert_eq!(graph.num_edges(), 0);
        assert_eq!(graph.num_roots(), 1);
        assert_eq!(graph.num_leaves(), 1);

        // Change the vertex value.
        *graph.vertex_mut(v0).unwrap() = -2;

        assert_eq!(*graph.vertex(v0).unwrap(), -2);

        assert_eq!(graph.num_vertices(), 1);
        assert_eq!(graph.num_edges(), 0);
        assert_eq!(graph.num_roots(), 1);
        assert_eq!(graph.num_leaves(), 1);

        // Second vertex.
        let v1 = graph.add_vertex(-3);

        assert_eq!(*graph.vertex(v1).unwrap(), -3);

        assert_eq!(graph.num_vertices(), 2);
        assert_eq!(graph.num_edges(), 0);
        assert_eq!(graph.num_roots(), 2);
        assert_eq!(graph.num_leaves(), 2);

        // Add an edge v0 -> v1.
        assert_eq!(graph.add_edge(v0, v1), Ok(AddEdgeStatus::Added));

        assert_eq!(graph.num_vertices(), 2);
        assert_eq!(graph.num_edges(), 1);
        assert_eq!(graph.num_roots(), 1);
        assert_eq!(graph.num_leaves(), 1);

        // v0 is a single root.
        for root in graph.roots() {
            assert_eq!(*root, v0);
        }

        // v1 is its single out neighbor.
        for neighbor in graph.out_neighbors(v0).unwrap() {
            assert_eq!(*neighbor, v1);
        }

        // v1 is a single leaf.
        for leaf in graph.leaves() {
            assert_eq!(*leaf, v1);
        }

        // v0 is its single in neighbor.
        for neighbor in graph.in_neighbors(v1).unwrap() {
            assert_eq!(*neighbor, v0);
        }

        assert_eq!(graph.add_edge(v0, v1), Ok(AddEdgeStatus::AlreadyExists));

        assert_eq!(graph.num_vertices(), 2);
        assert_eq!(graph.num_edges(), 1);
        assert_eq!(graph.num_roots(), 1);

        for leaf in graph.leaves() {
            assert_eq!(*leaf, v1);
        }

        for neighbor in graph.in_neighbors(v1).unwrap() {
            assert_eq!(*neighbor, v0);
        }

        // Remove the edge.
        assert_eq!(graph.remove_edge(v0, v1), Ok(RemoveEdgeStatus::Removed));

        assert_eq!(graph.num_vertices(), 2);
        assert_eq!(graph.num_edges(), 0);
        assert_eq!(graph.num_roots(), 2);
        assert_eq!(graph.num_leaves(), 2);

        assert_eq!(*graph.vertex(v0).unwrap(), -2);
        assert_eq!(*graph.vertex(v1).unwrap(), -3);

        assert_eq!(
            graph.remove_edge(v0, v1),
            Ok(RemoveEdgeStatus::DoesNotExist)
        );

        // Another vertex.
        let v2 = graph.add_vertex(7);

        assert_eq!(*graph.vertex(v2).unwrap(), 7);

        assert_eq!(graph.num_vertices(), 3);
        assert_eq!(graph.num_edges(), 0);
        assert_eq!(graph.num_roots(), 3);
        assert_eq!(graph.num_leaves(), 3);

        // v0 -> v1
        assert_eq!(graph.add_edge(v0, v1), Ok(AddEdgeStatus::Added));

        assert_eq!(graph.num_vertices(), 3);
        assert_eq!(graph.num_edges(), 1);
        assert_eq!(graph.num_roots(), 2);
        assert_eq!(graph.num_leaves(), 2);

        // v0 -> v1 -> v2
        assert_eq!(graph.add_edge(v1, v2), Ok(AddEdgeStatus::Added));

        assert_eq!(graph.num_vertices(), 3);
        assert_eq!(graph.num_edges(), 2);
        assert_eq!(graph.num_roots(), 1);
        assert_eq!(graph.num_leaves(), 1);

        // Root to leaf.
        for vertex in graph.roots() {
            assert_eq!(*vertex, v0);
            assert_eq!(*graph.vertex(*vertex).unwrap(), -2);

            for vertex in graph.out_neighbors(*vertex).unwrap() {
                assert_eq!(*vertex, v1);
                assert_eq!(*graph.vertex(*vertex).unwrap(), -3);

                for vertex in graph.out_neighbors(*vertex).unwrap() {
                    assert_eq!(*vertex, v2);
                    assert_eq!(*graph.vertex(*vertex).unwrap(), 7);
                }
            }
        }

        // Leaf to root.
        for vertex in graph.leaves() {
            assert_eq!(*vertex, v2);
            assert_eq!(*graph.vertex(*vertex).unwrap(), 7);

            for vertex in graph.in_neighbors(*vertex).unwrap() {
                assert_eq!(*vertex, v1);
                assert_eq!(*graph.vertex(*vertex).unwrap(), -3);

                for vertex in graph.in_neighbors(*vertex).unwrap() {
                    assert_eq!(*vertex, v0);
                    assert_eq!(*graph.vertex(*vertex).unwrap(), -2);
                }
            }
        }
    }

    #[test]
    fn non_cyclic() {
        // { A -> B -> C -> D -> E; A -> F; D -> F }

        let mut graph = Graph::<u32, ()>::new();

        let a = graph.add_vertex(());
        let b = graph.add_vertex(());
        let c = graph.add_vertex(());
        let d = graph.add_vertex(());
        let e = graph.add_vertex(());
        let f = graph.add_vertex(());

        graph.add_edge(a, b).unwrap();
        graph.add_edge(b, c).unwrap();
        graph.add_edge(c, d).unwrap();
        graph.add_edge(d, e).unwrap();

        graph.add_edge(a, f).unwrap();
        graph.add_edge(d, f).unwrap();

        assert_eq!(graph.num_vertices(), 6);
        assert_eq!(graph.num_edges(), 6);
        assert_eq!(graph.num_roots(), 1);
        assert_eq!(graph.num_leaves(), 2);

        assert!(!graph.is_cyclic());
    }

    #[test]
    fn cyclic() {
        // { A -> B -> C -> D -> E; D -> F -> B }

        let mut graph = Graph::<u32, ()>::new();

        let a = graph.add_vertex(());
        let b = graph.add_vertex(());
        let c = graph.add_vertex(());
        let d = graph.add_vertex(());
        let e = graph.add_vertex(());
        let f = graph.add_vertex(());

        graph.add_edge(a, b).unwrap();
        graph.add_edge(b, c).unwrap();
        graph.add_edge(c, d).unwrap();
        graph.add_edge(d, e).unwrap();

        graph.add_edge(d, f).unwrap();
        graph.add_edge(f, b).unwrap();

        assert_eq!(graph.num_vertices(), 6);
        assert_eq!(graph.num_edges(), 6);
        assert_eq!(graph.num_roots(), 1);
        assert_eq!(graph.num_leaves(), 1);

        assert!(graph.is_cyclic());
    }

    #[test]
    fn transitive_reduction() {
        // { A -> B -> C -> D; A -> C; A -> D } => { A -> B -> C -> D }

        let mut graph = Graph::<u32, ()>::new();

        let a = graph.add_vertex(());
        let b = graph.add_vertex(());
        let c = graph.add_vertex(());
        let d = graph.add_vertex(());

        graph.add_edge(a, b).unwrap();
        graph.add_edge(b, c).unwrap();
        graph.add_edge(c, d).unwrap();

        graph.add_edge(a, c).unwrap(); // Redundant edge.
        graph.add_edge(a, d).unwrap(); // Redundant edge.

        assert_eq!(graph.num_vertices(), 4);
        assert_eq!(graph.num_edges(), 5);
        assert_eq!(graph.num_roots(), 1);
        assert_eq!(graph.num_leaves(), 1);

        let reduced_graph = graph.transitive_reduction().unwrap();

        assert_eq!(reduced_graph.num_vertices(), 4);
        assert_eq!(reduced_graph.num_edges(), 3);
        assert_eq!(reduced_graph.num_roots(), 1);
        assert_eq!(reduced_graph.num_leaves(), 1);

        for root in reduced_graph.roots() {
            assert_eq!(*root, a);

            assert_eq!(reduced_graph.num_out_neighbors(*root).unwrap(), 1);

            for child in reduced_graph.out_neighbors(*root).unwrap() {
                assert_eq!(*child, b);

                assert_eq!(reduced_graph.num_out_neighbors(*child).unwrap(), 1);

                for child in reduced_graph.out_neighbors(*child).unwrap() {
                    assert_eq!(*child, c);

                    assert_eq!(reduced_graph.num_out_neighbors(*child).unwrap(), 1);

                    for child in reduced_graph.out_neighbors(*child).unwrap() {
                        assert_eq!(*child, d);

                        assert_eq!(reduced_graph.num_out_neighbors(*child).unwrap(), 0);
                    }
                }
            }
        }
    }

    #[test]
    fn transitive_reduction_non_root() {
        // { A -> B -> C -> D -> E; B -> D; B -> E } => { A -> B -> C -> D -> E }

        let mut graph = Graph::<u32, ()>::new();

        let a = graph.add_vertex(());
        let b = graph.add_vertex(());
        let c = graph.add_vertex(());
        let d = graph.add_vertex(());
        let e = graph.add_vertex(());

        graph.add_edge(a, b).unwrap();
        graph.add_edge(b, c).unwrap();
        graph.add_edge(c, d).unwrap();
        graph.add_edge(d, e).unwrap();

        graph.add_edge(b, d).unwrap(); // Redundant edge.
        graph.add_edge(b, e).unwrap(); // Redundant edge.

        assert_eq!(graph.num_vertices(), 5);
        assert_eq!(graph.num_edges(), 6);
        assert_eq!(graph.num_roots(), 1);
        assert_eq!(graph.num_leaves(), 1);

        let reduced_graph = graph.transitive_reduction().unwrap();

        assert_eq!(reduced_graph.num_vertices(), 5);
        assert_eq!(reduced_graph.num_edges(), 4);
        assert_eq!(reduced_graph.num_roots(), 1);
        assert_eq!(reduced_graph.num_leaves(), 1);

        for root in reduced_graph.roots() {
            assert_eq!(*root, a);

            assert_eq!(reduced_graph.num_out_neighbors(*root).unwrap(), 1);

            for child in reduced_graph.out_neighbors(*root).unwrap() {
                assert_eq!(*child, b);

                assert_eq!(reduced_graph.num_out_neighbors(*child).unwrap(), 1);

                for child in reduced_graph.out_neighbors(*child).unwrap() {
                    assert_eq!(*child, c);

                    assert_eq!(reduced_graph.num_out_neighbors(*child).unwrap(), 1);

                    for child in reduced_graph.out_neighbors(*child).unwrap() {
                        assert_eq!(*child, d);

                        assert_eq!(reduced_graph.num_out_neighbors(*child).unwrap(), 1);

                        for child in reduced_graph.out_neighbors(*child).unwrap() {
                            assert_eq!(*child, e);

                            assert_eq!(reduced_graph.num_out_neighbors(*child).unwrap(), 0);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn transitive_reduction_cycle_no_roots() {
        // { A -> B -> C -> A }

        let mut graph = Graph::<u32, ()>::new();

        let a = graph.add_vertex(());
        let b = graph.add_vertex(());
        let c = graph.add_vertex(());

        graph.add_edge(a, b).unwrap();
        graph.add_edge(b, c).unwrap();
        graph.add_edge(c, a).unwrap();

        assert_eq!(graph.num_vertices(), 3);
        assert_eq!(graph.num_edges(), 3);
        assert_eq!(graph.num_roots(), 0);
        assert_eq!(graph.num_leaves(), 0);

        assert_eq!(graph.transitive_reduction().err().unwrap(), TransitiveReductionError::CyclicGraph);
    }

    #[test]
    fn transitive_reduction_cycle_inner() {
        // { A -> B -> C -> D -> E; D -> F -> B }

        let mut graph = Graph::<u32, ()>::new();

        let a = graph.add_vertex(());
        let b = graph.add_vertex(());
        let c = graph.add_vertex(());
        let d = graph.add_vertex(());
        let e = graph.add_vertex(());
        let f = graph.add_vertex(());

        graph.add_edge(a, b).unwrap();
        graph.add_edge(b, c).unwrap();
        graph.add_edge(c, d).unwrap();
        graph.add_edge(d, e).unwrap();

        graph.add_edge(d, f).unwrap();
        graph.add_edge(f, b).unwrap();

        assert_eq!(graph.num_vertices(), 6);
        assert_eq!(graph.num_edges(), 6);
        assert_eq!(graph.num_roots(), 1);
        assert_eq!(graph.num_leaves(), 1);

        assert_eq!(graph.transitive_reduction().err().unwrap(), TransitiveReductionError::CyclicGraph);
    }

    #[test]
    fn transitive_reduction_cycle_inner_non_root() {
        // { A -> B -> C -> D -> E -> F; E -> G -> C }

        let mut graph = Graph::<u32, ()>::new();

        let a = graph.add_vertex(());
        let b = graph.add_vertex(());
        let c = graph.add_vertex(());
        let d = graph.add_vertex(());
        let e = graph.add_vertex(());
        let f = graph.add_vertex(());
        let g = graph.add_vertex(());

        graph.add_edge(a, b).unwrap();
        graph.add_edge(b, c).unwrap();
        graph.add_edge(c, d).unwrap();
        graph.add_edge(d, e).unwrap();
        graph.add_edge(e, f).unwrap();

        graph.add_edge(e, g).unwrap();
        graph.add_edge(g, c).unwrap();

        assert_eq!(graph.num_vertices(), 7);
        assert_eq!(graph.num_edges(), 7);
        assert_eq!(graph.num_roots(), 1);
        assert_eq!(graph.num_leaves(), 1);

        assert_eq!(graph.transitive_reduction().err().unwrap(), TransitiveReductionError::CyclicGraph);
    }
}
