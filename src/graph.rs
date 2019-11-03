use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use super::task_graph::TaskGraph;

use minihandle::IndexManager;
use num_traits::{PrimInt, Unsigned};

pub trait VertexID: PrimInt + Unsigned + Hash {}

impl<T: PrimInt + Unsigned + Hash> VertexID for T {}

pub struct Graph<VID: VertexID, T> {
    vertex_ids: IndexManager<VID>,

    vertices: HashMap<VID, T>,

    roots: HashSet<VID>,
    leaves: HashSet<VID>,

    in_edges: HashMap<VID, HashSet<VID>>,
    out_edges: HashMap<VID, HashSet<VID>>,
}

#[derive(Debug, PartialEq)]
pub enum AccessVertexError {
    InvalidVertex,
}

#[derive(Debug, PartialEq)]
pub enum AddEdgeStatus {
    Added,
    AlreadyExists,
}

#[derive(Debug, PartialEq)]
pub enum RemoveEdgeStatus {
    Removed,
    DoesNotExist,
}

#[derive(Debug, PartialEq)]
pub enum EdgeAccessError {
    InvalidFromVertex,
    InvalidToVertex,
}

impl<VID: VertexID, T> Graph<VID, T> {
    pub fn new() -> Self {
        Self {
            vertex_ids: IndexManager::new(),
            vertices: HashMap::new(),
            roots: HashSet::new(),
            leaves: HashSet::new(),
            in_edges: HashMap::new(),
            out_edges: HashMap::new(),
        }
    }

    pub fn add_vertex(&mut self, vertex: T) -> VID {
        let vertex_id = self.vertex_ids.create();

        let prev = self.vertices.insert(vertex_id, vertex);
        assert!(prev.is_none(), "Duplicate vertex ID in the graph.");

        self.roots.insert(vertex_id);
        self.leaves.insert(vertex_id);

        vertex_id
    }

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
            Err(AccessVertexError::InvalidVertex)
        }
    }

    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    pub fn vertex(&self, vertex_id: VID) -> Option<&T> {
        self.vertices.get(&vertex_id)
    }

    pub fn vertex_mut(&mut self, vertex_id: VID) -> Option<&mut T> {
        self.vertices.get_mut(&vertex_id)
    }

    pub fn vertices(&self) -> VertexIterator<'_, VID, T> {
        VertexIterator::new(self.vertices.iter())
    }

    pub fn add_edge(&mut self, from: VID, to: VID) -> Result<AddEdgeStatus, EdgeAccessError> {
        if !self.vertices.contains_key(&from) {
            return Err(EdgeAccessError::InvalidFromVertex);
        }

        if !self.vertices.contains_key(&to) {
            return Err(EdgeAccessError::InvalidToVertex);
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
            Ok(AddEdgeStatus::Added)
        } else {
            Ok(AddEdgeStatus::AlreadyExists)
        }
    }

    pub fn remove_edge(&mut self, from: VID, to: VID) -> Result<RemoveEdgeStatus, EdgeAccessError> {
        if !self.vertices.contains_key(&from) {
            return Err(EdgeAccessError::InvalidFromVertex);
        }

        if !self.vertices.contains_key(&to) {
            return Err(EdgeAccessError::InvalidToVertex);
        }

        if let Some(in_edges) = self.in_edges.get_mut(&to) {
            if !in_edges.remove(&from) {
                return Ok(RemoveEdgeStatus::DoesNotExist);
            }

            if in_edges.is_empty() {
                self.in_edges.remove(&to);
            }
        } else {
            return Ok(RemoveEdgeStatus::DoesNotExist);
        }

        let out_edges = self
            .out_edges
            .get_mut(&from)
            .expect("In / out edge mismatch.");

        out_edges.take(&to).expect("In / out edge mismatch.");

        if out_edges.is_empty() {
            self.out_edges.remove(&from);
        }

        if self.num_in_neighbors(to).unwrap() == 0 {
            let was_not_a_root = self.roots.insert(to);
            assert!(was_not_a_root, "Root vertex had inbound edges.")
        }

        if self.num_out_neighbors(from).unwrap() == 0 {
            let was_not_a_leaf = self.leaves.insert(from);
            assert!(was_not_a_leaf, "Leaf vertex had outbound edges.")
        }

        Ok(RemoveEdgeStatus::Removed)
    }

    pub fn num_edges(&self) -> usize {
        let num_edges = self.in_edges.len();
        debug_assert_eq!(num_edges, self.out_edges.len(), "In / out edge mismatch.");
        num_edges
    }

    pub fn num_roots(&self) -> usize {
        self.roots.len()
    }

    pub fn roots(&self) -> VertexIDIterator<'_, VID> {
        VertexIDIterator::new(Some(self.roots.iter()))
    }

    pub fn num_leaves(&self) -> usize {
        self.leaves.len()
    }

    pub fn leaves(&self) -> VertexIDIterator<'_, VID> {
        VertexIDIterator::new(Some(self.leaves.iter()))
    }

    pub fn num_in_neighbors(&self, vertex_id: VID) -> Result<usize, AccessVertexError> {
        if !self.vertices.contains_key(&vertex_id) {
            Err(AccessVertexError::InvalidVertex)
        } else {
            Ok(self
                .in_edges
                .get(&vertex_id)
                .map_or(0, |in_edges| in_edges.len()))
        }
    }

    pub fn in_neighbors(
        &self,
        vertex_id: VID,
    ) -> Result<VertexIDIterator<'_, VID>, AccessVertexError> {
        if !self.vertices.contains_key(&vertex_id) {
            Err(AccessVertexError::InvalidVertex)
        } else {
            Ok(VertexIDIterator::new(
                self.in_edges
                    .get(&vertex_id)
                    .map_or(None, |in_edges| Some(in_edges.iter())),
            ))
        }
    }

    pub fn num_out_neighbors(&self, vertex_id: VID) -> Result<usize, AccessVertexError> {
        if !self.vertices.contains_key(&vertex_id) {
            Err(AccessVertexError::InvalidVertex)
        } else {
            Ok(self
                .out_edges
                .get(&vertex_id)
                .map_or(0, |out_edges| out_edges.len()))
        }
    }

    pub fn out_neighbors(
        &self,
        vertex_id: VID,
    ) -> Result<VertexIDIterator<'_, VID>, AccessVertexError> {
        if !self.vertices.contains_key(&vertex_id) {
            Err(AccessVertexError::InvalidVertex)
        } else {
            Ok(VertexIDIterator::new(
                self.out_edges
                    .get(&vertex_id)
                    .map_or(None, |out_edges| Some(out_edges.iter())),
            ))
        }
    }
}

impl<VID: VertexID, T: Clone> Graph<VID, T> {
    pub fn task_graph(&self) -> TaskGraph<VID, T> {
        TaskGraph::new(self, &self.vertices, &self.roots, &self.out_edges)
    }
}

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
}
