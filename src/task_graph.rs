use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::iter::{Iterator, ExactSizeIterator};

use super::graph::{AccessVertexError, Graph, VertexID};

/// Task graph vertex payload wrapper which tracks the vertice's dependency completion
/// during task graph (parallel) execution.
pub struct TaskVertex<T> {
    /// Vertex payload.
    vertex: T,
    /// Constant number of inbound edges.
    num_dependencies: usize,
    /// Current number of completed dependencies (inbound edges).
    completed_dependencies: AtomicUsize,
}

impl<T> TaskVertex<T> {
    /// Access the vertex payload.
    pub fn vertex(&self) -> &T {
        &self.vertex
    }

    /// Increments the completed dependency counter.
    /// Returns `true` if all dependencies are now satisfied.
    pub fn is_ready(&self) -> bool {
        self.complete_dependency() >= self.num_dependencies
    }

    /// Returns the incremented counter value.
    fn complete_dependency(&self) -> usize {
        self.completed_dependencies.fetch_add(1, Ordering::SeqCst) + 1
    }

    fn new(vertex: T, num_dependencies: usize) -> Self {
        Self {
            vertex,
            num_dependencies,
            completed_dependencies: AtomicUsize::new(0),
        }
    }

    fn reset(&self) {
        self.completed_dependencies.store(0, Ordering::SeqCst);
    }
}

/// Simplified unidirectional representation of the (acyclic) [`Graph`]
/// used as the dependency/execution graph for task scheduling / (parallel) execution.
///
/// Edges represent vertex dependencies, directions correspond to vertex execution order.
///
/// [`Graph`]: struct.Graph.html
pub struct TaskGraph<VID: VertexID, T> {
    vertices: HashMap<VID, TaskVertex<T>>,
    roots: Vec<VID>,
    out_edges: HashMap<VID, Vec<VID>>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GetRootError {
    /// The root vertex index was out of bounds.
    RootIndexOutOfBounds,
}

impl Error for GetRootError {}

impl Display for GetRootError {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        use GetRootError::*;

        match self {
            RootIndexOutOfBounds => write!(f, "root vertex index was out of bounds"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GetDependencyError {
    /// The vertex ID was invalid.
    InvalidVertexID,
    /// The dependency vertex index was out of bounds.
    DependencyIndexOutOfBounds,
}

impl Error for GetDependencyError {}

impl Display for GetDependencyError {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        use GetDependencyError::*;

        match self {
            InvalidVertexID => write!(f, "vertex ID was invalid"),
            DependencyIndexOutOfBounds => write!(f, "dependency vertex index was out of bounds"),
        }
    }
}

impl<VID: VertexID, T> TaskGraph<VID, T> {
    /// Reset the completed dependency counters on all vertices,
    /// allowing the graph to be executed again.
    pub fn reset(&self) {
        for (_, vertex) in self.vertices.iter() {
            vertex.reset()
        }
    }

    /// Returns the current number of root vertices (with no inbound edges) in the graph.
    pub fn num_roots(&self) -> usize {
        self.roots.len()
    }

    /// If `root_index` is in range `0 .. num_roots()`, returns the root [`TaskVertex`] at `root_index`.
    ///
    /// [`TaskVertex`]: struct.TaskVertex.html
    pub fn get_root(&self, root_index: usize) -> Result<(&VID, &TaskVertex<T>), GetRootError> {
        let vertex_id = self
            .roots
            .get(root_index)
            .ok_or(GetRootError::RootIndexOutOfBounds)?;
        Ok((
            vertex_id,
            self.vertices
                .get(&vertex_id)
                .expect("Invalid root vertex ID."),
        ))
    }

    /// If `root_index` is in range `0 .. num_roots()`, returns the root [`TaskVertex`] at `root_index`.
    ///
    /// Otherwise panics.
    ///
    /// [`TaskVertex`]: struct.TaskVertex.html
    pub unsafe fn get_root_unchecked(&self, root_index: usize) -> (&VID, &TaskVertex<T>) {
        let vertex_id = self.roots.get_unchecked(root_index);
        (
            vertex_id,
            self.vertices
                .get(&vertex_id)
                .expect("Invalid root vertex ID."),
        )
    }

    /// Returns an iterator over all task graph roots (vertices with no dependencies).
    pub fn roots(&self) -> impl Iterator<Item = (&'_ VID, &'_ TaskVertex<T>)> {
        TaskVertexIterator::new(&self.vertices, Some(self.roots.iter()))
    }

    /// If `vertex_id` is valid, returns the number of vertices dependant on it.
    pub fn num_dependencies(&self, vertex_id: VID) -> Result<usize, AccessVertexError> {
        if !self.vertices.contains_key(&vertex_id) {
            Err(AccessVertexError::InvalidVertexID)
        } else {
            Ok(self
                .out_edges
                .get(&vertex_id)
                .map_or(0, |out_edges| out_edges.len()))
        }
    }

    /// If `vertex_id` is valid and `dependency_index` is in range `0 .. num_dependencies(vertex_id)`,
    /// returns the dependency [`TaskVertex`] at `dependency_index`.
    ///
    /// [`TaskVertex`]: struct.TaskVertex.html
    pub fn get_dependency(
        &self,
        vertex_id: VID,
        dependency_index: usize,
    ) -> Result<(&VID, &TaskVertex<T>), GetDependencyError> {
        if !self.vertices.contains_key(&vertex_id) {
            Err(GetDependencyError::InvalidVertexID)
        } else {
            let dependencies = self
                .out_edges
                .get(&vertex_id)
                .ok_or(GetDependencyError::DependencyIndexOutOfBounds)?;
            let dependency_vertex_id = dependencies
                .get(dependency_index)
                .ok_or(GetDependencyError::DependencyIndexOutOfBounds)?;
            Ok((
                dependency_vertex_id,
                self.vertices
                    .get(&dependency_vertex_id)
                    .expect("Invalid dependency vertex ID."),
            ))
        }
    }

    /// If `vertex_id` is valid and `dependency_index` is in range `0 .. num_dependencies(vertex_id)`,
    /// returns the dependency [`TaskVertex`] at `dependency_index`.
    ///
    /// Otherwise panics.
    ///
    /// [`TaskVertex`]: struct.TaskVertex.html
    pub unsafe fn get_dependency_unchecked(
        &self,
        vertex_id: VID,
        dependency_index: usize,
    ) -> (&VID, &TaskVertex<T>) {
        let dependencies = self.out_edges.get(&vertex_id).expect("Invalid vertex ID.");
        let dependency_vertex_id = dependencies.get_unchecked(dependency_index);
        (
            dependency_vertex_id,
            self.vertices
                .get(&dependency_vertex_id)
                .expect("Invalid dependency vertex ID."),
        )
    }

    /// If `vertex_id` is valid, returns an iterator over all vertices dependant on it.
    pub fn dependencies(
        &self,
        vertex_id: VID,
    ) -> Result<impl Iterator<Item = (&'_ VID, &'_ TaskVertex<T>)>, AccessVertexError> {
        if !self.vertices.contains_key(&vertex_id) {
            Err(AccessVertexError::InvalidVertexID)
        } else {
            Ok(TaskVertexIterator::new(
                &self.vertices,
                self.out_edges
                    .get(&vertex_id)
                    .map_or(None, |out_edges| Some(out_edges.iter())),
            ))
        }
    }
}

impl<VID: VertexID, T: Clone> TaskGraph<VID, T> {
    pub(super) fn new(
        graph: &Graph<VID, T>,
        out_edges: &HashMap<VID, HashSet<VID>>,
    ) -> Self {
        let vertices = graph.vertices()
            .map(|(vertex_id, vertex)| {
                (
                    *vertex_id,
                    TaskVertex::new(vertex.clone(), graph.num_in_neighbors(*vertex_id).unwrap()),
                )
            })
            .collect();

        let roots = graph.roots().map(|vertex_id| *vertex_id).collect();

        let out_edges = out_edges
            .iter()
            .map(|(vertex_id, vertex_ids)| {
                (
                    *vertex_id,
                    vertex_ids.iter().map(|vertex_id| *vertex_id).collect(),
                )
            })
            .collect();

        Self {
            vertices,
            roots,
            out_edges,
        }
    }
}

/// Iterates over [`TaskVertex`]'s, returning their vertex ID and payload.
///
/// [`TaskVertex`]: struct.TaskVertex.html
struct TaskVertexIterator<'a, VID: VertexID, T> {
    vertices: &'a HashMap<VID, TaskVertex<T>>,
    vertex_ids: Option<std::slice::Iter<'a, VID>>,
}

impl<'a, VID: VertexID, T> TaskVertexIterator<'a, VID, T> {
    fn new(
        vertices: &'a HashMap<VID, TaskVertex<T>>,
        vertex_ids: Option<std::slice::Iter<'a, VID>>,
    ) -> Self {
        Self {
            vertices,
            vertex_ids,
        }
    }
}

impl<'a, VID: VertexID, T> Iterator for TaskVertexIterator<'a, VID, T> {
    type Item = (&'a VID, &'a TaskVertex<T>);

    fn next(&mut self) -> Option<Self::Item> {
        self.vertex_ids
            .as_mut()
            .map(|vertex_ids| vertex_ids.next())
            .map_or(None, |vertex_id| {
                vertex_id.map(|vertex_id| (vertex_id, self.vertices.get(vertex_id).unwrap()))
            })
    }
}

impl<'a, VID: VertexID, T> ExactSizeIterator for TaskVertexIterator<'a, VID, T> {
    fn len(&self) -> usize {
        self.vertex_ids
            .as_ref()
            .map_or(0, |vertex_ids| vertex_ids.len())
    }
}
