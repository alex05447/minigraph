use {
    crate::*,
    minihandle::*,
    smallvec::SmallVec,
    std::{
        iter::{ExactSizeIterator, Iterator},
        sync::atomic::{AtomicUsize, Ordering},
    },
};

/// A [`TaskVertex`] which had all its dependencies [`completed`](TaskVertex::ready)
/// and may be now used to [`access`](ReadyTaskVertex::get_dependent) the dependent vertices.
pub struct ReadyTaskVertex<'a, VID: VertexID, T> {
    vertices: &'a [TaskVertexInner<VID, T>],
    index: usize,
}

impl<'a, VID: VertexID, T> ReadyTaskVertex<'a, VID, T> {
    /// Accesses the task vertex payload.
    pub fn vertex(&self) -> &T {
        &self.inner().vertex
    }

    /// Returns the number of [`vertices`](TaskVertex) dependant on this vertex.
    pub fn num_dependents(&self) -> usize {
        self.inner().dependents.len()
    }

    /// If dependent vertex `index` is valid (i.e. in range `0` .. [`num_dependents`](ReadyTaskVertex::num_dependents)),
    /// returns the dependent [`TaskVertex`] at this `index`.
    pub fn get_dependent(&self, index: usize) -> Option<TaskVertex<'a, VID, T>> {
        Some(TaskVertex {
            vertices: self.vertices,
            index: <VID as ToUsize>::to_usize(*self.inner().dependents.get(index)?),
        })
    }

    /// Returns the dependent [`TaskVertex`] at `index`.
    ///
    /// # Safety
    ///
    /// The caller guarantees the dependent vertex `index` is valid (i.e. in range `0` .. [`num_dependents`](ReadyTaskVertex::num_dependents)).
    pub unsafe fn get_dependent_unchecked(&self, index: usize) -> TaskVertex<'a, VID, T> {
        debug_assert!(index < self.inner().dependents.len());
        TaskVertex {
            vertices: self.vertices,
            index: <VID as ToUsize>::to_usize(*self.inner().dependents.get_unchecked(index)),
        }
    }

    /// Returns an iterator over all dependent [`vertices`](TaskVertex) dependent on this [`ReadyTaskVertex`] in unspecified order.
    pub fn dependents(&self) -> impl ExactSizeIterator<Item = TaskVertex<'_, VID, T>> {
        let inner = self.inner();
        (0..inner.dependents.len()).map(move |index| TaskVertex {
            vertices: self.vertices,
            index,
        })
    }

    fn inner(&self) -> &TaskVertexInner<VID, T> {
        // Must succeed - all vertex indices are valid.
        debug_assert!(self.index < self.vertices.len(), "invalid vertex index");
        unsafe { self.vertices.get_unchecked(self.index) }
    }
}

/// Task graph vertex payload wrapper which tracks the vertex's dependency completion
/// during [`TaskGraph`] (parallel) execution.
pub struct TaskVertex<'a, VID: VertexID, T> {
    vertices: &'a [TaskVertexInner<VID, T>],
    index: usize,
}

impl<'a, VID: VertexID, T> TaskVertex<'a, VID, T> {
    /// Increments the completed dependency counter
    /// and returns `Ok` if all dependencies are now satisfied,
    /// or `Err` if at least one dependency is still incomplete.
    ///
    /// On success the returned [`ReadyTaskVertex`] may be used to access the dependent vertices.
    pub fn ready(self) -> Result<ReadyTaskVertex<'a, VID, T>, Self> {
        let vertex = self.inner();
        (vertex.complete_dependency() >= vertex.num_dependencies)
            .then(|| ReadyTaskVertex {
                vertices: self.vertices,
                index: self.index,
            })
            .ok_or(self)
    }

    fn inner(&self) -> &TaskVertexInner<VID, T> {
        // Must succeed - all vertex indices are valid.
        debug_assert!(self.index < self.vertices.len(), "invalid vertex index");
        unsafe { self.vertices.get_unchecked(self.index) }
    }
}

/*
/// A [`TaskVertexMut`] which had all its dependencies [`completed`](TaskVertexMut::ready)
/// and may be now used to [`access`](ReadyTaskVertexMut::get_dependent) the dependent vertices.
pub struct ReadyTaskVertexMut<'a, VID: VertexID, T> {
    vertices: &'a mut [TaskVertexInner<VID, T>],
    index: usize,
}

impl<'a, VID: VertexID, T> ReadyTaskVertexMut<'a, VID, T> {
    /// Accesses the task vertex payload.
    pub fn vertex(&self) -> &T {
        &self.inner().vertex
    }

    /// Accesses the task vertex payload.
    pub fn vertex_mut(&'a mut self) -> &'a mut T {
        &mut self.inner_mut().vertex
    }

    /// Returns the number of [`vertices`](TaskVertexMut) dependant on this vertex.
    pub fn num_dependents(&self) -> usize {
        self.inner().dependents.len()
    }

    /// If dependent vertex `index` is valid (i.e. in range `0` .. [`num_dependents`](TaskVertexMut::num_dependents)),
    /// returns the dependent [`TaskVertex`] at this `index`.
    pub fn get_dependent(&'a self, index: usize) -> Option<TaskVertex<'a, VID, T>> {
        Some(TaskVertex {
            vertices: self.vertices,
            index: <VID as ToUsize>::to_usize(*self.inner().dependents.get(index)?),
        })
    }

    /// Returns the dependent [`TaskVertex`] at `index`.
    ///
    /// # Safety
    ///
    /// The caller guarantees the dependent vertex `index` is valid (i.e. in range `0` .. [`num_dependents`](TaskVertexMut::num_dependents)).
    pub unsafe fn get_dependent_unchecked(&'a self, index: usize) -> TaskVertex<'a, VID, T> {
        debug_assert!(index < self.inner().dependents.len());
        TaskVertex {
            vertices: self.vertices,
            index: <VID as ToUsize>::to_usize(*self.inner().dependents.get_unchecked(index)),
        }
    }

    /// If dependent vertex `index` is valid (i.e. in range `0` .. [`num_dependents`](TaskVertexMut::num_dependents)),
    /// returns the (mutable) dependent [`TaskVertexMut`] at this `index`.
    pub fn get_dependent_mut(&'a mut self, index: usize) -> Option<TaskVertexMut<'a, VID, T>> {
        let index = <VID as ToUsize>::to_usize(*self.inner().dependents.get(index)?);
        Some(TaskVertexMut {
            vertices: self.vertices,
            index,
        })
    }

    /// Returns the (mutable) dependent [`TaskVertexMut`] at `index`.
    ///
    /// # Safety
    ///
    /// The caller guarantees the dependent vertex `index` is valid (i.e. in range `0` .. [`num_dependents`](TaskVertexMut::num_dependents)).
    pub unsafe fn get_dependent_unchecked_mut(
        &'a mut self,
        index: usize,
    ) -> TaskVertexMut<'a, VID, T> {
        debug_assert!(index < self.inner().dependents.len());
        let index = <VID as ToUsize>::to_usize(*self.inner().dependents.get_unchecked(index));
        TaskVertexMut {
            vertices: self.vertices,
            index,
        }
    }

    /// Returns an iterator over all dependent [`vertices`](TaskVertex) dependent on this [`TaskVertex`] in unspecified order.
    pub fn dependents(&self) -> impl ExactSizeIterator<Item = TaskVertex<'_, VID, T>> {
        let inner = self.inner();
        (0..inner.dependents.len()).map(move |index| TaskVertex {
            vertices: self.vertices,
            index,
        })
    }

    fn inner(&self) -> &TaskVertexInner<VID, T> {
        // Must succeed - all vertex indices are valid.
        debug_assert!(self.index < self.vertices.len(), "invalid vertex index");
        unsafe { self.vertices.get_unchecked(self.index) }
    }

    fn inner_mut(&mut self) -> &mut TaskVertexInner<VID, T> {
        // Must succeed - all vertex indices are valid.
        debug_assert!(self.index < self.vertices.len(), "invalid vertex index");
        unsafe { self.vertices.get_unchecked_mut(self.index) }
    }
}

/// Task graph vertex (mutable) payload wrapper which tracks the vertex's dependency completion
/// during [`TaskGraph`] (parallel) execution.
pub struct TaskVertexMut<'a, VID: VertexID, T> {
    vertices: &'a mut [TaskVertexInner<VID, T>],
    index: usize,
}

impl<'a, VID: VertexID, T> TaskVertexMut<'a, VID, T> {
    /// Increments the completed dependency counter
    /// and returns `Ok` if all dependencies are now satisfied,
    /// or `Err` if at least one dependency is still incomplete.
    ///
    /// On success the returned [`ReadyTaskVertex`] may be used to access the dependent vertices.
    pub fn ready(self) -> Result<ReadyTaskVertexMut<'a, VID, T>, Self> {
        let vertex = self.inner();
        if vertex.complete_dependency() >= vertex.num_dependencies {
            Ok(ReadyTaskVertexMut {
                vertices: self.vertices,
                index: self.index,
            })
        } else {
            Err(self)
        }
    }

    fn inner(&self) -> &TaskVertexInner<VID, T> {
        // Must succeed - all vertex indices are valid.
        debug_assert!(self.index < self.vertices.len(), "invalid vertex index");
        unsafe { self.vertices.get_unchecked(self.index) }
    }
}
*/

pub(crate) struct TaskVertexInner<VID: VertexID, T> {
    /// Vertex payload.
    vertex: T,
    /// Constant number of dependencies of this vertex (inbound edges).
    num_dependencies: usize,
    /// Current number of completed dependencies of this vertex (inbound edges).
    completed_dependencies: AtomicUsize,
    /// Indices of all vertices dependent on this vertex (outbound edges) in the task graph `vertices` array.
    dependents: SmallVec<[VID; NUM_EDGES]>,
}

impl<VID: VertexID, T> TaskVertexInner<VID, T> {
    pub(crate) fn new(
        vertex: T,
        num_dependencies: usize,
        dependents: SmallVec<[VID; NUM_EDGES]>,
    ) -> Self {
        Self {
            vertex,
            num_dependencies,
            completed_dependencies: AtomicUsize::new(0),
            dependents,
        }
    }

    /// Increments the completed dependency counter, returns the incremented counter value.
    fn complete_dependency(&self) -> usize {
        self.completed_dependencies.fetch_add(1, Ordering::SeqCst) + 1
    }

    fn reset(&self) {
        self.completed_dependencies.store(0, Ordering::SeqCst);
    }
}

/// Simplified immutable unidirectional representation of the (acyclic) [`Graph`]
/// used as the dependency (or more precisely, execution) graph for task scheduling / (parallel) execution.
///
/// Edges represent vertex dependencies, and directions correspond to vertex execution order.
///
/// Example use:
///
/// ```
/// # use minigraph::*;
///
/// fn process_vertex<VID: VertexID, T>(v: TaskVertex<'_, VID, T>) {
///     // Increments the dependency counter, returns `Err` if not ready.
///     if let Ok(v) = v.ready() {
///         // All dependencies are complete, we may proceed.
///
///         let payload = v.vertex();
///         // ... do something with the payload ...
///
///         // Process all dependants recursively.
///         for v in v.dependents() {
///             process_vertex(v);
///         }
///     }
/// }
///
/// fn process_graph<VID: VertexID, T>(graph: &mut TaskGraph<VID, T>) {
///     // Start processing the graph with the root vertices;
///     // process each dependent vertex only after all its dependencies have been satisfied.
///     for root in graph.roots() {
///         process_vertex(root);
///     }
///
///     // Reset all dependency counters; the graph may be executed again.
///     graph.reset();
/// }
/// ```
pub struct TaskGraph<VID: VertexID, T> {
    /// Indices of vertices in `vertices` array with no dependencies.
    roots: Vec<VID>,
    vertices: Vec<TaskVertexInner<VID, T>>,
}

impl<VID: VertexID, T> TaskGraph<VID, T> {
    /// Resets the completed dependency counters on all [`vertices`](TaskVertex),
    /// allowing the [`TaskGraph`] to be executed again.
    pub fn reset(&self) {
        self.vertices.iter().for_each(TaskVertexInner::reset);
    }

    /// Returns the current number of root [`vertices`](TaskVertex) (i.e. ones with no dependencies / inbound edges) in the [`TaskGraph`].
    pub fn num_roots(&self) -> usize {
        self.roots.len()
    }

    /// If root vertex `index` is valid (i.e. in range [`0` .. [`num_roots`](TaskGraph::num_roots)]),
    /// returns the root (i.e. ones with no dependencies / inbound edges) [`TaskVertex`] at `index`.
    pub fn root(&self, index: usize) -> Option<TaskVertex<'_, VID, T>> {
        let vertex_index = <VID as ToUsize>::to_usize(*self.roots.get(index)?);
        // Must succeed - all root vertex indices are valid.
        debug_assert!(
            vertex_index < self.vertices.len(),
            "invalid root vertex index"
        );
        Some(TaskVertex {
            vertices: &self.vertices,
            index: vertex_index,
        })
    }

    /// Returns the root (i.e. ones with no dependencies / inbound edges) [`TaskVertex`] at `index`.
    ///
    /// # Safety
    ///
    /// The caller guarantees the root vertex `index` is valid (i.e. in range [`0` .. [`num_roots`](TaskGraph::num_roots)]).
    pub unsafe fn root_unchecked(&self, index: usize) -> TaskVertex<'_, VID, T> {
        let vertex_index = <VID as ToUsize>::to_usize(*self.roots.get_unchecked(index));
        // Must succeed - all root vertex indices are valid.
        debug_assert!(
            vertex_index < self.vertices.len(),
            "invalid root vertex index"
        );
        TaskVertex {
            vertices: &self.vertices,
            index: vertex_index,
        }
    }

    /*
    /// If root vertex `index` is valid (i.e. in range [`0` .. [`num_roots`](TaskGraph::num_roots)]),
    /// returns the root (i.e. ones with no dependencies / inbound edges) [`TaskVertexMut`] at `index`.
    pub fn root_mut(&mut self, index: usize) -> Option<TaskVertexMut<'_, VID, T>> {
        let vertex_index = <VID as ToUsize>::to_usize(*self.roots.get(index)?);
        // Must succeed - all root vertex indices are valid.
        debug_assert!(
            vertex_index < self.vertices.len(),
            "invalid root vertex index"
        );
        Some(TaskVertexMut {
            vertices: &mut self.vertices,
            index: vertex_index,
        })
    }

    /// Returns the root (i.e. ones with no dependencies / inbound edges) [`TaskVertexMut`] at `index`.
    ///
    /// # Safety
    ///
    /// The caller guarantees the root vertex `index` is valid (i.e. in range [`0` .. [`num_roots`](TaskGraph::num_roots)]).
    pub unsafe fn root_unchecked_mut(&mut self, index: usize) -> TaskVertexMut<'_, VID, T> {
        let vertex_index = <VID as ToUsize>::to_usize(*self.roots.get_unchecked(index));
        // Must succeed - all root vertex indices are valid.
        debug_assert!(
            vertex_index < self.vertices.len(),
            "invalid root vertex index"
        );
        TaskVertexMut {
            vertices: &mut self.vertices,
            index: vertex_index,
        }
    }
    */

    /// Returns an iterator over all root (i.e. ones with no dependencies / inbound edges) [`vertices`](TaskVertex) in the [`TaskGraph`] in unspecified order.
    pub fn roots(&self) -> impl ExactSizeIterator<Item = TaskVertex<'_, VID, T>> {
        (0..self.vertices.len()).map(move |index| TaskVertex {
            vertices: &self.vertices,
            index,
        })
    }
}

impl<VID: VertexID, T: Clone> TaskGraph<VID, T> {
    pub(crate) fn new(vertices: Vec<TaskVertexInner<VID, T>>, roots: Vec<VID>) -> Self {
        // Sanity check - make sure all root indices are in bounds.
        debug_assert!(
            roots
                .iter()
                .all(|&root| <VID as ToUsize>::to_usize(root) < vertices.len()),
            "invalid root index"
        );
        // Sanity check - make sure all dependent indices are in bounds.
        debug_assert!(vertices
            .iter()
            .flat_map(|vertex| vertex.dependents.iter())
            .all(|&index| <VID as ToUsize>::to_usize(index) < vertices.len()));

        Self { vertices, roots }
    }
}
