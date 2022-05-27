use {
    crate::*,
    minihandle::*,
    std::{collections::HashSet, hash::Hash, iter::Iterator},
};

/// Trait for the [`Graph`] unique vertex identifier - a primitive unsigned integer convertible to `usize`.
pub trait VertexID: Index + Hash {}

impl<T: Index + Hash> VertexID for T {}

/// Number of in- and outbound edges (i.e. their [`VertexID`]'s) per vertex we store inline
/// (using [`smallvec::SmallVec`]).
///
/// TODO: this value is completely arbitrary. Expose as a const parameter if/when necessary.
pub(crate) const NUM_EDGES: usize = 4;

/// Storage for the vertex's in- and outbound edges (i.e. their [`VertexID`]'s).
type Edges<VID> = SmallSet<VID, NUM_EDGES>;

/// Graph vertex state as contained in the [`Graph`].
///
/// Its payload plus in- and outbound edges (i.e. their [`VertexID`]'s).
#[derive(Clone)]
pub(crate) struct Vertex<VID: VertexID, T> {
    /// Needed to map back from the vertex to its ID,
    /// because the map-like container we use for vertices ([`IndexArray`])
    /// does not provide a "pairs" iterator over keys/values, only the values iterator.
    ///
    /// Used by [`vertices`](Graph::vertices), as well as for transitive reduction.
    id: VID,
    /// Arbitrary vertex payload.
    payload: T,
    /// Inbound edges for this vertex (i.e. [`ID`]'s of their vertices).
    in_edges: Edges<VID>,
    /// Outbound edges for this vertex (i.e. [`ID`]'s of their vertices).
    out_edges: Edges<VID>,
}

impl<VID: VertexID, T> Vertex<VID, T> {
    /// Creates a new vertex with given `payload` and an uninitialized `id`.
    fn new(payload: T) -> Self {
        Self {
            id: VID::zero(),
            payload,
            in_edges: Edges::new(),
            out_edges: Edges::new(),
        }
    }
}

impl<VID: VertexID, T: Clone> Vertex<VID, T> {
    /// Clones the graph vertex into a [`TaskVertexInner`].
    pub(crate) fn task_vertex(&self, graph: &Graph<VID, T>) -> TaskVertexInner<VID, T> {
        TaskVertexInner::new(
            self.payload.clone(),
            self.in_edges.len(),
            self.out_edges
                .inner()
                .iter()
                // Flatten out edge vertex IDs into object (vertex) indices in the `vertices` inner array,
                // removing the indirection.
                // See `Graph::task_graph`.
                // Must succeed, all out edges are valid vertex IDs.
                .map(|&out_edge| unsafe { graph.vertices.object_index_unchecked(out_edge) })
                .collect(),
        )
    }
}

type Vertices<VID, T> = IndexArray<VID, Vertex<VID, T>>;

/// Represents a digraph.
///
/// `VID` is a [`primitive unsigned integer type`](Index), used to uniquely identify graph vertices.
/// `T` is the type of arbitrary vertex payload.
#[derive(Clone)]
pub struct Graph<VID: VertexID, T> {
    /// Allocates vertex IDs and stores payloads for all vertices in the graph.
    vertices: Vertices<VID, T>,
    /// Stores IDs of all root vertices, i.e. vertices with no inbound edges.
    roots: HashSet<VID>,
    /// Stores IDs of all leaf vertices, i.e. vertices with no outbound edges.
    leaves: HashSet<VID>,
    /// Number of edges in the graph.
    num_edges: usize,
}

impl<VID: VertexID, T> Graph<VID, T> {
    /// Creates a new empty graph.
    pub fn new() -> Self {
        Self {
            vertices: Vertices::new(),
            roots: HashSet::new(),
            leaves: HashSet::new(),
            num_edges: 0,
        }
    }

    /// Adds a new unconnected vertex with given payload to the graph,
    /// returning its unique [`vertex identifier`](VertexID) in this [`Graph`].
    ///
    /// Returned [`VertexID`] can be used to access / modify the vertex and its edges.
    pub fn add_vertex(&mut self, vertex: T) -> VID {
        // Allocate a new vertex and its ID.
        let (id, vertex) = self.vertices.insert_entry(Vertex::new(vertex));
        // Must update the vertex ID.
        vertex.id = id;

        // The new vertex is unconnected and is simultaneously a root and a leaf.
        let _true = self.roots.insert(id);
        debug_assert!(_true, "duplicate root vertex ID in the graph");
        let _true = self.leaves.insert(id);
        debug_assert!(_true, "duplicate leaf vertex ID in the graph");

        id
    }

    /// If vertex `id` is valid, removes the vertex payload from the graph and returns it.
    ///
    /// Removes any edges to / from the removed vertex.
    pub fn remove_vertex(&mut self, id: VID) -> Option<T> {
        // Remove the vertex payload.
        let vertex = self.vertices.remove(id)?;

        // Remove the inbound edges, if any.
        for &in_neighbor in vertex.in_edges.iter() {
            Self::try_remove_edge(
                &mut self.vertices,
                &mut self.roots,
                &mut self.leaves,
                &mut self.num_edges,
                in_neighbor,
                vertex.id,
            );
        }

        // Remove the outbound edges, if any.
        for &out_neighbor in vertex.out_edges.iter() {
            Self::try_remove_edge(
                &mut self.vertices,
                &mut self.roots,
                &mut self.leaves,
                &mut self.num_edges,
                vertex.id,
                out_neighbor,
            );
        }

        Some(vertex.payload)
    }

    /// Returns the current number of vertices in the graph.
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// If vertex `id` is valid, returns a reference to the vertex payload.
    pub fn vertex(&self, id: VID) -> Option<&T> {
        self.vertices.get(id).map(|vertex| &vertex.payload)
    }

    /// If vertex `id` is valid, returns the mutable reference to the vertex payload.
    pub fn vertex_mut(&mut self, id: VID) -> Option<&mut T> {
        self.vertices.get_mut(id).map(|vertex| &mut vertex.payload)
    }

    /// Returns an iterator over all vertices in the graph, in no particular order.
    pub fn vertices(&self) -> impl Iterator<Item = (VID, &T)> {
        self.vertices_inner()
            .map(|vertex| (vertex.id, &vertex.payload))
    }

    /// If vertex IDs `from` and `to` are valid, tries to add a directed edge between them.
    ///
    /// Does nothing if the edge already exists.
    pub fn add_edge(&mut self, from: VID, to: VID) -> Result<AddEdgeStatus, AccessEdgeError> {
        use AccessEdgeError::*;

        if !self.vertices.is_valid(from) {
            return Err(InvalidFromVertex);
        }

        if !self.vertices.is_valid(to) {
            return Err(InvalidToVertex);
        }

        if from == to {
            return Err(LoopEdge);
        }

        // Must succeed, checked above.
        let from_vertex = unsafe { self.vertices.get_unchecked_mut(from) };

        if !from_vertex.out_edges.insert(to) {
            debug_assert!(!self.roots.contains(&to), "root vertex had inbound edges");
            debug_assert!(
                !self.leaves.contains(&from),
                "leaf vertex had outbound edges"
            );
            debug_assert!(
                unsafe { self.vertices.get_unchecked(to).in_edges.contains(&from) },
                "in / out edge mismatch"
            );

            return Ok(AddEdgeStatus::AlreadyExists);
        }

        if from_vertex.out_edges.len() == 1 {
            let was_a_leaf = self.leaves.remove(&from);
            debug_assert!(was_a_leaf, "vertex with no outbound edges was not a leaf");
        }

        // Must succeed, checked above.
        let to_vertex = unsafe { self.vertices.get_unchecked_mut(to) };

        let did_not_exist = to_vertex.in_edges.insert(from);
        debug_assert!(did_not_exist, "in / out edge mismatch");

        if to_vertex.in_edges.len() == 1 {
            let was_a_root = self.roots.remove(&to);
            debug_assert!(was_a_root, "vertex with no inbound edges was not a root");
        }

        // Increment the edge counter.
        self.num_edges += 1;

        Ok(AddEdgeStatus::Added)
    }

    /// If vertex IDs `from` and `to` are valid, returns `true` if a directed edge exists in the graph between them.
    pub fn has_edge(&self, from: VID, to: VID) -> Result<bool, AccessEdgeError> {
        use AccessEdgeError::*;

        let from_vertex = self.vertices.get(from).ok_or(InvalidFromVertex)?;
        let to_vertex = self.vertices.get(to).ok_or(InvalidToVertex)?;

        if from == to {
            return Err(LoopEdge);
        }

        let has_edge = from_vertex.out_edges.contains(&to);

        debug_assert_eq!(
            has_edge,
            to_vertex.in_edges.contains(&from),
            "in / out edge mismatch"
        );

        Ok(has_edge)
    }

    /// If vertex IDs `from` and `to` are valid, tries to remove a directed edge between them.
    ///
    /// Does nothing if the edge does not exist.
    pub fn remove_edge(&mut self, from: VID, to: VID) -> Result<RemoveEdgeStatus, AccessEdgeError> {
        use AccessEdgeError::*;

        if !self.vertices.is_valid(from) {
            return Err(InvalidFromVertex);
        }

        if !self.vertices.is_valid(to) {
            return Err(InvalidToVertex);
        }

        if from == to {
            return Err(LoopEdge);
        }

        // Must succeed, checked above.
        let from_vertex = unsafe { self.vertices.get_unchecked_mut(from) };

        // Remove the `to` vertex from the set of outbound edges of the `from` vertex.
        if !from_vertex.out_edges.remove(&to) {
            return Ok(RemoveEdgeStatus::DoesNotExist);
        }

        // If the `from` vertex has no more outbound edges, make it a leaf vertex.
        if from_vertex.out_edges.is_empty() {
            let was_not_a_leaf = self.leaves.insert(from);
            debug_assert!(was_not_a_leaf, "leaf vertex had outbound edges")
        }

        // Must succeed, checked above.
        let to_vertex = unsafe { self.vertices.get_unchecked_mut(to) };

        let existed = to_vertex.in_edges.remove(&from);
        debug_assert!(existed, "in / out edge mismatch");

        // If the `to` vertex has no more inbound edges, make it a root vertex.
        if to_vertex.in_edges.is_empty() {
            let was_not_a_root = self.roots.insert(to);
            debug_assert!(was_not_a_root, "root vertex had inbound edges")
        }

        // Decrement the edge counter.
        debug_assert!(self.num_edges > 0);
        self.num_edges -= 1;

        Ok(RemoveEdgeStatus::Removed)
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
    pub fn roots(&self) -> impl Iterator<Item = &VID> {
        self.roots.iter()
    }

    /// Returns the current number of leaf vertices (with no outbound edges) in the graph.
    pub fn num_leaves(&self) -> usize {
        self.leaves.len()
    }

    /// Returns an iterator over all leaf vertices (with no outbound edges) in the graph, in no particular order.
    pub fn leaves(&self) -> impl Iterator<Item = &VID> {
        self.leaves.iter()
    }

    /// If vertex `id` is valid, returns the number of inbound esges to the vertex.
    pub fn num_in_neighbors(&self, id: VID) -> Option<usize> {
        self.vertices.get(id).map(|vertex| vertex.in_edges.len())
    }

    /// If vertex `id` is valid, returns an iterator over all vertices whith inbound edges to the vertex, in no particular order.
    pub fn in_neighbors(&self, id: VID) -> Option<impl Iterator<Item = &VID>> {
        self.vertices.get(id).map(|vertex| vertex.in_edges.iter())
    }

    /// If vertex `id` is valid, returns the number of outbound esges from the vertex.
    pub fn num_out_neighbors(&self, id: VID) -> Option<usize> {
        self.vertices.get(id).map(|vertex| vertex.out_edges.len())
    }

    /// If vertex `id` is valid, returns an iterator over all vertices whith outbound edges from the vertex, in no particular order.
    pub fn out_neighbors(&self, id: VID) -> Option<impl Iterator<Item = &VID>> {
        self.vertices.get(id).map(|vertex| vertex.out_edges.iter())
    }

    /// Returns `true` if the graph contains a cycle.
    pub fn is_cyclic(&self) -> bool {
        if self.num_vertices() > 0 && (self.num_roots() == 0 || self.num_leaves() == 0) {
            return true;
        }

        for root in self.roots() {
            if self.is_cyclic_dfs(*root, &mut HashSet::new()) {
                return true;
            }
        }

        false
    }

    pub(crate) fn vertices_inner(&self) -> impl Iterator<Item = &Vertex<VID, T>> {
        self.vertices.iter()
    }

    fn roots_inner(&self) -> impl Iterator<Item = &Vertex<VID, T>> {
        self.roots
            .iter()
            .map(move |&root| unsafe { self.vertices.get_unchecked(root) })
    }

    /// Tries to remove a directed edge between vertices `from` and `to`.
    ///
    /// Returns `true` if a directed edge between them existed and was removed;
    /// returns `false` otherwise.
    fn try_remove_edge(
        vertices: &mut Vertices<VID, T>,
        roots: &mut HashSet<VID>,
        leaves: &mut HashSet<VID>,
        num_edges: &mut usize,
        from: VID,
        to: VID,
    ) -> bool {
        let mut any = false;

        if vertices
            .get_mut(to)
            .and_then(|to_vertex| {
                to_vertex.in_edges.remove(&from).then(|| {
                    // If the `to` vertex has no more inbound edges, make it a root vertex.
                    if to_vertex.in_edges.is_empty() {
                        let was_not_a_root = roots.insert(to);
                        debug_assert!(was_not_a_root, "root vertex had inbound edges")
                    }
                })
            })
            .is_some()
        {
            any = true;
        }

        // Try to remove the `to` vertex from the set of outbound edges of the `from` vertex.
        if vertices
            .get_mut(from)
            .and_then(|from_vertex| {
                from_vertex.out_edges.remove(&to).then(|| {
                    // If the `from` vertex has no more outbound edges, make it a leaf vertex.
                    if from_vertex.out_edges.is_empty() {
                        let was_not_a_leaf = leaves.insert(from);
                        debug_assert!(was_not_a_leaf, "leaf vertex had outbound edges")
                    }
                })
            })
            .is_some()
        {
            any = true;
        }

        // Decrement the edge counter.
        if any {
            debug_assert!(*num_edges > 0);
            *num_edges -= 1;
        }

        any
    }

    /// The caller guarantees the vertex `id` is valid.
    fn out_neighbors_unchecked(&self, id: VID) -> impl Iterator<Item = &VID> {
        unsafe { self.vertices.get_unchecked(id) }.out_edges.iter()
    }

    /// Performs recursive depth first stack based cycle graph detection.
    /// Returns `true` if a cycle is detected starting at vertex `id`.
    ///
    /// The caller guarantees the vertex `id` is valid.
    fn is_cyclic_dfs(&self, id: VID, stack: &mut HashSet<VID>) -> bool {
        for child in self.out_neighbors_unchecked(id) {
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

impl<VID: VertexID, T: Clone> Graph<VID, T> {
    /// Returns a copy of the graph with redundant transitive edges removed.
    /// Returns `None` for a cyclic graph.
    ///
    /// The implementation is probably suboptimal, but gets the job done for relatively small graphs.
    ///
    /// <https://en.wikipedia.org/wiki/Transitive_reduction>
    pub fn transitive_reduction(&self) -> Option<Self> {
        if self.is_cyclic() {
            return None;
        }

        let mut graph = self.clone();

        let mut processed_grandchildren = HashSet::new();

        for root in self.roots_inner() {
            for &child in root.out_edges.iter() {
                // Skip children which were already processed as grandchildren.
                if processed_grandchildren.contains(&child) {
                    continue;
                }
                graph.transitive_reduction_process_vertex_and_child(
                    self,
                    root.id,
                    child,
                    &mut processed_grandchildren,
                );
            }

            processed_grandchildren.clear();
        }

        Some(graph)
    }

    /// Creates a [`TaskGraph`] representation of the graph.
    pub fn task_graph(&self) -> Option<TaskGraph<VID, T>> {
        let reduced = self.transitive_reduction()?;

        // Flatten the vertex IDs into actual object (vertex) indices in the `vertices` inner array,
        // removing the indirection.
        // Task graphs are immutable, we won't add or remove vertices or edges, only access the existing ones,
        // so simple indices is all we'll need.
        // Object indices are the same type as vertex IDs.

        let vertices = reduced
            .vertices_inner()
            .map(|vertex| vertex.task_vertex(&reduced))
            .collect();
        let roots = reduced
            .roots()
            // Must succeed, all roots are valid vertex IDs.
            .map(|&root| unsafe { reduced.vertices.object_index_unchecked(root) })
            .collect();

        Some(TaskGraph::new(vertices, roots))
    }

    /// Recursively removes redundant transitive edges from vertex `id` to all children of vertex `child`
    /// (i.e. grandchildren of `id`) (but not to `child` itself, of course);
    /// and then same recursively for `child` and all its grandchildren, and so on.
    ///
    /// The caller guarantees the vertex IDs `id` and `child` are valid.
    fn transitive_reduction_process_vertex_and_child(
        &mut self,
        orig: &Self,
        id: VID,
        child: VID,
        processed_grandchildren: &mut HashSet<VID>,
    ) {
        self.transitive_reduction_process_vertex_grandchildren(orig, id, child);

        for &grandchild in orig.out_neighbors_unchecked(child) {
            processed_grandchildren.insert(grandchild);

            self.transitive_reduction_process_vertex_and_child(
                orig,
                child,
                grandchild,
                processed_grandchildren,
            );
        }
    }

    /// Recursively removes redundant transitive edges from vertex `id` to all children of vertex `child`
    /// (i.e. grandchildren of `id`) (but not to `child` itself, of course).
    ///
    /// The caller guarantees the vertex IDs `id` and `child` are valid.
    fn transitive_reduction_process_vertex_grandchildren(
        &mut self,
        orig: &Self,
        id: VID,
        child: VID,
    ) {
        for &grandchild in orig.out_neighbors_unchecked(child) {
            Self::try_remove_edge(
                &mut self.vertices,
                &mut self.roots,
                &mut self.leaves,
                &mut self.num_edges,
                id,
                grandchild,
            );

            self.transitive_reduction_process_vertex_grandchildren(orig, id, grandchild);
        }
    }
}

#[cfg(test)]
mod tests {
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

        // Remove v1.
        assert_eq!(graph.remove_vertex(v1).unwrap(), -3);

        assert_eq!(graph.num_vertices(), 2);
        assert_eq!(graph.num_edges(), 0);
        assert_eq!(graph.num_roots(), 2);
        assert_eq!(graph.num_leaves(), 2);

        // v0 and v2 is a root.
        for &root in graph.roots() {
            assert!(root == v0 || root == v2);
        }

        // v0 and v2 is a leaf.
        for &leaf in graph.leaves() {
            assert!(leaf == v0 || leaf == v2);
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
        //   /----->-----\
        //  /---->---\    \
        // A -> B -> C -> D -> E    A -> B -> C -> D -> E
        //  \    \------>-----/      \--> F
        //   \--> F
        //                       =>
        // G -> H -> I               G -> H -> I
        //  \--->---/

        let mut graph = Graph::<u32, ()>::new();

        let a = graph.add_vertex(());
        let b = graph.add_vertex(());
        let c = graph.add_vertex(());
        let d = graph.add_vertex(());
        let e = graph.add_vertex(());
        let f = graph.add_vertex(());
        let g = graph.add_vertex(());
        let h = graph.add_vertex(());
        let i = graph.add_vertex(());

        graph.add_edge(a, b).unwrap();
        graph.add_edge(a, f).unwrap();
        graph.add_edge(b, c).unwrap();
        graph.add_edge(c, d).unwrap();
        graph.add_edge(d, e).unwrap();

        graph.add_edge(a, c).unwrap(); // Redundant edge.
        graph.add_edge(a, d).unwrap(); // Redundant edge.
        graph.add_edge(b, e).unwrap(); // Redundant edge.

        graph.add_edge(g, h).unwrap();
        graph.add_edge(h, i).unwrap();

        graph.add_edge(g, i).unwrap(); // Redundant edge.

        assert_eq!(graph.num_vertices(), 9);
        assert_eq!(graph.num_edges(), 11);
        assert_eq!(graph.num_roots(), 2);
        assert_eq!(graph.num_leaves(), 3);

        let reduced_graph = graph.transitive_reduction().unwrap();

        assert_eq!(reduced_graph.num_vertices(), 9);
        assert_eq!(reduced_graph.num_edges(), 7); // Redundant edges removed.
        assert_eq!(reduced_graph.num_roots(), 2);
        assert_eq!(reduced_graph.num_leaves(), 3);

        for &root in reduced_graph.roots() {
            assert!(root == a || root == g);

            if root == a {
                assert_eq!(reduced_graph.num_out_neighbors(root).unwrap(), 2);

                for &child in reduced_graph.out_neighbors(root).unwrap() {
                    assert!(child == b || child == f);

                    if child == b {
                        assert_eq!(reduced_graph.num_out_neighbors(child).unwrap(), 1);

                        for &child in reduced_graph.out_neighbors(child).unwrap() {
                            assert_eq!(child, c);

                            assert_eq!(reduced_graph.num_out_neighbors(child).unwrap(), 1);

                            for &child in reduced_graph.out_neighbors(child).unwrap() {
                                assert_eq!(child, d);

                                assert_eq!(reduced_graph.num_out_neighbors(child).unwrap(), 1);

                                for &child in reduced_graph.out_neighbors(child).unwrap() {
                                    assert_eq!(child, e);

                                    assert_eq!(reduced_graph.num_out_neighbors(child).unwrap(), 0);
                                }
                            }
                        }
                    } else {
                        assert_eq!(reduced_graph.num_out_neighbors(child).unwrap(), 0);
                    }
                }
            } else if root == g {
                assert_eq!(reduced_graph.num_out_neighbors(root).unwrap(), 1);

                for &child in reduced_graph.out_neighbors(root).unwrap() {
                    assert_eq!(child, h);

                    assert_eq!(reduced_graph.num_out_neighbors(child).unwrap(), 1);

                    for &child in reduced_graph.out_neighbors(child).unwrap() {
                        assert_eq!(child, i);

                        assert_eq!(reduced_graph.num_out_neighbors(child).unwrap(), 0);
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

        assert!(graph.transitive_reduction().is_none());
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

        assert!(graph.transitive_reduction().is_none());
    }
}
