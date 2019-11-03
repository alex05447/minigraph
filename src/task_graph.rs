use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

use super::graph::{Graph, VertexID};

pub struct TaskVertex<T> {
    vertex: T,
    num_dependencies: usize,
    completed_dependencies: AtomicUsize,
}

impl<T> TaskVertex<T> {
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

pub struct TaskGraph<VID: VertexID, T> {
    vertices: HashMap<VID, TaskVertex<T>>,
    roots: Vec<VID>,
    out_edges: HashMap<VID, Vec<VID>>,
}

impl<VID: VertexID, T: Clone> TaskGraph<VID, T> {
    pub(super) fn new(
        graph: &Graph<VID, T>,
        vertices: &HashMap<VID, T>,
        roots: &HashSet<VID>,
        out_edges: &HashMap<VID, HashSet<VID>>,
    ) -> Self {
        let vertices = vertices
            .iter()
            .map(|(vertex_id, vertex)| {
                (
                    *vertex_id,
                    TaskVertex::new(vertex.clone(), graph.num_in_neighbors(*vertex_id).unwrap()),
                )
            })
            .collect();

        let roots = roots.iter().map(|vertex_id| *vertex_id).collect();

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

impl<VID: VertexID, T> TaskGraph<VID, T> {
    pub fn reset(&self) {
        for (_, vertex) in self.vertices.iter() {
            vertex.reset()
        }
    }

    pub fn roots(&self) -> TaskVertexIterator<'_, VID, T> {
        TaskVertexIterator::new(&self.vertices, Some(self.roots.iter()))
    }

    pub fn dependencies(&self, vertex_id: VID) -> TaskVertexIterator<'_, VID, T> {
        debug_assert!(self.vertices.contains_key(&vertex_id));

        TaskVertexIterator::new(
            &self.vertices,
            self.out_edges
                .get(&vertex_id)
                .map_or(None, |out_edges| Some(out_edges.iter())),
        )
    }
}

pub struct TaskVertexIterator<'a, VID: VertexID, T> {
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

impl<'a, VID: VertexID, T> std::iter::Iterator for TaskVertexIterator<'a, VID, T> {
    type Item = (VID, &'a TaskVertex<T>);

    fn next(&mut self) -> Option<Self::Item> {
        self.vertex_ids
            .as_mut()
            .map(|vertex_ids| vertex_ids.next())
            .map_or(None, |vertex_id| {
                vertex_id.map(|vertex_id| (*vertex_id, self.vertices.get(vertex_id).unwrap()))
            })
    }
}

pub struct SystemDesc<'a, SID, RID> {
    pub id: SID,
    pub read: &'a [RID],
    pub write: &'a [RID],
}

impl<'a, SID, RID> SystemDesc<'a, SID, RID> {
    pub fn new(id: SID, read: &'a [RID], write: &'a [RID]) -> Self {
        Self { id, read, write }
    }
}

#[derive(Debug)]
pub enum BuildSystemGraphError<SID> {
    DuplicateSystemID(SID),
    CyclicGraph,
}

pub fn build_system_graph<'a, VID, SID, RID>(
    systems: &[SystemDesc<'a, SID, RID>],
) -> Result<Graph<VID, SID>, BuildSystemGraphError<SID>>
where
    VID: VertexID,
    SID: PartialEq + Clone,
    RID: PartialEq,
{
    let mut systems_and_vertex_ids = Vec::with_capacity(systems.len());
    let mut graph = Graph::new();

    for system in systems.iter() {
        add_system_to_graph(system, &mut graph, &mut systems_and_vertex_ids)?;
    }

    if graph.num_vertices() > 0 && (graph.num_roots() == 0 || graph.num_leaves() == 0) {
        return Err(BuildSystemGraphError::CyclicGraph);
    }

    Ok(graph)
}

fn add_system_to_graph<'a, 'b, VID, SID, RID>(
    system: &'b SystemDesc<'b, SID, RID>,
    graph: &mut Graph<VID, SID>,
    systems_and_vertex_ids: &mut Vec<(&'a SystemDesc<'a, SID, RID>, VID)>,
) -> Result<(), BuildSystemGraphError<SID>>
where
    'b: 'a,
    VID: VertexID,
    SID: PartialEq + Clone,
    RID: PartialEq,
{
    // Duplicate system id.
    if let Some(_) = systems_and_vertex_ids
        .iter()
        .find(|(added_system, _)| added_system.id == system.id)
    {
        return Err(BuildSystemGraphError::DuplicateSystemID(system.id.clone()));
    }

    // Add to the graph.
    let vertex_id = graph.add_vertex(system.id.clone());

    // For each of the system's reads check if it was written by a previous system.
    // If true, add an edge from the latest writer to the system.
    for read in system.read.iter() {
        for (writer, writer_vertex_id) in systems_and_vertex_ids.iter().rev() {
            if writer.write.contains(read) {
                graph.add_edge(*writer_vertex_id, vertex_id).unwrap();
                break;
            }
        }
    }

    // For each of the system's writes check if it was written by a previous system.
    // If true, add an edge from the latest writer to the system.
    for write in system.write.iter() {
        for (writer, writer_vertex_id) in systems_and_vertex_ids.iter().rev() {
            if writer.write.contains(write) {
                graph.add_edge(*writer_vertex_id, vertex_id).unwrap();
                break;
            }
        }
    }

    systems_and_vertex_ids.push((system, vertex_id));

    Ok(())
}
