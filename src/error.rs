use std::{
    error::Error,
    fmt::{Display, Formatter},
};

/// An error returned when accessing the [`Graph`](crate::Graph) edge.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AccessEdgeError {
    /// The `from` vertex ID is invalid.
    InvalidFromVertex,
    /// The `to` vertex ID is invalid.
    InvalidToVertex,
    /// Loop edge (`from` and `to` vertices are the same).
    LoopEdge,
}

impl Error for AccessEdgeError {}

impl Display for AccessEdgeError {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        use AccessEdgeError::*;

        match self {
            InvalidFromVertex => "`from` vertex ID is invalid".fmt(f),
            InvalidToVertex => "`to` vertex ID is invalid".fmt(f),
            LoopEdge => "loop edge (`from` and `to` vertex IDs are the same)".fmt(f),
        }
    }
}

/// Status returned by [`Graph::add_edge`](crate::Graph::add_edge).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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
            Added => "a new directed edge was added to the graph between the two vertices".fmt(f),
            AlreadyExists => "a directed edge between the two vertices already exists".fmt(f),
        }
    }
}

/// Status returned by [`Graph::remove_edge`](crate::Graph::remove_edge).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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
            Removed => {
                "a previously existing directed edge between the two vertices was removed".fmt(f)
            }
            DoesNotExist => "a directed edge between the two vertices does not exist".fmt(f),
        }
    }
}
