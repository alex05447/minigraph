use {
    smallvec::SmallVec,
    std::{borrow::Borrow, iter::Iterator},
};

/// [`std::collections::HashSet`], but optimized for a (very) small number of entries.
///
/// Uses a [`smallvec::SmallVec`] for storage.
/// `O(n)` [`insert`](SmallSet::insert) / [`remove`](SmallSet::remove) / [`contains`](SmallSet::contains) complexity.
#[derive(Clone)]
pub(crate) struct SmallSet<T, const N: usize>(SmallVec<[T; N]>);

impl<T: Eq, const N: usize> SmallSet<T, N> {
    /// See [`std::collections::HashSet::new`].
    pub(crate) fn new() -> Self {
        Self(SmallVec::new())
    }

    /// See [`std::collections::HashSet::insert`].
    ///
    /// If the set did not have this value present, true is returned.
    /// If the set did have this value present, false is returned.
    pub(crate) fn insert(&mut self, value: T) -> bool {
        self.0.iter().position(|v| *v == value).map_or_else(
            || {
                self.0.push(value);
                true
            },
            |_| false,
        )
    }

    /// See [`std::collections::HashSet::removed`].
    ///
    /// Returns whether the value was present in the set.
    pub(crate) fn remove<Q: ?Sized>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Eq,
    {
        self.0
            .iter()
            .position(|v| v.borrow() == value)
            .map_or(false, |p| {
                self.0.swap_remove(p);
                true
            })
    }

    /// See [`std::collections::HashSet::contains`].
    pub(crate) fn contains<Q: ?Sized>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Eq,
    {
        self.0.iter().find(|v| (*v).borrow() == value).is_some()
    }

    /// See [`std::collections::HashSet::iter`].
    pub(crate) fn iter(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
    }

    /// See [`std::collections::HashSet::len`].
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    /// See [`std::collections::HashSet::is_empty`].
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn inner(&self) -> &SmallVec<[T; N]> {
        &self.0
    }
}
