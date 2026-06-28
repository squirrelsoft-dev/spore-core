//! Poison-tolerant locking helpers.
//!
//! The harness takes `std::sync::Mutex` / `RwLock` guards in many
//! fire-and-forget paths (observability emit, registries, sensor state). A
//! panic while any of those guards is held would poison the lock and turn every
//! later `.lock().unwrap()` into a cascading panic that takes down an otherwise
//! healthy run. None of these critical sections hold a guard across an `.await`
//! and none leave the protected data in a torn state, so recovering the guard
//! from a [`PoisonError`] is always safe here — we simply continue with the
//! data as it was.
//!
//! Use these extension methods instead of `.lock().unwrap()` /
//! `.read().unwrap()` / `.write().unwrap()` for any critical section.

use std::sync::{Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Poison-recovering access to a [`Mutex`].
pub(crate) trait MutexExt<T: ?Sized> {
    /// Lock the mutex, recovering the guard if the lock was poisoned.
    fn lock_recover(&self) -> MutexGuard<'_, T>;
}

impl<T: ?Sized> MutexExt<T> for Mutex<T> {
    #[inline]
    fn lock_recover(&self) -> MutexGuard<'_, T> {
        self.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Poison-recovering access to an [`RwLock`].
pub(crate) trait RwLockExt<T: ?Sized> {
    /// Acquire a shared read guard, recovering on poison.
    fn read_recover(&self) -> RwLockReadGuard<'_, T>;
    /// Acquire an exclusive write guard, recovering on poison.
    fn write_recover(&self) -> RwLockWriteGuard<'_, T>;
}

impl<T: ?Sized> RwLockExt<T> for RwLock<T> {
    #[inline]
    fn read_recover(&self) -> RwLockReadGuard<'_, T> {
        self.read().unwrap_or_else(PoisonError::into_inner)
    }

    #[inline]
    fn write_recover(&self) -> RwLockWriteGuard<'_, T> {
        self.write().unwrap_or_else(PoisonError::into_inner)
    }
}
