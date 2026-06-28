//! Small UTF-8 string helpers shared across modules.
//!
//! Several call sites (tool stderr elision, web byte-offset slicing,
//! observability field capture) take a raw byte index that may land in the
//! middle of a multibyte char and must back off to a valid boundary before
//! slicing. `str::floor_char_boundary` / `ceil_char_boundary` cover this but are
//! still unstable, so these are the one crate-internal implementation.

/// Largest char boundary `<= idx`, clamped to `s.len()`.
pub(crate) fn floor_char_boundary(s: &str, idx: usize) -> usize {
    let mut idx = idx.min(s.len());
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

/// Smallest char boundary `>= idx`, clamped to `s.len()`.
pub(crate) fn ceil_char_boundary(s: &str, idx: usize) -> usize {
    let mut idx = idx.min(s.len());
    while idx < s.len() && !s.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}
