//! Cache revision deletion.
//!
//! [`HFCacheInfo::delete_revisions`] computes a [`DeleteCacheStrategy`] that
//! describes everything that would be removed if the targeted commit hashes
//! were deleted from the cache. [`DeleteCacheStrategy::execute`] applies the
//! plan to the filesystem.
//!
//! The split mirrors `huggingface_hub`'s `HFCacheInfo.delete_revisions` /
//! `DeleteCacheStrategy.execute` pair, which exists so a UI can show the
//! estimated freed space and per-path detail before the user commits to the
//! deletion.

use std::collections::HashSet;
use std::io;
use std::path::{Path, PathBuf};

use super::{CachedRepoInfo, HFCacheInfo};

/// Plan for removing one or more cached revisions from the local Hugging
/// Face cache. Returned by [`HFCacheInfo::delete_revisions`]; apply with
/// [`DeleteCacheStrategy::execute`].
#[derive(Debug, Clone)]
pub struct DeleteCacheStrategy {
    /// Total bytes that will be freed once [`execute`](Self::execute) runs.
    /// Approximates `du` on the targeted blobs — does not include lock or
    /// `.no_exist` cleanup, which `execute` performs but cannot pre-size.
    pub expected_freed_size: u64,
    /// Individual blob files under `<repo>/blobs/` to remove. A blob is only
    /// listed if no surviving revision in the same repo still points to it.
    pub blobs: HashSet<PathBuf>,
    /// `<repo>/refs/<name>` files for revisions being deleted.
    pub refs: HashSet<PathBuf>,
    /// Full repo directories to remove when every revision of a repo is being
    /// deleted (cleaner than wiping blobs/refs/snapshots one by one).
    pub repos: HashSet<PathBuf>,
    /// `<repo>/snapshots/<commit>/` directories for individual revisions
    /// being deleted (when other revisions of the same repo survive).
    pub snapshots: HashSet<PathBuf>,
    /// `<cache>/.locks/<repo_folder>/` directories whose blobs are being
    /// fully removed (whole-repo deletion only — orphan locks are safe to
    /// wipe once the corresponding blobs are gone). Empty for per-revision
    /// deletions, where other revisions may still need the locks.
    pub locks: HashSet<PathBuf>,
    /// Commit hashes the caller asked to delete that weren't found in the
    /// cache. Informational; deletion proceeds for the hashes that were
    /// found.
    pub missing_revisions: Vec<String>,
}

/// Per-path outcome of [`DeleteCacheStrategy::execute`]. Successful removals
/// produce no entry; tolerated failures (`NotFound`, `PermissionDenied`)
/// populate [`failures`](Self::failures); unexpected errors bubble out of
/// `execute` as `Err`.
#[derive(Debug)]
pub struct ExecuteResult {
    pub failures: Vec<Failure>,
}

/// A single tolerated failure during [`DeleteCacheStrategy::execute`].
#[derive(Debug)]
pub struct Failure {
    pub path: PathBuf,
    pub kind: PathKind,
    pub error: io::Error,
}

/// Which deletion phase a [`Failure::path`] came from. Mirrors the order
/// `execute` removes things in (`repos → snapshots → refs → blobs → locks`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PathKind {
    Repo,
    Snapshot,
    Ref,
    Blob,
    Locks,
}

impl HFCacheInfo {
    /// Plan the deletion of one or more cached revisions by commit hash.
    ///
    /// Hashes not found in the cache are returned in
    /// [`DeleteCacheStrategy::missing_revisions`] rather than failing —
    /// callers can surface these to the user without aborting the rest of
    /// the deletion.
    ///
    /// The returned strategy is purely a plan; nothing is removed from disk
    /// until [`DeleteCacheStrategy::execute`] is called. This split mirrors
    /// `huggingface_hub`'s `HFCacheInfo.delete_revisions` and gives callers a
    /// chance to surface the expected freed size and per-path detail in a UI
    /// before committing.
    pub fn delete_revisions(&self, commit_hashes: &[String]) -> DeleteCacheStrategy {
        DeleteCacheStrategy::compute(self, commit_hashes)
    }
}

impl DeleteCacheStrategy {
    /// Apply the deletion order `repos → snapshots → refs → blobs → locks`.
    /// Removing references (snapshots and refs) before the content they
    /// point at (blobs) keeps the cache walkable across an interrupted run:
    /// a half-finished deletion leaves extra blobs, not dangling snapshot
    /// pointers.
    ///
    /// Returns per-path failures rather than aborting on the first error —
    /// the cache should never end up half-deleted because a single stale
    /// lock file failed to remove.
    ///
    /// Idempotent: re-executing a strategy whose paths have already been
    /// removed reports each path as a `NotFound` entry in
    /// [`ExecuteResult::failures`] (matching Python `_try_delete_path`'s
    /// `FileNotFoundError`/`PermissionError` skip-and-continue behavior).
    pub fn execute(&self) -> io::Result<ExecuteResult> {
        let mut failures: Vec<Failure> = Vec::new();
        for path in &self.repos {
            remove_with_tolerance(path, PathKind::Repo, &mut failures)?;
        }
        for path in &self.snapshots {
            remove_with_tolerance(path, PathKind::Snapshot, &mut failures)?;
        }
        for path in &self.refs {
            remove_with_tolerance(path, PathKind::Ref, &mut failures)?;
        }
        for path in &self.blobs {
            remove_with_tolerance(path, PathKind::Blob, &mut failures)?;
        }
        for path in &self.locks {
            remove_with_tolerance(path, PathKind::Locks, &mut failures)?;
        }
        Ok(ExecuteResult { failures })
    }

    /// Build the strategy from a cache snapshot plus a set of target commit
    /// hashes. Direct port of `huggingface_hub`'s
    /// `HFCacheInfo.delete_revisions` (`utils/_cache_manager.py`).
    fn compute(cache_info: &HFCacheInfo, commit_hashes: &[String]) -> Self {
        let mut working_set: HashSet<String> = commit_hashes.iter().cloned().collect();
        // Group target hashes by their owning repo, in the order repos are
        // walked. `huggingface_hub` attributes a given commit hash to the
        // first repo it appears under (rare cross-repo SHA collisions
        // notwithstanding); we do the same by removing hashes from the
        // working set as we go.
        let mut queued: Vec<(&CachedRepoInfo, Vec<usize>)> = Vec::new();
        for repo in &cache_info.repos {
            let mut matched: Vec<usize> = Vec::new();
            for (idx, revision) in repo.revisions.iter().enumerate() {
                if working_set.remove(&revision.commit_hash) {
                    matched.push(idx);
                }
            }
            if !matched.is_empty() {
                queued.push((repo, matched));
            }
        }
        let mut missing_revisions: Vec<String> = working_set.into_iter().collect();
        missing_revisions.sort();

        let mut repo_paths: HashSet<PathBuf> = HashSet::new();
        let mut snapshot_paths: HashSet<PathBuf> = HashSet::new();
        let mut ref_paths: HashSet<PathBuf> = HashSet::new();
        let mut blob_paths: HashSet<PathBuf> = HashSet::new();
        let mut lock_paths: HashSet<PathBuf> = HashSet::new();
        let mut freed_size: u64 = 0;

        for (repo, deleted_indices) in queued {
            let every_revision_targeted = deleted_indices.len() == repo.revisions.len();
            if every_revision_targeted {
                repo_paths.insert(repo.repo_path.clone());
                freed_size += repo.size_on_disk;
                // Only plan a `.locks/<repo_folder>` deletion when the
                // directory actually exists; planning a removal of a
                // non-existent path would surface a redundant entry in
                // `ExecuteResult::failures` for every call site that never
                // ran a download (or ran one without `.locks/` ever being
                // created on this platform).
                if let Some(lock_dir) = locks_directory_for(repo, &cache_info.cache_dir)
                    && lock_dir.is_dir()
                {
                    lock_paths.insert(lock_dir);
                }
                continue;
            }
            // Per-revision path. Collect snapshots, refs, and revision-unique
            // blobs. Surviving revisions in the same repo keep their blobs.
            let deleted_set: HashSet<&str> = deleted_indices
                .iter()
                .map(|&i| repo.revisions[i].commit_hash.as_str())
                .collect();
            let surviving_revisions: Vec<&_> = repo
                .revisions
                .iter()
                .filter(|r| !deleted_set.contains(r.commit_hash.as_str()))
                .collect();
            for &idx in &deleted_indices {
                let revision = &repo.revisions[idx];
                snapshot_paths.insert(revision.snapshot_path.clone());
                for ref_name in &revision.refs {
                    let ref_path = repo.repo_path.join("refs").join(ref_name);
                    ref_paths.insert(ref_path);
                }
                for file in &revision.files {
                    let blob = &file.blob_path;
                    // Guard against double-counting when sibling deleted
                    // revisions share a blob.
                    if blob_paths.contains(blob) {
                        continue;
                    }
                    let still_referenced = surviving_revisions
                        .iter()
                        .any(|rev| rev.files.iter().any(|f| &f.blob_path == blob));
                    if !still_referenced {
                        blob_paths.insert(blob.clone());
                        freed_size += file.size_on_disk;
                    }
                }
            }
        }

        Self {
            expected_freed_size: freed_size,
            blobs: blob_paths,
            refs: ref_paths,
            repos: repo_paths,
            snapshots: snapshot_paths,
            locks: lock_paths,
            missing_revisions,
        }
    }
}

/// Map a cached repo to its sibling `.locks/<repo_folder>/` directory.
/// Returns `None` if the repo folder name is empty (an unusual edge case
/// that should never happen in practice but isn't worth panicking over).
fn locks_directory_for(repo: &CachedRepoInfo, cache_dir: &Path) -> Option<PathBuf> {
    let folder = repo.repo_path.file_name()?;
    if folder.is_empty() {
        return None;
    }
    Some(cache_dir.join(".locks").join(folder))
}

/// Remove `path` (file or directory). Collects "no such file" and "permission
/// denied" outcomes into `failures` (matching `huggingface_hub`'s
/// `_try_delete_path` skip-and-continue behavior); other errors bubble up.
fn remove_with_tolerance(path: &Path, kind: PathKind, failures: &mut Vec<Failure>) -> io::Result<()> {
    let result = if path.is_dir() {
        std::fs::remove_dir_all(path)
    } else {
        std::fs::remove_file(path)
    };
    match result {
        Ok(()) => Ok(()),
        Err(err) if is_tolerated(&err) => {
            failures.push(Failure {
                path: path.to_path_buf(),
                kind,
                error: err,
            });
            Ok(())
        },
        Err(err) => Err(err),
    }
}

fn is_tolerated(err: &io::Error) -> bool {
    matches!(err.kind(), io::ErrorKind::NotFound | io::ErrorKind::PermissionDenied)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::SystemTime;

    use tempfile::TempDir;

    use super::*;
    use crate::cache::{CachedFileInfo, CachedRepoInfo, CachedRevisionInfo, HFCacheInfo};

    fn make_file(snapshot: &Path, blobs: &Path, name: &str, contents: &[u8]) -> CachedFileInfo {
        let snap_path = snapshot.join(name);
        let blob_path = blobs.join(name);
        if let Some(parent) = snap_path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&blob_path, contents).unwrap();
        // Snapshot entry is a symlink on Unix; for the test fixture we use a
        // regular file copy so the test runs identically on platforms without
        // symlink support.
        fs::write(&snap_path, contents).unwrap();
        let metadata = fs::metadata(&blob_path).unwrap();
        CachedFileInfo {
            file_name: name.into(),
            file_path: snap_path,
            blob_path,
            size_on_disk: contents.len() as u64,
            blob_last_accessed: metadata.accessed().unwrap_or(SystemTime::now()),
            blob_last_modified: metadata.modified().unwrap_or(SystemTime::now()),
        }
    }

    fn make_revision(
        repo_path: &Path,
        commit: &str,
        files: Vec<CachedFileInfo>,
        refs: Vec<String>,
    ) -> CachedRevisionInfo {
        let snapshot = repo_path.join("snapshots").join(commit);
        fs::create_dir_all(&snapshot).unwrap();
        let size: u64 = files.iter().map(|f| f.size_on_disk).sum();
        CachedRevisionInfo {
            commit_hash: commit.into(),
            snapshot_path: snapshot,
            files,
            size_on_disk: size,
            refs,
            last_modified: SystemTime::now(),
        }
    }

    fn build_fixture() -> (TempDir, HFCacheInfo) {
        let dir = TempDir::new().unwrap();
        let cache_dir = dir.path().to_path_buf();
        let repo_folder = cache_dir.join("models--owner--repo");
        fs::create_dir_all(repo_folder.join("blobs")).unwrap();
        fs::create_dir_all(repo_folder.join("refs")).unwrap();
        fs::create_dir_all(repo_folder.join("snapshots")).unwrap();
        let snap_a = repo_folder.join("snapshots").join("aaaa");
        let snap_b = repo_folder.join("snapshots").join("bbbb");
        fs::create_dir_all(&snap_a).unwrap();
        fs::create_dir_all(&snap_b).unwrap();
        let blobs = repo_folder.join("blobs");
        let file1 = make_file(&snap_a, &blobs, "config.json", b"shared-config");
        let file2_a = make_file(&snap_a, &blobs, "model-a.bin", b"only-in-a");
        let file2_b = make_file(&snap_b, &blobs, "model-b.bin", b"only-in-b");
        // Build a CachedFileInfo entry for revision B that points at the SAME
        // shared blob path as revision A's `config.json` — this is the
        // dedup-by-blob-path behavior we want to exercise.
        let file1_b = CachedFileInfo {
            file_name: "config.json".into(),
            file_path: snap_b.join("config.json"),
            blob_path: file1.blob_path.clone(),
            size_on_disk: file1.size_on_disk,
            blob_last_accessed: file1.blob_last_accessed,
            blob_last_modified: file1.blob_last_modified,
        };
        // Write the snapshot pointer for the shared file in revision B too.
        fs::write(&file1_b.file_path, b"shared-config").unwrap();
        // Write a ref file so per-revision deletion has something to clean.
        fs::write(repo_folder.join("refs").join("main"), b"aaaa").unwrap();
        let revision_a = make_revision(&repo_folder, "aaaa", vec![file1, file2_a], vec!["main".into()]);
        let revision_b = make_revision(&repo_folder, "bbbb", vec![file1_b, file2_b], vec![]);
        let total_size: u64 = revision_a
            .files
            .iter()
            .chain(revision_b.files.iter())
            .map(|f| f.size_on_disk)
            .sum();
        // De-dup the shared blob in the repo total.
        let unique_blobs: HashSet<PathBuf> = revision_a
            .files
            .iter()
            .chain(revision_b.files.iter())
            .map(|f| f.blob_path.clone())
            .collect();
        let dedup_size: u64 = unique_blobs.iter().map(|p| fs::metadata(p).unwrap().len()).sum();
        let _ = total_size;
        let repo = CachedRepoInfo {
            repo_id: "owner/repo".into(),
            repo_type: "model",
            repo_path: repo_folder,
            revisions: vec![revision_a, revision_b],
            nb_files: unique_blobs.len(),
            size_on_disk: dedup_size,
            last_accessed: SystemTime::now(),
            last_modified: SystemTime::now(),
        };
        let info = HFCacheInfo {
            cache_dir,
            repos: vec![repo],
            size_on_disk: dedup_size,
            warnings: Vec::new(),
        };
        (dir, info)
    }

    #[test]
    fn delete_single_revision_keeps_shared_blob() {
        let (_dir, info) = build_fixture();
        let strategy = info.delete_revisions(&["aaaa".into()]);
        assert!(strategy.repos.is_empty(), "single-revision delete should not wipe the repo dir");
        assert_eq!(strategy.snapshots.len(), 1, "snapshot for the deleted revision");
        assert_eq!(strategy.refs.len(), 1, "ref pointing at the deleted revision");
        // Only revision-A's unique blob `model-a.bin` should be queued;
        // `config.json` is shared with revision B and survives.
        assert_eq!(strategy.blobs.len(), 1, "shared blob must survive: {:?}", strategy.blobs);
        assert_eq!(strategy.expected_freed_size, b"only-in-a".len() as u64);
        assert!(strategy.locks.is_empty(), "per-revision delete does not wipe locks");
        assert!(strategy.missing_revisions.is_empty());
    }

    #[test]
    fn delete_all_revisions_wipes_repo() {
        let (_dir, info) = build_fixture();
        let strategy = info.delete_revisions(&["aaaa".into(), "bbbb".into()]);
        assert_eq!(strategy.repos.len(), 1);
        assert!(strategy.snapshots.is_empty());
        assert!(strategy.refs.is_empty());
        assert!(strategy.blobs.is_empty());
        let expected_size = info.repos[0].size_on_disk;
        assert_eq!(strategy.expected_freed_size, expected_size);
        assert!(strategy.missing_revisions.is_empty());
    }

    #[test]
    fn missing_revisions_are_reported() {
        let (_dir, info) = build_fixture();
        let strategy = info.delete_revisions(&["aaaa".into(), "ffff".into(), "9999".into()]);
        // Only `aaaa` exists; the other two are reported as missing.
        let missing: HashSet<&str> = strategy.missing_revisions.iter().map(String::as_str).collect();
        assert!(missing.contains("ffff"));
        assert!(missing.contains("9999"));
        assert_eq!(strategy.missing_revisions.len(), 2);
    }

    #[test]
    fn execute_removes_the_targeted_paths() {
        let (_dir, info) = build_fixture();
        let repo_path = info.repos[0].repo_path.clone();
        let strategy = info.delete_revisions(&["aaaa".into(), "bbbb".into()]);
        let result = strategy.execute().unwrap();
        assert!(result.failures.is_empty(), "no failures expected: {:?}", result.failures);
        assert!(!repo_path.exists());
    }

    #[test]
    fn execute_is_idempotent_via_failure_reporting() {
        let (_dir, info) = build_fixture();
        let strategy = info.delete_revisions(&["aaaa".into(), "bbbb".into()]);
        strategy.execute().unwrap();
        // Re-running the same plan should not error; every path now
        // produces a NotFound entry in `failures`.
        let second = strategy.execute().unwrap();
        assert!(!second.failures.is_empty(), "second run should report tolerated NotFound failures");
        assert!(
            second.failures.iter().all(|f| matches!(f.kind, PathKind::Repo)),
            "only the repo path was queued for deletion"
        );
        assert!(second.failures.iter().all(|f| f.error.kind() == io::ErrorKind::NotFound));
    }
}
