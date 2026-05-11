//! Integration tests for downloads and progress tracking.
//!
//! Read-only tests (downloads from hardcoded repos) use **prod** (huggingface.co).
//! Write tests (upload progress) create temporary repos on **hub-ci** and require HF_TEST_WRITE=1.
//!
//! Run read-only: HF_TOKEN=hf_xxx cargo test -p hf-hub --test download_test
//! Run all: HF_TOKEN=hf_xxx HF_TEST_WRITE=1 cargo test -p hf-hub --test download_test
//!
//! CI: read-only tests use HF_PROD_TOKEN, write tests use HF_CI_TOKEN against hub-ci.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use hf_hub::progress::{DownloadEvent, FileStatus, ProgressEvent, ProgressHandler, UploadEvent};
use hf_hub::repository::{AddSource, CommitOperation};
use hf_hub::{HFClient, HFClientBuilder, HFRepository, RepoTypeDataset, RepoTypeModel};
use integration_tests::test_utils::*;
use sha2::{Digest, Sha256};

fn prod_api() -> Option<HFClient> {
    if is_ci() {
        let token = resolve_prod_token()?;
        Some(build_client(&token, PROD_ENDPOINT))
    } else {
        default_api()
    }
}

fn hub_ci_api() -> Option<HFClient> {
    if is_ci() {
        let token = std::env::var(HF_CI_TOKEN).ok()?;
        Some(build_client(&token, HUB_CI_ENDPOINT))
    } else {
        default_api()
    }
}

fn default_api() -> Option<HFClient> {
    let token = std::env::var(HF_TOKEN).ok()?;
    let endpoint = std::env::var(HF_ENDPOINT).unwrap_or_else(|_| PROD_ENDPOINT.to_string());
    Some(build_client(&token, &endpoint))
}

fn build_client(token: &str, endpoint: &str) -> HFClient {
    HFClientBuilder::new()
        .token(token)
        .endpoint(endpoint)
        .build()
        .expect("Failed to create HFClient")
}

fn uuid_short() -> String {
    format!("{:016x}", rand::random::<u64>())
}

async fn cached_username(client: &HFClient) -> String {
    client.whoami().send().await.expect("whoami failed").username
}

async fn create_test_repo(client: &HFClient) -> String {
    let username = cached_username(client).await;
    let repo_id = format!("{}/hfrs-progress-test-{}", username, uuid_short());
    client
        .create_repository()
        .repo_type(RepoTypeModel)
        .repo_id(&repo_id)
        .private(true)
        .exist_ok(false)
        .send()
        .await
        .expect("create_repo failed");
    repo_id
}

async fn delete_test_repo(client: &HFClient, repo_id: &str) {
    let _ = client
        .delete_repository()
        .repo_type(RepoTypeModel)
        .repo_id(repo_id)
        .send()
        .await;
}

const TEST_MODEL_PARTS: (&str, &str) = ("hf-internal-testing", "tiny-gemma3");
const TEST_DATASET_PARTS: (&str, &str) = ("hf-internal-testing", "cats_vs_dogs_sample");

fn model(client: &HFClient, owner: &str, name: &str) -> HFRepository<RepoTypeModel> {
    client.model(owner, name)
}

fn dataset(client: &HFClient, owner: &str, name: &str) -> HFRepository<RepoTypeDataset> {
    client.dataset(owner, name)
}

#[tokio::test]
async fn test_download_small_json_file() {
    let Some(client) = prod_api() else { return };
    let dir = tempfile::tempdir().unwrap();
    let (owner, name) = TEST_MODEL_PARTS;

    let path = model(&client, owner, name)
        .download_file()
        .filename("config.json")
        .local_dir(dir.path().to_path_buf())
        .send()
        .await
        .unwrap();

    assert!(path.exists());
    let content = std::fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert!(json.get("model_type").is_some());
}

#[tokio::test]
async fn test_download_preserves_subdirectory_structure() {
    let Some(client) = prod_api() else { return };
    let dir = tempfile::tempdir().unwrap();
    let (owner, name) = TEST_MODEL_PARTS;

    let path = model(&client, owner, name)
        .download_file()
        .filename("config.json")
        .local_dir(dir.path().to_path_buf())
        .send()
        .await
        .unwrap();

    assert_eq!(path, dir.path().join("config.json"));
    assert!(path.exists());
}

#[tokio::test]
async fn test_download_with_specific_revision() {
    let Some(client) = prod_api() else { return };
    let dir = tempfile::tempdir().unwrap();
    let (owner, name) = TEST_MODEL_PARTS;

    let path = model(&client, owner, name)
        .download_file()
        .filename("config.json")
        .local_dir(dir.path().to_path_buf())
        .revision("main")
        .send()
        .await
        .unwrap();

    assert!(path.exists());
    let content = std::fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert!(json.get("model_type").is_some());
}

#[tokio::test]
async fn test_download_dataset_file() {
    let Some(client) = prod_api() else { return };
    let dir = tempfile::tempdir().unwrap();
    let (owner, name) = TEST_DATASET_PARTS;

    let path = dataset(&client, owner, name)
        .download_file()
        .filename("README.md")
        .local_dir(dir.path().to_path_buf())
        .send()
        .await
        .unwrap();

    assert!(path.exists());
    let content = std::fs::read_to_string(&path).unwrap();
    assert!(!content.is_empty());
}

#[tokio::test]
async fn test_download_nonexistent_file_returns_error() {
    let Some(client) = prod_api() else { return };
    let dir = tempfile::tempdir().unwrap();
    let (owner, name) = TEST_MODEL_PARTS;

    let result = model(&client, owner, name)
        .download_file()
        .filename("this_file_does_not_exist_at_all.bin")
        .local_dir(dir.path().to_path_buf())
        .send()
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_download_from_nonexistent_repo_returns_error() {
    let Some(client) = prod_api() else { return };
    let dir = tempfile::tempdir().unwrap();

    let result = model(&client, "this-user-does-not-exist-99999", "this-repo-does-not-exist")
        .download_file()
        .filename("anything.txt")
        .local_dir(dir.path().to_path_buf())
        .send()
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_download_multiple_files_to_same_dir() {
    let Some(client) = prod_api() else { return };
    let dir = tempfile::tempdir().unwrap();
    let (owner, name) = TEST_MODEL_PARTS;
    let repo = model(&client, owner, name);

    for filename in &["config.json", "README.md"] {
        let path = repo
            .download_file()
            .filename(*filename)
            .local_dir(dir.path().to_path_buf())
            .send()
            .await
            .unwrap();
        assert!(path.exists());
    }

    assert!(dir.path().join("config.json").exists());
    assert!(dir.path().join("README.md").exists());
}

#[tokio::test]
async fn test_download_file_content_is_deterministic() {
    let Some(client) = prod_api() else { return };
    let dir1 = tempfile::tempdir().unwrap();
    let dir2 = tempfile::tempdir().unwrap();
    let (owner, name) = TEST_MODEL_PARTS;
    let repo = model(&client, owner, name);

    for dir in [&dir1, &dir2] {
        repo.download_file()
            .filename("config.json")
            .local_dir(dir.path().to_path_buf())
            .send()
            .await
            .unwrap();
    }

    let content1 = std::fs::read(dir1.path().join("config.json")).unwrap();
    let content2 = std::fs::read(dir2.path().join("config.json")).unwrap();

    let hash1 = Sha256::digest(&content1);
    let hash2 = Sha256::digest(&content2);
    assert_eq!(hash1, hash2);
}

#[tokio::test]
async fn test_download_overwrites_existing_file() {
    let Some(client) = prod_api() else { return };
    let dir = tempfile::tempdir().unwrap();
    let (owner, name) = TEST_MODEL_PARTS;

    let dest = dir.path().join("config.json");
    std::fs::write(&dest, "old content").unwrap();

    model(&client, owner, name)
        .download_file()
        .filename("config.json")
        .local_dir(dir.path().to_path_buf())
        .send()
        .await
        .unwrap();

    let content = std::fs::read_to_string(&dest).unwrap();
    assert_ne!(content, "old content");
    assert!(content.contains("model_type"));
}

// --- Range / partial download tests (non-xet) ---

#[tokio::test]
async fn test_download_stream_full_file() {
    let Some(client) = prod_api() else { return };
    let (owner, name) = TEST_MODEL_PARTS;
    let repo = model(&client, owner, name);

    let (content_length, stream) = repo.download_file_stream().filename("config.json").send().await.unwrap();

    assert!(content_length.is_some());

    futures::pin_mut!(stream);
    let mut bytes = Vec::new();
    while let Some(chunk) = stream.next().await {
        bytes.extend_from_slice(&chunk.unwrap());
    }

    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(json.get("model_type").is_some());
}

#[tokio::test]
async fn test_download_stream_range_first_bytes() {
    let Some(client) = prod_api() else { return };
    let (owner, name) = TEST_MODEL_PARTS;
    let repo = model(&client, owner, name);

    // Download just the first 20 bytes
    let (content_length, stream) = repo
        .download_file_stream()
        .filename("config.json")
        .range(0..20u64)
        .send()
        .await
        .unwrap();

    assert!(content_length.unwrap() <= 20);

    futures::pin_mut!(stream);
    let mut bytes = Vec::new();
    while let Some(chunk) = stream.next().await {
        bytes.extend_from_slice(&chunk.unwrap());
    }
    assert_eq!(bytes.len(), 20);
}

#[tokio::test]
async fn test_download_stream_range_middle_bytes() {
    let Some(client) = prod_api() else { return };
    let (owner, name) = TEST_MODEL_PARTS;
    let repo = model(&client, owner, name);

    // First download the full file for comparison
    let (_len, full_stream) = repo.download_file_stream().filename("config.json").send().await.unwrap();
    futures::pin_mut!(full_stream);
    let mut full_bytes = Vec::new();
    while let Some(chunk) = full_stream.next().await {
        full_bytes.extend_from_slice(&chunk.unwrap());
    }

    // Now download a range from the middle
    let start = 10u64;
    let end = 50u64;
    let (_len, range_stream) = repo
        .download_file_stream()
        .filename("config.json")
        .range(start..end)
        .send()
        .await
        .unwrap();

    futures::pin_mut!(range_stream);
    let mut range_bytes = Vec::new();
    while let Some(chunk) = range_stream.next().await {
        range_bytes.extend_from_slice(&chunk.unwrap());
    }

    assert_eq!(range_bytes.len(), (end - start) as usize);
    assert_eq!(range_bytes, &full_bytes[start as usize..end as usize]);
}

#[tokio::test]
async fn test_download_stream_range_content_matches_full_download() {
    let Some(client) = prod_api() else { return };
    let (owner, name) = TEST_MODEL_PARTS;
    let repo = model(&client, owner, name);
    let dir = tempfile::tempdir().unwrap();

    // Download full file to disk for reference
    let path = repo
        .download_file()
        .filename("config.json")
        .local_dir(dir.path().to_path_buf())
        .send()
        .await
        .unwrap();
    let full_bytes = std::fs::read(&path).unwrap();

    // Stream the first 100 bytes
    let range_end = 100u64.min(full_bytes.len() as u64);
    let (_len, stream) = repo
        .download_file_stream()
        .filename("config.json")
        .range(0..range_end)
        .send()
        .await
        .unwrap();

    futures::pin_mut!(stream);
    let mut streamed = Vec::new();
    while let Some(chunk) = stream.next().await {
        streamed.extend_from_slice(&chunk.unwrap());
    }

    assert_eq!(streamed, &full_bytes[..range_end as usize]);
}

// --- Progress tracking tests ---

struct RecordingHandler {
    events: Mutex<Vec<ProgressEvent>>,
}

impl RecordingHandler {
    fn new() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
        }
    }

    fn events(&self) -> Vec<ProgressEvent> {
        self.events.lock().unwrap().clone()
    }
}

impl ProgressHandler for RecordingHandler {
    fn on_progress(&self, event: &ProgressEvent) {
        self.events.lock().unwrap().push(event.clone());
    }
}

#[tokio::test]
async fn test_download_file_with_progress_to_local_dir() {
    let Some(client) = prod_api() else { return };
    let (owner, name) = TEST_MODEL_PARTS;
    let repo = model(&client, owner, name);

    let handler = Arc::new(RecordingHandler::new());

    let dir = tempfile::tempdir().unwrap();
    let path = repo
        .download_file()
        .filename("config.json")
        .local_dir(dir.path().to_path_buf())
        .progress(handler.clone())
        .send()
        .await
        .unwrap();
    assert!(path.exists());

    let events = handler.events();
    assert!(!events.is_empty(), "should have received progress events");

    // First event should be Download(Start)
    assert!(
        matches!(&events[0], ProgressEvent::Download(DownloadEvent::Start { total_files: 1, .. })),
        "first event should be Download(Start), got {:?}",
        events[0]
    );

    // Last event should be Download(Complete)
    assert!(
        matches!(events.last().unwrap(), ProgressEvent::Download(DownloadEvent::Complete)),
        "last event should be Download(Complete)"
    );

    // Should have at least one Progress event with InProgress or Complete
    let has_progress = events
        .iter()
        .any(|e| matches!(e, ProgressEvent::Download(DownloadEvent::Progress { .. })));
    assert!(has_progress, "should have at least one Progress event");

    // Should have a Complete file status
    let has_file_complete = events.iter().any(|e| {
        if let ProgressEvent::Download(DownloadEvent::Progress { files }) = e {
            files.iter().any(|f| f.status == FileStatus::Complete)
        } else {
            false
        }
    });
    assert!(has_file_complete, "should have a file Complete status event");
}

#[tokio::test]
async fn test_download_file_with_progress_to_cache() {
    let Some(client) = prod_api() else { return };
    let (owner, name) = TEST_MODEL_PARTS;
    let repo = model(&client, owner, name);

    let handler = Arc::new(RecordingHandler::new());

    let path = repo
        .download_file()
        .filename("config.json")
        .force_download(true)
        .progress(handler.clone())
        .send()
        .await
        .unwrap();
    assert!(path.exists());

    let events = handler.events();
    assert!(!events.is_empty(), "should have received progress events");

    assert!(
        matches!(&events[0], ProgressEvent::Download(DownloadEvent::Start { total_files: 1, .. })),
        "first event should be Download(Start)"
    );
    assert!(
        matches!(events.last().unwrap(), ProgressEvent::Download(DownloadEvent::Complete)),
        "last event should be Download(Complete)"
    );
}

#[tokio::test]
async fn test_download_file_to_bytes_with_progress() {
    let Some(client) = prod_api() else { return };
    let (owner, name) = TEST_MODEL_PARTS;
    let repo = model(&client, owner, name);

    let handler = Arc::new(RecordingHandler::new());

    let bytes = repo
        .download_file_to_bytes()
        .filename("config.json")
        .progress(handler.clone())
        .send()
        .await
        .unwrap();
    assert!(!bytes.is_empty());

    let events = handler.events();
    assert!(!events.is_empty(), "should have received progress events");
    assert!(
        matches!(&events[0], ProgressEvent::Download(DownloadEvent::Start { total_files: 1, .. })),
        "first event should be Download(Start), got {:?}",
        events[0]
    );
    assert!(
        matches!(events.last().unwrap(), ProgressEvent::Download(DownloadEvent::Complete)),
        "last event should be Download(Complete)"
    );
    let has_progress = events
        .iter()
        .any(|e| matches!(e, ProgressEvent::Download(DownloadEvent::Progress { .. })));
    assert!(has_progress, "should have at least one Progress event");
}

#[tokio::test]
async fn test_download_file_stream_with_progress() {
    let Some(client) = prod_api() else { return };
    let (owner, name) = TEST_MODEL_PARTS;
    let repo = model(&client, owner, name);

    let handler = Arc::new(RecordingHandler::new());

    let (_content_length, mut stream) = repo
        .download_file_stream()
        .filename("config.json")
        .progress(handler.clone())
        .send()
        .await
        .unwrap();

    let mut total = 0u64;
    while let Some(chunk) = stream.next().await {
        total += chunk.unwrap().len() as u64;
    }
    assert!(total > 0);

    let events = handler.events();
    assert!(!events.is_empty(), "should have received progress events");
    assert!(
        matches!(&events[0], ProgressEvent::Download(DownloadEvent::Start { total_files: 1, .. })),
        "first event should be Download(Start), got {:?}",
        events[0]
    );
    assert!(
        matches!(events.last().unwrap(), ProgressEvent::Download(DownloadEvent::Complete)),
        "last event should be Download(Complete)"
    );
    let has_progress = events
        .iter()
        .any(|e| matches!(e, ProgressEvent::Download(DownloadEvent::Progress { .. })));
    assert!(has_progress, "should have at least one Progress event");
}

#[tokio::test]
async fn test_download_with_no_progress_handler() {
    let Some(client) = prod_api() else { return };
    let (owner, name) = TEST_MODEL_PARTS;
    let repo = model(&client, owner, name);

    let dir = tempfile::tempdir().unwrap();
    let path = repo
        .download_file()
        .filename("config.json")
        .local_dir(dir.path().to_path_buf())
        .send()
        .await
        .unwrap();
    assert!(path.exists());
}

// --- Upload progress tests (write to hub-ci) ---

fn repo_from_id(client: &HFClient, repo_id: &str) -> HFRepository<RepoTypeModel> {
    let parts: Vec<&str> = repo_id.splitn(2, '/').collect();
    client.model(parts[0], parts[1])
}

#[tokio::test]
async fn test_upload_file_with_progress() {
    let Some(client) = hub_ci_api() else { return };
    if !write_enabled() {
        return;
    }
    let repo_id = create_test_repo(&client).await;
    let repo = repo_from_id(&client, &repo_id);

    let handler = Arc::new(RecordingHandler::new());

    let result = repo
        .upload_file()
        .source(AddSource::bytes(b"hello from progress test".as_slice()))
        .path_in_repo("progress_test.txt")
        .commit_message("upload with progress tracking")
        .progress(handler.clone())
        .send()
        .await;

    delete_test_repo(&client, &repo_id).await;
    let commit = result.unwrap();
    assert!(commit.commit_oid.is_some());

    let events = handler.events();
    assert!(!events.is_empty(), "should have received upload progress events");

    assert!(
        matches!(&events[0], ProgressEvent::Upload(UploadEvent::Start { total_files: 1, .. })),
        "first event should be Upload(Start), got {:?}",
        events[0]
    );

    assert!(
        matches!(events.last().unwrap(), ProgressEvent::Upload(UploadEvent::Complete)),
        "last event should be Upload(Complete)"
    );

    let has_committing = events
        .iter()
        .any(|e| matches!(e, ProgressEvent::Upload(UploadEvent::Committing)));
    assert!(has_committing, "should have a Committing event");
}

#[tokio::test]
async fn test_create_commit_with_progress_multiple_files() {
    let Some(client) = hub_ci_api() else { return };
    if !write_enabled() {
        return;
    }
    let repo_id = create_test_repo(&client).await;
    let repo = repo_from_id(&client, &repo_id);

    let handler = Arc::new(RecordingHandler::new());

    let result = repo
        .create_commit()
        .operations(vec![
            CommitOperation::add_bytes("file_a.txt", b"content a".to_vec()),
            CommitOperation::add_bytes("file_b.txt", b"content b".to_vec()),
        ])
        .commit_message("multi-file commit with progress")
        .progress(handler.clone())
        .send()
        .await;

    delete_test_repo(&client, &repo_id).await;
    let commit = result.unwrap();
    assert!(commit.commit_oid.is_some());

    let events = handler.events();
    assert!(!events.is_empty(), "should have received upload progress events");

    if let ProgressEvent::Upload(UploadEvent::Start {
        total_files,
        total_bytes,
    }) = &events[0]
    {
        assert_eq!(*total_files, 2);
        assert_eq!(*total_bytes, 18); // "content a" + "content b" = 9 + 9
    } else {
        panic!("first event should be Upload(Start), got {:?}", events[0]);
    }

    assert!(
        matches!(events.last().unwrap(), ProgressEvent::Upload(UploadEvent::Complete)),
        "last event should be Upload(Complete)"
    );
}

#[tokio::test]
async fn test_upload_with_no_progress_handler() {
    let Some(client) = hub_ci_api() else { return };
    if !write_enabled() {
        return;
    }
    let repo_id = create_test_repo(&client).await;
    let repo = repo_from_id(&client, &repo_id);

    let result = repo
        .upload_file()
        .source(AddSource::bytes(b"no handler test".as_slice()))
        .path_in_repo("no_handler.txt")
        .commit_message("upload without progress handler")
        .send()
        .await;

    delete_test_repo(&client, &repo_id).await;
    result.unwrap();
}

#[tokio::test]
async fn test_snapshot_download_exactly_one_complete_per_file() {
    let Some(client) = prod_api() else { return };
    let (owner, name) = TEST_MODEL_PARTS;
    let repo = model(&client, owner, name);

    let handler = Arc::new(RecordingHandler::new());
    let dir = tempfile::tempdir().unwrap();

    repo.snapshot_download()
        .local_dir(dir.path().to_path_buf())
        .allow_patterns(vec!["*.json".to_string()])
        .force_download(true)
        .progress(handler.clone())
        .send()
        .await
        .unwrap();

    let events = handler.events();

    // Count Complete events per filename
    let mut complete_counts: HashMap<String, usize> = HashMap::new();
    for event in &events {
        if let ProgressEvent::Download(DownloadEvent::Progress { files }) = event {
            for fp in files {
                if fp.status == FileStatus::Complete {
                    *complete_counts.entry(fp.filename.clone()).or_default() += 1;
                }
            }
        }
    }

    assert!(!complete_counts.is_empty(), "should have at least one file Complete event");

    for (filename, count) in &complete_counts {
        assert_eq!(*count, 1, "file '{filename}' had {count} Complete events, expected exactly 1");
    }

    // The files_bar count should match: total_files from Start == number of distinct Complete files
    if let Some(ProgressEvent::Download(DownloadEvent::Start { total_files, .. })) = events.first() {
        assert_eq!(
            *total_files,
            complete_counts.len(),
            "total_files in Start ({total_files}) should match number of completed files ({})",
            complete_counts.len()
        );
    }
}

// --- Resume tests ---

/// Build a client whose cache lives in `cache_dir`. Uses prod credentials when
/// available; returns `None` to skip the test otherwise (matches other tests
/// in this file).
fn cache_scoped_client(cache_dir: &Path) -> Option<HFClient> {
    let token = std::env::var(HF_TOKEN).ok().or_else(|| resolve_prod_token())?;
    let endpoint = std::env::var(HF_ENDPOINT).unwrap_or_else(|_| PROD_ENDPOINT.to_string());
    Some(
        HFClientBuilder::new()
            .token(token)
            .endpoint(endpoint)
            .cache_dir(cache_dir)
            .build()
            .expect("Failed to create HFClient"),
    )
}

/// Resolve the equivalent on-disk path in `dest_cache` given a canonical blob
/// path that lives under `source_cache`. Both caches must share the layout
/// (same repo, same etag), which is true for any two clients hitting the same
/// file on the same Hub.
fn rebase_under_cache(canonical: &Path, source_cache: &Path, dest_cache: &Path) -> PathBuf {
    // On macOS, `tempdir()` returns paths under `/var/folders/...` but
    // `canonicalize` resolves the `/var → /private/var` symlink. Resolve both
    // sides so `strip_prefix` lines up.
    let source = std::fs::canonicalize(source_cache).unwrap_or_else(|_| source_cache.to_path_buf());
    let relative = canonical.strip_prefix(&source).expect("canonical path was not under source cache");
    dest_cache.join(relative)
}

#[tokio::test]
async fn test_resume_completes_partial_download() {
    // Establish ground truth: download config.json once, capture the bytes
    // and the cache-relative layout (repo folder + etag-keyed blob path).
    let cache_a_dir = tempfile::tempdir().unwrap();
    let Some(client_a) = cache_scoped_client(cache_a_dir.path()) else {
        return;
    };
    let (owner, name) = TEST_MODEL_PARTS;

    let snapshot = client_a
        .model(owner, name)
        .download_file()
        .filename("config.json")
        .send()
        .await
        .unwrap();
    let blob_path = std::fs::canonicalize(&snapshot).expect("snapshot symlink should resolve");
    let full_bytes = std::fs::read(&blob_path).expect("blob should be readable");
    assert!(full_bytes.len() > 32, "test fixture too small to split meaningfully");

    // Pre-populate a fresh cache's `.incomplete` with the *correct* first half
    // of the file, then run the same download. A working resume implementation
    // will send `Range: bytes=N-` and only fetch the remaining bytes; a broken
    // one would truncate-and-restart, redownloading everything.
    let cache_b_dir = tempfile::tempdir().unwrap();
    let cache_b_blob = rebase_under_cache(&blob_path, cache_a_dir.path(), cache_b_dir.path());
    let cache_b_incomplete = PathBuf::from(format!("{}.incomplete", cache_b_blob.display()));
    std::fs::create_dir_all(cache_b_incomplete.parent().unwrap()).unwrap();

    let partial_size = full_bytes.len() / 2;
    let partial_bytes = &full_bytes[..partial_size];
    std::fs::write(&cache_b_incomplete, partial_bytes).unwrap();

    let client_b = cache_scoped_client(cache_b_dir.path()).unwrap();
    let handler = Arc::new(RecordingHandler::new());
    let resumed_snapshot = client_b
        .model(owner, name)
        .download_file()
        .filename("config.json")
        .progress(handler.clone())
        .send()
        .await
        .unwrap();

    // Correctness: the final blob must byte-equal the full known content,
    // regardless of how the patch got there.
    let resumed_bytes = std::fs::read(&resumed_snapshot).unwrap();
    assert_eq!(
        Sha256::digest(&resumed_bytes),
        Sha256::digest(&full_bytes),
        "resumed download must produce byte-identical content to a clean download"
    );

    // Resume actually fired: the first Started event reports a non-zero
    // initial offset – which is the patch's signal that it picked up the
    // existing bytes from `.incomplete`. A truncate-and-restart implementation
    // would start from 0.
    let events = handler.events();
    let first_started_offset = events
        .iter()
        .find_map(|e| match e {
            ProgressEvent::Download(DownloadEvent::Progress { files }) => files
                .iter()
                .find(|f| f.status == FileStatus::Started)
                .map(|f| f.bytes_completed),
            _ => None,
        })
        .expect("should have emitted a Started progress event");
    assert_eq!(
        first_started_offset, partial_size as u64,
        "Started event should report the resume offset, not 0 (truncate-and-restart)"
    );
}

#[tokio::test]
async fn test_force_download_clears_incomplete() {
    // Same setup: capture ground truth bytes + layout.
    let cache_a_dir = tempfile::tempdir().unwrap();
    let Some(client_a) = cache_scoped_client(cache_a_dir.path()) else {
        return;
    };
    let (owner, name) = TEST_MODEL_PARTS;

    let snapshot = client_a
        .model(owner, name)
        .download_file()
        .filename("config.json")
        .send()
        .await
        .unwrap();
    let blob_path = std::fs::canonicalize(&snapshot).unwrap();
    let full_bytes = std::fs::read(&blob_path).unwrap();

    // Populate a fresh cache's `.incomplete` with *garbage* of plausible size.
    // If `force_download(true)` did not clear it, the resume code would append
    // the upstream remainder to the garbage prefix and produce corrupt
    // content. The patched code wipes the partial before resume can kick in.
    let cache_b_dir = tempfile::tempdir().unwrap();
    let cache_b_blob = rebase_under_cache(&blob_path, cache_a_dir.path(), cache_b_dir.path());
    let cache_b_incomplete = PathBuf::from(format!("{}.incomplete", cache_b_blob.display()));
    std::fs::create_dir_all(cache_b_incomplete.parent().unwrap()).unwrap();

    let garbage = vec![0xFFu8; full_bytes.len() / 2];
    std::fs::write(&cache_b_incomplete, &garbage).unwrap();

    let client_b = cache_scoped_client(cache_b_dir.path()).unwrap();
    let resumed_snapshot = client_b
        .model(owner, name)
        .download_file()
        .filename("config.json")
        .force_download(true)
        .send()
        .await
        .unwrap();

    let resumed_bytes = std::fs::read(&resumed_snapshot).unwrap();
    assert_eq!(
        Sha256::digest(&resumed_bytes),
        Sha256::digest(&full_bytes),
        "force_download(true) must discard the partial and produce the canonical content"
    );
}
