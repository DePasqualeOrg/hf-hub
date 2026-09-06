use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::extract::{Request, State};
use axum::response::Response;
use sha2::{Digest, Sha256};
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use xet::xet_session::{Sha256Policy, XetSessionBuilder};
use xet_client::cas_client::{LocalServer, LocalServerConfig};
use xet_runtime::config::XetConfig;
use xet_runtime::core::XetContext;

use crate::HFClient;
use crate::progress::{DownloadEvent, FileProgress, FileStatus, ProgressEvent, ProgressHandler};

const COMMIT: &str = "1111111111111111111111111111111111111111";

#[derive(Default)]
struct Events {
    files: Mutex<Vec<FileProgress>>,
    total: Mutex<Option<u64>>,
    initial: Mutex<Option<Vec<FileProgress>>>,
    changed: tokio::sync::Notify,
}

impl ProgressHandler for Events {
    fn on_progress(&self, event: &ProgressEvent) {
        if let ProgressEvent::Download(DownloadEvent::Progress { files }) = event {
            self.initial.lock().unwrap().get_or_insert_with(|| files.clone());
            self.files.lock().unwrap().extend(files.clone());
            self.changed.notify_one();
        } else if let ProgressEvent::Download(DownloadEvent::Start { total_bytes, .. }) = event {
            *self.total.lock().unwrap() = Some(*total_bytes);
        }
    }
}

struct ServerState {
    upstream: String,
    endpoint: String,
    hash: String,
    etag: String,
    size: usize,
    omit_size: AtomicBool,
    hold_payload: AtomicBool,
    payload_permits: tokio::sync::Semaphore,
    client: reqwest::Client,
    ranges: Mutex<Vec<String>>,
    payload_bytes: Mutex<usize>,
}

async fn handle(State(state): State<Arc<ServerState>>, request: Request) -> Response {
    let path = request.uri().path();
    if path.ends_with("/telemetry") {
        return Response::new(Body::empty());
    }
    if path.ends_with("/config.json") {
        return Response::builder()
            .header("etag", COMMIT)
            .header("x-repo-commit", COMMIT)
            .header("content-length", "2")
            .body(if request.method() == http::Method::HEAD {
                Body::empty()
            } else {
                Body::from("{}")
            })
            .unwrap();
    }
    if path.contains("/resolve/") {
        return Response::builder()
            .header("etag", &state.etag)
            .header("x-repo-commit", COMMIT)
            .header("x-xet-hash", &state.hash)
            .header(
                "x-linked-size",
                if state.omit_size.load(Ordering::Relaxed) {
                    "invalid".to_string()
                } else {
                    state.size.to_string()
                },
            )
            .body(Body::empty())
            .unwrap();
    }
    if path.contains("xet-read-token") {
        return Response::new(Body::from(
            serde_json::json!({
                "accessToken": "fixture-only", "exp": 4102444800_u64, "casUrl": state.endpoint,
            })
            .to_string(),
        ));
    }
    if path.contains("/tree/") {
        return Response::new(Body::from(
            serde_json::json!([
                {"type": "file", "oid": state.etag, "size": state.size, "path": "model.bin"},
                {"type": "file", "oid": state.etag, "size": state.size, "path": "alias.bin"},
            {"type": "file", "oid": COMMIT, "size": 2, "path": "config.json"},
            ])
            .to_string(),
        ));
    }
    if path.starts_with("/api/") {
        return Response::new(Body::from(serde_json::json!({"sha": COMMIT}).to_string()));
    }
    if path.contains("/reconstructions/") {
        state
            .ranges
            .lock()
            .unwrap()
            .push(request.headers().get("range").map_or("", |v| v.to_str().unwrap()).to_string());
    }
    let response = state
        .client
        .get(format!("{}{}", state.upstream, request.uri()))
        .headers(request.headers().clone())
        .send()
        .await
        .unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let body = response.bytes().await.unwrap();
    if path.contains("fetch_term") {
        *state.payload_bytes.lock().unwrap() += body.len();
    }
    let body = if path.contains("fetch_term") && state.hold_payload.load(Ordering::Relaxed) {
        // Deliver part of every term, then wait for the test to allow further network bytes.
        Body::from_stream(futures::stream::unfold((body, state, true), |(mut bytes, state, first)| async move {
            if bytes.is_empty() {
                return None;
            }
            if !first && let Ok(permit) = state.payload_permits.acquire().await {
                permit.forget();
            }
            let chunk = bytes.split_to(bytes.len().min(16 * 1024));
            Some((Ok::<_, std::io::Error>(chunk), (bytes, state, false)))
        }))
    } else {
        Body::from(body)
    };
    let mut response = Response::builder().status(status).body(body).unwrap();
    *response.headers_mut() = headers;
    response
}

struct Fixture {
    temp: TempDir,
    state: Arc<ServerState>,
    tasks: Vec<JoinHandle<()>>,
    data: Vec<u8>,
}

impl Fixture {
    async fn new(mut seed: u64) -> Self {
        let temp = tempfile::tempdir().unwrap();
        let mut config = XetConfig::new();
        config.data.cache_root = temp.path().join("xet").to_string_lossy().into_owned();
        config.xorb.simulation_max_bytes = Some(xet_runtime::utils::ByteSize::from("1mb"));
        let context = XetContext::with_config(config.clone()).unwrap();
        // The simulation server binds its own socket, so reserve an available port first.
        let reservation = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = reservation.local_addr().unwrap().port();
        let server = LocalServer::new(
            context,
            LocalServerConfig {
                port,
                host: "127.0.0.1".into(),
                in_memory: false,
                data_directory: temp.path().join("cas"),
            },
        )
        .await
        .unwrap();
        drop(reservation);
        let cas_task = tokio::spawn(async move {
            server.run().await.unwrap();
        });
        let upstream = format!("http://127.0.0.1:{port}");
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if client.get(format!("{upstream}/health")).send().await.is_ok() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        // Incompressible, deterministic bytes make transferred payload size meaningful.
        let data: Vec<u8> = (0..8 * 1024 * 1024)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed as u8
            })
            .collect();
        let session = XetSessionBuilder::new_with_config(config).build().unwrap();
        let commit = session
            .new_upload_commit()
            .unwrap()
            .with_endpoint(&upstream)
            .build()
            .await
            .unwrap();
        let upload = commit
            .upload_bytes(data.clone(), Sha256Policy::Compute, Some("model.bin".into()))
            .await
            .unwrap();
        let metadata = upload.finalize_ingestion().await.unwrap();
        commit.commit().await.unwrap();
        let hash = metadata.xet_info.hash().to_string();
        let etag = Sha256::digest(&data).iter().map(|b| format!("{b:02x}")).collect();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let state = Arc::new(ServerState {
            endpoint: format!("http://{}", listener.local_addr().unwrap()),
            upstream,
            hash,
            etag,
            size: data.len(),
            omit_size: AtomicBool::new(false),
            hold_payload: AtomicBool::new(false),
            payload_permits: tokio::sync::Semaphore::new(0),
            client,
            ranges: Mutex::default(),
            payload_bytes: Mutex::default(),
        });
        let router = Router::new().fallback(handle).with_state(state.clone());
        let hub_task = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        Self {
            temp,
            state,
            tasks: vec![cas_task, hub_task],
            data,
        }
    }

    fn client(&self) -> HFClient {
        HFClient::builder()
            .endpoint(&self.state.endpoint)
            .token("fixture-only")
            .cache_dir(self.temp.path().join("hub"))
            .build()
            .unwrap()
    }

    fn partial(&self) -> PathBuf {
        let blobs = self.temp.path().join("hub/models--fixture--resume/blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        blobs.join(format!("{}.incomplete", self.state.etag))
    }

    async fn shutdown(mut self) {
        for task in self.tasks.drain(..) {
            task.abort();
            assert!(task.await.unwrap_err().is_cancelled());
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cached_xet_resumes_decoded_bytes_and_checks_integrity() {
    tokio::time::timeout(Duration::from_secs(90), async {
        let fixture = Fixture::new(19).await;
        let client = fixture.client();
        let repo = client.model("fixture", "resume");
        let offset = fixture.data.len() * 3 / 4 + 123;
        std::fs::write(fixture.partial(), &fixture.data[..offset]).unwrap();
        fixture.state.omit_size.store(true, Ordering::Relaxed);
        assert!(repo.download_file().filename("model.bin").send().await.is_err());
        assert_eq!(std::fs::read(fixture.partial()).unwrap(), fixture.data[..offset]);
        fixture.state.omit_size.store(false, Ordering::Relaxed);
        let events = Arc::new(Events::default());
        let path = repo
            .download_file()
            .filename("model.bin")
            .progress(events.clone())
            .send()
            .await
            .unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), fixture.data);
        assert!(!fixture.partial().exists());
        assert!(
            events
                .files
                .lock()
                .unwrap()
                .iter()
                .any(|p| p.status == FileStatus::Started && p.bytes_completed == offset as u64)
        );
        assert!(
            fixture
                .state
                .ranges
                .lock()
                .unwrap()
                .iter()
                .any(|r| r.starts_with(&format!("bytes={offset}-")))
        );
        let transferred = *fixture.state.payload_bytes.lock().unwrap();
        assert!(transferred > 0 && transferred < fixture.data.len() / 2, "payload bytes: {transferred}");

        // A forced download must ignore a saved prefix, even when a complete blob exists.
        std::fs::write(fixture.partial(), &fixture.data[..offset]).unwrap();
        fixture.state.ranges.lock().unwrap().clear();
        repo.download_file()
            .filename("model.bin")
            .force_download(true)
            .send()
            .await
            .unwrap();
        assert!(fixture.state.ranges.lock().unwrap().iter().any(|r| r.starts_with("bytes=0-")));

        // Corrupt retained bytes must never become a completed cache entry.
        let blob = fixture.partial().with_extension("");
        std::fs::remove_file(&blob).unwrap();
        std::fs::write(fixture.partial(), vec![0_u8; offset]).unwrap();
        assert!(repo.download_file().filename("model.bin").send().await.is_err());
        assert!(!blob.exists());
        assert_eq!(std::fs::metadata(fixture.partial()).unwrap().len(), 0);
        let path = repo.download_file().filename("model.bin").send().await.unwrap();
        assert_eq!(std::fs::read(path).unwrap(), fixture.data);
        // A fully written partial is validated and promoted without fetching payload.
        std::fs::rename(&blob, fixture.partial()).unwrap();
        fixture.state.ranges.lock().unwrap().clear();
        repo.download_file().filename("model.bin").send().await.unwrap();
        assert!(fixture.state.ranges.lock().unwrap().is_empty());

        // Oversized partials restart safely.
        std::fs::remove_file(&blob).unwrap();
        std::fs::File::create(fixture.partial())
            .unwrap()
            .set_len(fixture.data.len() as u64 + 1)
            .unwrap();
        let path = repo.download_file().filename("model.bin").send().await.unwrap();
        assert_eq!(std::fs::read(path).unwrap(), fixture.data);

        // Snapshot aliases share a blob; acquiring every lock at once would deadlock.
        std::fs::remove_file(&blob).unwrap();
        std::fs::write(fixture.partial(), &fixture.data[..offset]).unwrap();
        fixture.state.ranges.lock().unwrap().clear();
        let (a, b) = tokio::join!(
            repo.snapshot_download().max_workers(2).send(),
            repo.download_file().filename("model.bin").send(),
        );
        assert_eq!(std::fs::read(a.unwrap().join("alias.bin")).unwrap(), fixture.data);
        assert_eq!(std::fs::read(b.unwrap()).unwrap(), fixture.data);
        assert_eq!(fixture.state.ranges.lock().unwrap().len(), 1);

        // A completed HTTP file and a partial Xet blob share the snapshot total.
        std::fs::remove_file(&blob).unwrap();
        std::fs::write(fixture.partial(), &fixture.data[..offset]).unwrap();
        let events = Arc::new(Events::default());
        let snapshot = repo
            .snapshot_download()
            .max_workers(2)
            .progress(events.clone())
            .send()
            .await
            .unwrap();
        assert_eq!(std::fs::read(snapshot.join("config.json")).unwrap(), b"{}");
        assert_eq!(*events.total.lock().unwrap(), Some(2 * fixture.data.len() as u64 + 2));
        let initial = events.initial.lock().unwrap().clone().unwrap();
        assert_eq!(initial.len(), 3);
        assert_eq!(initial.iter().map(|p| p.bytes_completed).sum::<u64>(), 2 * offset as u64 + 2);
        assert!(events.files.lock().unwrap().iter().any(|p| p.filename == "config.json"
            && p.status == FileStatus::Complete
            && p.bytes_completed == 2
            && p.total_bytes == 2));
        assert!(
            events
                .files
                .lock()
                .unwrap()
                .iter()
                .any(|p| p.filename == "model.bin" && p.bytes_completed == offset as u64)
        );

        // Cancellation drops the writer before releasing its blob lock.
        std::fs::remove_file(&blob).unwrap();
        let (start, ready) = tokio::sync::oneshot::channel();
        let cancellation = Arc::new(AbortAfterWrite(Mutex::new(None), fixture.partial()));
        let handler = cancellation.clone();
        let cancelled_repo = repo.clone();
        let task = tokio::spawn(async move {
            ready.await.unwrap();
            cancelled_repo
                .download_file()
                .filename("model.bin")
                .progress(handler)
                .send()
                .await
        });
        *cancellation.0.lock().unwrap() = Some(task.abort_handle());
        start.send(()).unwrap();
        assert!(task.await.unwrap_err().is_cancelled());
        let prefix = std::fs::read(fixture.partial()).unwrap();
        assert!(!prefix.is_empty() && prefix.len() < fixture.data.len());
        assert_eq!(prefix, fixture.data[..prefix.len()]);
        let path = repo.download_file().filename("model.bin").send().await.unwrap();
        assert_eq!(std::fs::read(path).unwrap(), fixture.data);

        // Abrupt process exit leaves real decoded output; subsequent processes keep it.
        std::fs::remove_file(&blob).unwrap();
        let mut retained = 0;
        for _ in 0..2 {
            let child = tokio::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "xet::resume_tests::cached_xet_interrupted_child",
                    "--ignored",
                    "--nocapture",
                ])
                .env("HF_RESUME_TEST_ENDPOINT", &fixture.state.endpoint)
                .env("HF_RESUME_TEST_CACHE", fixture.temp.path().join("hub"))
                .env("HF_RESUME_TEST_PARTIAL", fixture.partial())
                .kill_on_drop(true)
                .output()
                .await
                .unwrap();
            assert_eq!(child.status.code(), Some(87), "child output: {}", String::from_utf8_lossy(&child.stderr));
            let bytes = std::fs::read(fixture.partial()).unwrap();
            assert!(bytes.len() > retained && bytes.len() < fixture.data.len());
            assert_eq!(bytes, fixture.data[..bytes.len()]);
            retained = bytes.len();
        }
        fixture.state.ranges.lock().unwrap().clear();
        let path = fixture
            .client()
            .model("fixture", "resume")
            .download_file()
            .filename("model.bin")
            .send()
            .await
            .unwrap();
        assert_eq!(std::fs::read(path).unwrap(), fixture.data);
        assert!(
            fixture
                .state
                .ranges
                .lock()
                .unwrap()
                .iter()
                .any(|r| r.starts_with(&format!("bytes={retained}-")))
        );
        fixture.shutdown().await;
    })
    .await
    .expect("local Xet resume test timed out");
}

struct ExitAfterWrite {
    partial: PathBuf,
    retained: u64,
}
impl ProgressHandler for ExitAfterWrite {
    fn on_progress(&self, event: &ProgressEvent) {
        if let ProgressEvent::Download(DownloadEvent::Progress { files }) = event
            && files
                .iter()
                .any(|f| f.status == FileStatus::InProgress && f.bytes_completed > 0)
            && std::fs::metadata(&self.partial).is_ok_and(|m| m.len() > self.retained)
        {
            std::process::exit(87);
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "subprocess helper; exercised by the parent resume regression"]
async fn cached_xet_interrupted_child() {
    let endpoint = std::env::var("HF_RESUME_TEST_ENDPOINT").unwrap();
    assert!(endpoint.starts_with("http://127.0.0.1:"));
    let client = HFClient::builder()
        .endpoint(endpoint)
        .token("fixture-only")
        .cache_dir(std::env::var("HF_RESUME_TEST_CACHE").unwrap())
        .build()
        .unwrap();
    client
        .model("fixture", "resume")
        .download_file()
        .filename("model.bin")
        .progress({
            let partial = PathBuf::from(std::env::var("HF_RESUME_TEST_PARTIAL").unwrap());
            let retained = std::fs::metadata(&partial).map_or(0, |m| m.len());
            ExitAfterWrite { partial, retained }
        })
        .send()
        .await
        .unwrap();
    panic!("child completed without interruption");
}

struct AbortAfterWrite(Mutex<Option<tokio::task::AbortHandle>>, PathBuf);
impl ProgressHandler for AbortAfterWrite {
    fn on_progress(&self, event: &ProgressEvent) {
        if let ProgressEvent::Download(DownloadEvent::Progress { files }) = event
            && files
                .iter()
                .any(|f| f.status == FileStatus::InProgress && f.bytes_completed > 0)
            && std::fs::metadata(&self.1).is_ok_and(|m| m.len() > 0)
        {
            self.0.lock().unwrap().as_ref().unwrap().abort();
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cached_xet_changed_content_uses_a_different_partial() {
    tokio::time::timeout(Duration::from_secs(90), async {
        let old = Fixture::new(19).await;
        let new = Fixture::new(23).await;
        let prefix = &old.data[..123456];
        std::fs::write(old.partial(), prefix).unwrap();
        let client = HFClient::builder()
            .endpoint(&new.state.endpoint)
            .token("fixture-only")
            .cache_dir(old.temp.path().join("hub"))
            .build()
            .unwrap();
        let events = Arc::new(Events::default());
        let path = client
            .model("fixture", "resume")
            .download_file()
            .filename("model.bin")
            .progress(events.clone())
            .send()
            .await
            .unwrap();
        assert_eq!(std::fs::read(path).unwrap(), new.data);
        assert_eq!(std::fs::read(old.partial()).unwrap(), prefix);
        assert!(
            events
                .files
                .lock()
                .unwrap()
                .iter()
                .any(|p| p.status == FileStatus::Started && p.bytes_completed == 0)
        );
        old.shutdown().await;
        new.shutdown().await;
    })
    .await
    .expect("changed-content regression timed out");
}

/// Keeps the same local CAS fixture alive while a foreign-language client exercises the binary.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "fixture process for the Swift integration test; requires HF_RESUME_FIXTURE"]
async fn cached_xet_ffi_fixture() {
    let descriptor = std::env::var("HF_RESUME_FIXTURE").unwrap();
    let fixture = Fixture::new(19).await;
    fixture
        .client()
        .model("fixture", "resume")
        .download_file()
        .filename("config.json")
        .send()
        .await
        .unwrap();
    let offset = fixture.data.len() * 3 / 4 + 123;
    std::fs::write(fixture.partial(), &fixture.data[..offset]).unwrap();
    let expected = fixture.temp.path().join("expected.bin");
    std::fs::write(&expected, &fixture.data).unwrap();
    std::fs::write(
        descriptor,
        serde_json::json!({
            "endpoint": fixture.state.endpoint,
            "cacheDirectory": fixture.temp.path().join("hub"),
            "expectedFile": expected,
            "retainedBytes": 2 * offset + 2,
            "totalBytes": 2 * fixture.data.len() + 2,
        })
        .to_string(),
    )
    .unwrap();
    eprintln!("HF_RESUME_FIXTURE_READY");
    tokio::task::spawn_blocking(|| std::io::stdin().read_line(&mut String::new()))
        .await
        .unwrap()
        .unwrap();
    fixture.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cached_xet_reports_network_progress_before_output() {
    tokio::time::timeout(Duration::from_secs(30), async {
        for retained in [0, 6 * 1024 * 1024 + 123] {
            let fixture = Fixture::new(47).await;
            let partial = fixture.partial();
            std::fs::write(&partial, &fixture.data[..retained]).unwrap();
            fixture.state.hold_payload.store(true, Ordering::Relaxed);
            let events = Arc::new(Events::default());
            let repo = fixture.client().model("fixture", "resume");
            let download = repo.download_file().filename("model.bin").progress(events.clone()).send();
            let observe = async {
                let mut previous = retained as u64;
                for _ in 0..3 {
                    loop {
                        let changed = events.changed.notified();
                        let latest = events
                            .files
                            .lock()
                            .unwrap()
                            .iter()
                            .map(|p| p.bytes_completed)
                            .max()
                            .unwrap_or(0);
                        if latest > previous {
                            assert!(latest < fixture.data.len() as u64);
                            assert_eq!(std::fs::metadata(&partial).unwrap().len(), retained as u64);
                            previous = latest;
                            break;
                        }
                        changed.await;
                    }
                    fixture.state.payload_permits.add_permits(1);
                }
                fixture.state.payload_permits.close();
            };
            let (result, ()) = tokio::join!(download, observe);
            assert_eq!(std::fs::read(result.unwrap()).unwrap(), fixture.data);
            let reports = events.files.lock().unwrap().clone();
            assert_eq!(reports.first().unwrap().bytes_completed, retained as u64);
            assert!(reports.windows(2).all(|p| p[0].bytes_completed <= p[1].bytes_completed));
            assert!(
                reports
                    .iter()
                    .filter(|p| p.status == FileStatus::InProgress)
                    .all(|p| p.bytes_completed < p.total_bytes)
            );
            assert_eq!(reports.last().unwrap().status, FileStatus::Complete);
            fixture.shutdown().await;
        }
    })
    .await
    .expect("network progress stalled before Xet produced output");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cached_xet_network_progress_is_not_a_resume_offset() {
    tokio::time::timeout(Duration::from_secs(30), async {
        let fixture = Fixture::new(53).await;
        let retained = 123456;
        let partial = fixture.partial();
        std::fs::write(&partial, &fixture.data[..retained]).unwrap();
        fixture.state.hold_payload.store(true, Ordering::Relaxed);
        let events = Arc::new(Events::default());
        let repo = fixture.client().model("fixture", "resume");
        {
            let download = repo.download_file().filename("model.bin").progress(events.clone()).send();
            let network_received = async {
                loop {
                    let changed = events.changed.notified();
                    if events.files.lock().unwrap().iter().any(|p| p.bytes_completed > retained as u64) {
                        break;
                    }
                    changed.await;
                }
            };
            tokio::select! {
                result = download => panic!("download finished while payload was withheld: {result:?}"),
                () = network_received => {},
            }
        }
        assert_eq!(std::fs::read(&partial).unwrap(), fixture.data[..retained]);
        fixture.state.payload_permits.close();
        fixture.state.hold_payload.store(false, Ordering::Relaxed);
        fixture.state.ranges.lock().unwrap().clear();
        let result = repo.download_file().filename("model.bin").send().await.unwrap();
        assert_eq!(std::fs::read(result).unwrap(), fixture.data);
        assert!(
            fixture
                .state
                .ranges
                .lock()
                .unwrap()
                .iter()
                .any(|r| r.starts_with(&format!("bytes={retained}-")))
        );
        fixture.shutdown().await;
    })
    .await
    .expect("cancelled network progress did not resume from the saved prefix");
}
