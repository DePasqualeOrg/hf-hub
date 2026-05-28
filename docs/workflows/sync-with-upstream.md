# Syncing the patches branch with upstream main

This repo is an independently maintained fork of `huggingface/hf-hub`. The `swift-hf-api-patches` branch carries our patches on top of upstream's `main`. We keep it in sync by rebasing, not merging, so the branch always reads as a clean diff against whatever upstream currently is.

This document is the workflow an agent (or human) should follow when upstream has moved and we want to bring the new commits in.

## When to sync

Sync when `upstream/main` has moved and you want our patches to ride on top of the new tip. The routine is: fetch, look at the new commits, decide whether the changes warrant a rebase now. A drive-by CI bump on its own is usually not worth a sync; a substantive refactor of the public API surface or of code our patches touch is.

## 1. Pre-flight

Fetch upstream and inspect what's new:

```bash
git fetch upstream main
git log --oneline main..upstream/main
git log --stat main..upstream/main          # new upstream commits, with files
git log --stat upstream/main..swift-hf-api-patches  # our patches, with files
```

Compare the two file lists. Where they overlap is where conflicts are likely. The repository-handle internals under `hf-hub/src/repository/` are the historical hotspot — upstream iterates on them frequently, and many of our patches touch them too.

Read the new upstream commit messages. Two questions matter:

- Does any upstream commit refactor an API our patches build on? If so, expect at least one conflict whose resolution is to re-express our patch's intent on top of the new API shape.
- Does any upstream commit reintroduce or fix something one of our patches independently addressed? If so, drop the now-redundant patch instead of carrying it forward.

Create a timestamped backup before touching the branch, so you can diff against it later to filter pre-existing issues:

```bash
git checkout swift-hf-api-patches
git branch swift-hf-api-patches-backup-$(date +%Y%m%d-%H%M%S)
```

## 2. Rebase

```bash
git rebase upstream/main
```

For each conflict, resolve and continue with `git rebase --continue`. Two resolution patterns recur:

- **Upstream restructured something our patch also touched.** Take the upstream form as the new substrate and re-apply the *intent* of our patch on top of it. Do not blindly accept "theirs" or "ours" — the upstream side has the new API surface, our side has the behavior we wanted to add, and the right answer combines both.
- **Both sides added entries to the same import or re-export block.** Combine the entries, preserving alphabetical or grouped ordering as the surrounding code does.

When the right resolution is unclear, look at our patch's commit message — it almost always explains the *why* of the change, which makes it obvious how to translate that intent onto the new upstream shape. If a patch's intent no longer applies because upstream has independently solved the same problem, drop the patch with `git rebase --skip` instead of forcing it through.

## 3. Verify

Some of our patches depend on features in a sibling fork of `xet-core` that have not yet been released to crates.io. Building the rebased branch without a path override therefore fails dependency resolution — this is a pre-existing condition of the patches, not a rebase regression. To verify, add a temporary `[patch.crates-io]` block to the workspace `Cargo.toml`:

```toml
# Temporary — do NOT commit. Point this at your local xet-core checkout.
[patch.crates-io]
hf-xet = { path = "<absolute-path-to-local-xet-core>/xet_pkg" }
```

Then run the checks listed in [AGENTS.md](../../AGENTS.md) under "Formatting and Linting" and the test commands under "Testing":

```bash
cargo build -p hf-hub --all-features
cargo clippy -p hf-hub --all-features -- -D warnings
cargo +nightly fmt --check
cargo test -p hf-hub
```

### Filter pre-existing noise

Some clippy and fmt warnings predate any given sync. To tell which warnings are rebase regressions and which are pre-existing, run the same checks on the backup branch (with the same temporary `[patch.crates-io]` override applied there too) and diff the outputs. Anything reported on both branches is pre-existing.

A useful approach:

```bash
# Capture current branch output
cargo clippy -p hf-hub --all-features -- -D warnings > /tmp/clippy-post-rebase.txt 2>&1 || true

# Switch to backup, re-apply the same override, re-run
git checkout swift-hf-api-patches-backup-<timestamp>
# ... apply override, run again, write to /tmp/clippy-pre-rebase.txt ...

# Diff
diff /tmp/clippy-pre-rebase.txt /tmp/clippy-post-rebase.txt
```

If a lint fires on a line that the rebase produced or moved, fix it. Fold the fix into the patch that introduced the line, not as a separate follow-up commit (see "Folding fixups").

If a lint fires on code one of our patches has always had, leave it. Routine sync is not the time to clean up unrelated pre-existing issues; that belongs in a separate change.

When verification is finished, revert the temporary override:

```bash
git checkout HEAD -- Cargo.toml Cargo.lock
```

## 4. Folding fixups

If verification surfaced a small correction that belongs inside one of our existing patches, fold it in rather than appending a "fix" commit:

```bash
git add <files>
git commit --fixup=<sha-of-target-patch>
GIT_SEQUENCE_EDITOR=: git rebase -i --autosquash upstream/main
```

The fixup is squashed into the target commit and the branch keeps the same shape as before the sync, just with the upstream changes underneath. Each of our patches should still tell one coherent story when read alone.

## 5. Wrap up and push

Inspect the final state:

```bash
git log --oneline upstream/main..swift-hf-api-patches
```

The number and titles of our patches should match what you started with, modulo any patches dropped because upstream subsumed them and any fixups folded in.

The branch will have diverged from `origin/swift-hf-api-patches` after a rebase. Push with `--force-with-lease`, and confirm with the user before doing so — never force-push without explicit confirmation:

```bash
git push --force-with-lease origin swift-hf-api-patches
```

Keep the backup branch around until you've confirmed the new branch is healthy in whatever downstream consumer cares about it (Swift bindings, etc.). Delete it once you're sure:

```bash
git branch -D swift-hf-api-patches-backup-<timestamp>
```

## Pre-existing conditions to ignore

The following are known to be pre-existing on the patches branch and should not be "fixed" as part of a routine sync. Verify them against the backup before chasing:

- **`cargo build` / `cargo clippy` fail without a local `xet-core` path override.** Some patches use features in `hf-xet` that have not been released to crates.io. The temporary `[patch.crates-io]` in section 3 is the workaround for verification; the patches themselves remain valid because the dependent xet-core changes are expected to land upstream eventually.
- **`cargo +nightly fmt --check` reports formatting violations.** These predate individual syncs. Confirm with a diff against the backup; if the violations are the same on both sides, they're not this sync's problem.

If either condition stops being true — fmt becomes clean across the branch, or the dependent xet-core features ship to crates.io — update this section accordingly.
