# Preparing a public source release

The public release is intentionally code-only while weight/data redistribution permission is unknown. No aggregate company-test metrics are included.

```bash
python scripts/check_release.py
python scripts/export_public.py ../anpr-public-source
```

The destination must be new. The export uses an explicit allowlist, excluding datasets, local configs, run artifacts, private notes, historical configs, environment snapshots, weights and per-image reports. Review its contents before committing. This is not a generic secret scanner or a substitute for checking existing Git history.

For the existing repository, make a local branch from its current HEAD and replace its working tree with the reviewed source release. Preserve the old commit. Do not force-push or rewrite history merely to reorganize files. Removing an old tracked weight in the new commit does not remove it from Git history; its redistribution rights should also be checked.

Before public publication, the author still needs to choose the license for original code and verify that code created during company work can be shared. Keep the third-party PaddleOCR license. Dataset/weight rights are separate from source-code rights.

The existing public repository has a historical detector weight. If that file turns out to be confidential or unlicensed for sharing, address the existing publication/history explicitly; creating a new repository does not erase the old one.

Reference: [GitHub guidance on sensitive data in history](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).
