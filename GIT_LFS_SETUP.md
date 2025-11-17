# Git LFS Setup Instructions

The repository has a large checkpoint file (801.65 MB) that exceeds GitHub's 100 MB limit. Follow these steps to set up Git LFS and fix the issue.

## Step 1: Install Git LFS (if not already installed)

**Why?** Git LFS is a separate tool that needs to be installed. It's not part of the standard Git installation.

### Option A: Install via Conda (Recommended for cluster systems, no sudo required)

```bash
# Make sure your conda environment is activated
conda install -c conda-forge git-lfs -y

# Or use the provided script
bash install_git_lfs.sh
```

### Option B: System-wide installation (requires sudo/admin access)

```bash
# On Ubuntu/Debian
sudo apt-get install git-lfs

# On macOS
brew install git-lfs

# On CentOS/RHEL
sudo yum install git-lfs
```

### Option C: Manual installation (if you don't have sudo or conda)

```bash
# Download and install manually
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

# Or download binary from: https://github.com/git-lfs/git-lfs/releases
```

## Step 2: Initialize Git LFS

```bash
cd /cluster/tufts/liulab/yiwan01/SpecBridge
git lfs install
```

## Step 3: Track .pt files with Git LFS

The `.gitattributes` file is already created. Verify it:

```bash
cat .gitattributes
```

You should see:
```
*.pt filter=lfs diff=lfs merge=lfs -text
```

If needed, you can also run:
```bash
git lfs track "*.pt"
```

## Step 4: Remove the large file from git history

Since the file is already committed, you need to remove it from history. Choose one of these options:

### Option A: If the file is only in the most recent commit(s) and NOT pushed yet

```bash
# Check how many commits back the file was added
git log --oneline --all -- "runs/specbridge_align_chemberta_pub_v3g_nosample/ckpt_000400.pt"

# Reset to before the file was added (adjust HEAD~N as needed)
git reset HEAD~1  # or HEAD~2, etc.

# Now add the file with git-lfs
git add .gitattributes
git add runs/specbridge_align_chemberta_pub_v3g_nosample/ckpt_000400.pt

# Verify it's using git-lfs (should show "Git LFS" in the output)
git lfs ls-files

# Commit
git commit -m "Migrate checkpoint files to Git LFS"
```

### Option B: If the file is in older commits or already pushed (use git filter-branch)

```bash
# Remove the file from all commits
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch runs/specbridge_align_chemberta_pub_v3g_nosample/ckpt_000400.pt' \
  --prune-empty --tag-name-filter cat -- --all

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Now add the file back with git-lfs
git add .gitattributes
git add runs/specbridge_align_chemberta_pub_v3g_nosample/ckpt_000400.pt
git commit -m "Add checkpoint file with Git LFS"
```

### Option C: Use BFG Repo-Cleaner (faster, recommended for large repos)

```bash
# Download BFG from https://rtyley.github.io/bfg-repo-cleaner/
# Then run:
java -jar bfg.jar --delete-files ckpt_000400.pt
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Add the file back with git-lfs
git add .gitattributes
git add runs/specbridge_align_chemberta_pub_v3g_nosample/ckpt_000400.pt
git commit -m "Add checkpoint file with Git LFS"
```

## Step 5: Verify Git LFS is working

```bash
# Check that the file is tracked by git-lfs
git lfs ls-files

# You should see the .pt file listed
```

## Step 6: Push to remote

```bash
# If you used filter-branch or BFG and the file was already pushed, you'll need to force push
# WARNING: Only do this if you're sure no one else has pulled your changes
git push origin master --force

# Otherwise, normal push is fine
git push origin master
```

## Troubleshooting

- If you get "git-lfs: command not found", make sure git-lfs is installed and in your PATH
- If the file still shows as large, make sure `.gitattributes` is committed and the file was added after git-lfs was initialized
- If you need to check file sizes: `git lfs ls-files` shows files tracked by LFS, `git ls-files` shows all tracked files

## Note about .gitignore

The `runs/` directory is in `.gitignore`, so checkpoint files there won't be tracked by default. If you want to track specific checkpoint files, you can either:
1. Force add them: `git add -f runs/path/to/file.pt`
2. Or add an exception in `.gitignore`: `!runs/**/*.pt` (but this will track ALL .pt files in runs/)

