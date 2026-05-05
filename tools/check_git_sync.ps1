param()

$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "== $Title =="
}

Write-Section "Repository"
$repoRoot = git rev-parse --show-toplevel
Write-Host "Root: $repoRoot"

Write-Section "Branch"
$branch = git branch --show-current
if (-not $branch) {
    $branch = "(detached HEAD)"
}
Write-Host "Current branch: $branch"

Write-Section "Worktree"
$statusLines = git status --short
if ($statusLines) {
    Write-Host "Worktree has uncommitted changes:"
    $statusLines | ForEach-Object { Write-Host "  $_" }
} else {
    Write-Host "Worktree is clean."
}

Write-Section "Remotes"
$fetchUrl = git remote get-url origin
$pushUrls = git remote get-url --push --all origin
Write-Host "origin fetch: $fetchUrl"
Write-Host "origin push URLs:"
$pushUrls | ForEach-Object { Write-Host "  $_" }
if (($pushUrls | Measure-Object).Count -lt 2) {
    Write-Warning "origin has fewer than two push URLs. GitHub and GitLab may not both be updated."
}

Write-Section "Ahead / Behind"
$upstream = ""
try {
    $upstream = git rev-parse --abbrev-ref --symbolic-full-name "@{u}" 2>$null
} catch {
    $upstream = ""
}

if ($upstream) {
    $counts = git rev-list --left-right --count "$upstream...HEAD"
    $parts = $counts -split "\s+"
    $behind = [int]$parts[0]
    $ahead = [int]$parts[1]
    Write-Host "Upstream: $upstream"
    Write-Host "Ahead: $ahead"
    Write-Host "Behind: $behind"
} else {
    Write-Warning "No upstream configured for the current branch."
}

Write-Section "Reminder"
Write-Host "Ready-to-share work should be committed and pushed through origin so both push URLs stay in sync."
