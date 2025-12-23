#!/usr/bin/env bash
set -euo pipefail

: "${GITHUB_TOKEN:?Set GITHUB_TOKEN first (export GITHUB_TOKEN=...)}"
OWNER="${OWNER:-B-A-M-N}"
REPO="${REPO:-OllamaRerank}"
BRANCH="${BRANCH:-main}"

need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing dependency: $1"; exit 1; }; }
need curl; need jq; need base64; need git

api() {
  curl -sS -H "Authorization: Bearer $GITHUB_TOKEN" \
          -H "Accept: application/vnd.github+json" \
          "$@"
}

mkblob() {
  local path="$1"
  local b64
  b64="$(base64 -w 0 "$path")"
  api -X POST "https://api.github.com/repos/$OWNER/$REPO/git/blobs" \
    -d "$(jq -n --arg content "$b64" --arg enc "base64" '{content:$content, encoding:$enc}')" \
    | jq -r '.sha'
}

echo "Repo: $OWNER/$REPO  Branch: $BRANCH"

BASE_COMMIT_SHA="$(api "https://api.github.com/repos/$OWNER/$REPO/git/ref/heads/$BRANCH" | jq -r '.object.sha')"
BASE_TREE_SHA="$(api "https://api.github.com/repos/$OWNER/$REPO/git/commits/$BASE_COMMIT_SHA" | jq -r '.tree.sha')"

echo "BASE_COMMIT_SHA=$BASE_COMMIT_SHA"
echo "BASE_TREE_SHA=$BASE_TREE_SHA"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# Modified tracked (staged + unstaged)
{ git diff --name-only HEAD; git diff --name-only --cached; } \
  | sed '/^\s*$/d' | sort -u > "$tmp/modified.txt"

# Untracked files (includes files inside untracked dirs)
git ls-files -o --exclude-standard \
  | sed '/^\s*$/d' | sort -u > "$tmp/untracked.txt"

cat "$tmp/modified.txt" "$tmp/untracked.txt" | sed '/^\s*$/d' | sort -u > "$tmp/files.txt"

if ! [ -s "$tmp/files.txt" ]; then
  echo "No changes detected (nothing to commit)."
  exit 0
fi

echo "Files to commit:"
sed 's/^/  - /' "$tmp/files.txt"

TREE_ITEMS="[]"
while IFS= read -r f; do
  [ -f "$f" ] || continue
  echo "Uploading: $f"
  sha="$(mkblob "$f")"
  echo "  blob_sha=$sha"
  TREE_ITEMS="$(jq -n \
    --argjson arr "$TREE_ITEMS" \
    --arg path "$f" \
    --arg sha "$sha" \
    '$arr + [{path:$path, mode:"100644", type:"blob", sha:$sha}]')"
done < "$tmp/files.txt"

NEW_TREE_SHA="$(api -X POST "https://api.github.com/repos/$OWNER/$REPO/git/trees" \
  -d "$(jq -n --arg base "$BASE_TREE_SHA" --argjson items "$TREE_ITEMS" '{base_tree:$base, tree:$items}')" \
  | jq -r '.sha')"

echo "NEW_TREE_SHA=$NEW_TREE_SHA"

COMMIT_MSG="${COMMIT_MSG:-Update via API}"
NEW_COMMIT_SHA="$(api -X POST "https://api.github.com/repos/$OWNER/$REPO/git/commits" \
  -d "$(jq -n --arg msg "$COMMIT_MSG" --arg tree "$NEW_TREE_SHA" --arg parent "$BASE_COMMIT_SHA" \
    '{message:$msg, tree:$tree, parents:[$parent]}')" \
  | jq -r '.sha')"

echo "NEW_COMMIT_SHA=$NEW_COMMIT_SHA"

api -X PATCH "https://api.github.com/repos/$OWNER/$REPO/git/refs/heads/$BRANCH" \
  -d "$(jq -n --arg sha "$NEW_COMMIT_SHA" '{sha:$sha, force:false}')" \
  | jq '{ref: .ref, new_sha: .object.sha}'
