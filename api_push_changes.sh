#!/usr/bin/env bash
# api_commit_staged.sh
#
# Commits ONLY staged changes (git index) to GitHub via REST API:
# - Reads staged changes via: git diff --cached --name-status -z
# - Uploads blobs for added/modified files
# - Adds delete entries for removed files
# - Handles renames as delete(old) + add(new)
# - Creates a new tree on top of the branch HEAD tree
# - Creates a commit and updates refs/heads/$BRANCH
#
# Usage:
#   export GITHUB_TOKEN="..."
#   export OWNER="B-A-M-N"
#   export REPO="OllamaRerank"
#   export BRANCH="main"
#   export COMMIT_MSG="Update via API"
#   git add -A
#   ./api_commit_staged.sh
#
# Notes:
# - Requires: curl jq base64 git
# - Binary detection: NUL-byte heuristic
# - Executable bit preserved (100755) for files that are +x locally

set -euo pipefail

: "${GITHUB_TOKEN:?Set GITHUB_TOKEN first (export GITHUB_TOKEN=...)}"
OWNER="${OWNER:-B-A-M-N}"
REPO="${REPO:-OllamaRerank}"
BRANCH="${BRANCH:-main}"
COMMIT_MSG="${COMMIT_MSG:-Update via API (staged)}"

need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing dependency: $1" >&2; exit 1; }; }
need curl; need jq; need base64; need git

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "ERROR: not inside a git repo" >&2; exit 1; }

# Require staged changes
if git diff --cached --quiet; then
  echo "ERROR: No staged changes. Stage files first (git add ...)." >&2
  exit 1
fi

API_ROOT="https://api.github.com"
AUTH_HEADER="Authorization: token $GITHUB_TOKEN"
ACCEPT_HEADER="Accept: application/vnd.github+json"
CT_HEADER="Content-Type: application/json"

api() {
  # For GETs (no JSON body required)
  curl -sS -H "$AUTH_HEADER" -H "$ACCEPT_HEADER" "$@"
}

api_json() {
  # For POST/PATCH with JSON body
  curl -sS -H "$AUTH_HEADER" -H "$ACCEPT_HEADER" -H "$CT_HEADER" "$@"
}

req() {
  local name="$1" val="$2"
  if [[ -z "$val" || "$val" == "null" ]]; then
    echo "ERROR: $name is empty/null. API call likely failed." >&2
    exit 1
  fi
}

echo "Repo: $OWNER/$REPO  Branch: $BRANCH"
echo "Commit message: $COMMIT_MSG"

# Fetch HEAD commit SHA for the branch
REF_JSON="$(api "$API_ROOT/repos/$OWNER/$REPO/git/ref/heads/$BRANCH")"
BASE_COMMIT_SHA="$(echo "$REF_JSON" | jq -r '.object.sha')"
if [[ -z "$BASE_COMMIT_SHA" || "$BASE_COMMIT_SHA" == "null" ]]; then
  echo "ERROR: Could not read branch ref. Full response:" >&2
  echo "$REF_JSON" | jq . >&2
  exit 1
fi

# Fetch base tree SHA from HEAD commit
COMMIT_JSON="$(api "$API_ROOT/repos/$OWNER/$REPO/git/commits/$BASE_COMMIT_SHA")"
BASE_TREE_SHA="$(echo "$COMMIT_JSON" | jq -r '.tree.sha')"
if [[ -z "$BASE_TREE_SHA" || "$BASE_TREE_SHA" == "null" ]]; then
  echo "ERROR: Could not read base commit/tree. Full response:" >&2
  echo "$COMMIT_JSON" | jq . >&2
  exit 1
fi

echo "BASE_COMMIT_SHA=$BASE_COMMIT_SHA"
echo "BASE_TREE_SHA=$BASE_TREE_SHA"

# Create a blob for a path; echoes: "<mode> <sha>"
create_blob_for_path() {
  local path="$1"
  [[ -f "$path" ]] || { echo "ERROR: file missing on disk: $path" >&2; exit 1; }

  local mode="100644"
  [[ -x "$path" ]] && mode="100755"

  local blob_json blob_sha

  # NUL-byte heuristic for binary
  if (LC_ALL=C grep -qU $'\x00' "$path" 2>/dev/null); then
    local b64
    b64="$(base64 -w 0 < "$path")"
    blob_json="$(api_json -X POST "$API_ROOT/repos/$OWNER/$REPO/git/blobs" \
      -d "$(jq -n --arg content "$b64" '{content:$content, encoding:"base64"}')")"
  else
    # Safely JSON-escape full file
    local content_json
    content_json="$(jq -Rs . < "$path")"
    blob_json="$(api_json -X POST "$API_ROOT/repos/$OWNER/$REPO/git/blobs" \
      -d "{\"content\":${content_json},\"encoding\":\"utf-8\"}")"
  fi

  blob_sha="$(echo "$blob_json" | jq -r '.sha')"
  if [[ -z "$blob_sha" || "$blob_sha" == "null" ]]; then
    echo "ERROR: failed to create blob for $path. Full response:" >&2
    echo "$blob_json" | jq . >&2
    exit 1
  fi

  echo "$mode $blob_sha"
}

# Build tree entries
TREE_ITEMS='[]'

add_tree_blob() {
  local path="$1" mode="$2" sha="$3"
  TREE_ITEMS="$(echo "$TREE_ITEMS" | jq \
    --arg path "$path" \
    --arg mode "$mode" \
    --arg sha "$sha" \
    '. + [{path:$path, mode:$mode, type:"blob", sha:$sha}]')"
}

add_tree_delete() {
  local path="$1"
  TREE_ITEMS="$(echo "$TREE_ITEMS" | jq \
    --arg path "$path" \
    '. + [{path:$path, mode:"100644", type:"blob", sha:null}]')"
}

# Parse staged changes NUL-delimited:
# - M\0path\0
# - A\0path\0
# - D\0path\0
# - R100\0old\0new\0
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
git diff --cached --name-status -z > "$tmp"

# Read records from fd 3
while IFS= read -r -d '' status <&3; do
  case "$status" in
    R* )
      # rename: read old and new
      IFS= read -r -d '' oldpath <&3
      IFS= read -r -d '' newpath <&3

      # delete old path
      add_tree_delete "$oldpath"

      # add new path as blob
      read -r mode sha < <(create_blob_for_path "$newpath")
      add_tree_blob "$newpath" "$mode" "$sha"
      ;;
    D )
      IFS= read -r -d '' path <&3
      add_tree_delete "$path"
      ;;
    A|M|T|C|U )
      IFS= read -r -d '' path <&3
      read -r mode sha < <(create_blob_for_path "$path")
      add_tree_blob "$path" "$mode" "$sha"
      ;;
    * )
      # read path (best effort) for debug
      if IFS= read -r -d '' path <&3; then :; else path=""; fi
      echo "ERROR: unsupported staged status '$status' (path: ${path:-<none>})" >&2
      exit 1
      ;;
  esac
done 3<"$tmp"

if [[ "$(echo "$TREE_ITEMS" | jq 'length')" -eq 0 ]]; then
  echo "ERROR: No committable staged entries found." >&2
  exit 1
fi

echo "Staged entries to commit: $(echo "$TREE_ITEMS" | jq 'length')"

# Create new tree
TREE_JSON="$(api_json -X POST "$API_ROOT/repos/$OWNER/$REPO/git/trees" \
  -d "$(jq -n --arg base "$BASE_TREE_SHA" --argjson items "$TREE_ITEMS" '{base_tree:$base, tree:$items}')" )"
NEW_TREE_SHA="$(echo "$TREE_JSON" | jq -r '.sha')"
if [[ -z "$NEW_TREE_SHA" || "$NEW_TREE_SHA" == "null" ]]; then
  echo "ERROR: failed to create tree. Full response:" >&2
  echo "$TREE_JSON" | jq . >&2
  exit 1
fi

echo "NEW_TREE_SHA=$NEW_TREE_SHA"

# Create commit
COMMIT_CREATE_JSON="$(api_json -X POST "$API_ROOT/repos/$OWNER/$REPO/git/commits" \
  -d "$(jq -n --arg msg "$COMMIT_MSG" --arg tree "$NEW_TREE_SHA" --arg parent "$BASE_COMMIT_SHA" \
    '{message:$msg, tree:$tree, parents:[$parent]}')" )"
NEW_COMMIT_SHA="$(echo "$COMMIT_CREATE_JSON" | jq -r '.sha')"
if [[ -z "$NEW_COMMIT_SHA" || "$NEW_COMMIT_SHA" == "null" ]]; then
  echo "ERROR: failed to create commit. Full response:" >&2
  echo "$COMMIT_CREATE_JSON" | jq . >&2
  exit 1
fi

echo "NEW_COMMIT_SHA=$NEW_COMMIT_SHA"

# Update branch ref
PATCH_JSON="$(api_json -X PATCH "$API_ROOT/repos/$OWNER/$REPO/git/refs/heads/$BRANCH" \
  -d "$(jq -n --arg sha "$NEW_COMMIT_SHA" '{sha:$sha, force:false}')" )"

# Print full PATCH response (for debugging / transparency)
echo "PATCH response:"
echo "$PATCH_JSON" | jq .

REF="$(echo "$PATCH_JSON" | jq -r '.ref')"
UPDATED_SHA="$(echo "$PATCH_JSON" | jq -r '.object.sha')"
if [[ "$REF" == "null" || "$UPDATED_SHA" == "null" ]]; then
  echo "ERROR: ref update failed." >&2
  echo "$PATCH_JSON" | jq -r '.message? // empty' >&2
  exit 1
fi

echo "OK: pushed staged changes via API"
echo "Branch: $BRANCH"
echo "Ref: $REF"
echo "Commit: $UPDATED_SHA"

