#!/usr/bin/env bash
set -euo pipefail

# CLI to allow users to start reading a new paper in vibe mode or in read mode.
#
# Usage:
#   new.sh [--vibe | --read] [PaperName]
#
# If no mode is specified, defaults to vibe mode.
# If no PaperName is given, defaults to YYYY-MM.
#
# Behavior:
#   - Vibe mode:
#       VIBE.md -> vibe/<PaperName>.md
#   - Read mode:
#       READ.md -> read/<PaperName>.md
#       And if vibe/<PaperName>.md exists, use it to fill some fields in READ.md:
#         * [Title]
#         * **Authors:**
#         * **Link:**

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="vibe"    # default mode
PAPER_NAME=""

# Detect sed -i flavor (GNU vs BSD/macOS)
if sed --version >/dev/null 2>&1; then
  # GNU sed
  SED_INPLACE=(sed -i)
else
  # BSD/macOS sed
  SED_INPLACE=(sed -i '')
fi

escape_sed() {
  # Escape & and / for sed replacement
  printf '%s' "$1" | sed -e 's/[&/]/\\&/g'
}

# --- Parse arguments ---

if [[ $# -gt 0 ]]; then
  case "$1" in
    --read)
      MODE="read"
      shift
      ;;
    --vibe)
      MODE="vibe"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--vibe | --read] [PaperName]"
      exit 0
      ;;
    *)
      # No explicit mode flag; fall through, treat $1 as part of paper name
      ;;
  esac
fi

# Remaining args (if any) form the paper name
if [[ $# -gt 0 ]]; then
  PAPER_NAME="$*"
else
  PAPER_NAME="$(date +"%Y-%m")"
fi

# Filename-safe version
SAFE_NAME="${PAPER_NAME// /_}"

READ_TEMPLATE="$SCRIPT_DIR/READ.md"
VIBE_TEMPLATE="$SCRIPT_DIR/VIBE.md"

TARGET_DIR="$SCRIPT_DIR/$MODE"
mkdir -p "$TARGET_DIR"
TARGET_FILE="$TARGET_DIR/${SAFE_NAME}.md"

VIBE_NOTES_DIR="$SCRIPT_DIR/vibe"
VIBE_NOTES_FILE="$VIBE_NOTES_DIR/${SAFE_NAME}.md"

# --- Choose template and sanity-check ---

if [[ "$MODE" == "read" ]]; then
  TEMPLATE_FILE="$READ_TEMPLATE"
  if [[ ! -f "$READ_TEMPLATE" ]]; then
    echo "Error: READ template not found at $READ_TEMPLATE" >&2
    exit 1
  fi
else
  TEMPLATE_FILE="$VIBE_TEMPLATE"
  if [[ ! -f "$VIBE_TEMPLATE" ]]; then
    echo "Error: VIBE template not found at $VIBE_TEMPLATE" >&2
    exit 1
  fi
fi

if [[ -e "$TARGET_FILE" ]]; then
  echo "Error: target already exists: $TARGET_FILE" >&2
  exit 1
fi

# --- Create base file from template ---

cp "$TEMPLATE_FILE" "$TARGET_FILE"

# --- In all cases, at least set [Title] from PAPER_NAME if present ---

TITLE_FROM_NAME="$PAPER_NAME"

# Fill [Title] with the paper name first (will be overwritten by vibe data if available)
if grep -q '\[Title\]' "$TARGET_FILE"; then
  esc_title="$(escape_sed "$TITLE_FROM_NAME")"
  # Replace the first occurrence of [Title]
  "${SED_INPLACE[@]}" "0,/\\[Title\\]/s//${esc_title}/" "$TARGET_FILE"
fi

# --- If read mode, use vibe notes as a data source to fill fields ---

if [[ "$MODE" == "read" && -f "$VIBE_NOTES_FILE" ]]; then
  # Extract title from vibe file: '# Quick Screen: Something'
  TITLE_LINE="$(grep -m1 '^# Quick Screen:' "$VIBE_NOTES_FILE" || true)"
  TITLE_VAL="$(printf '%s' "$TITLE_LINE" | sed 's/^# Quick Screen:[[:space:]]*//')"
  # If user filled title and it's not the placeholder, override the title in READ
  if [[ -n "$TITLE_VAL" && "$TITLE_VAL" != "[Title]" ]]; then
    esc_title="$(escape_sed "$TITLE_VAL")"
    if grep -q '\[Title\]' "$TARGET_FILE"; then
      "${SED_INPLACE[@]}" "0,/\\[Title\\]/s//${esc_title}/" "$TARGET_FILE"
    else
      # Fallback: replace the heading line with the new title
      "${SED_INPLACE[@]}" "1s/^# Paper Analysis: .*/# Paper Analysis: ${esc_title}/" "$TARGET_FILE"
    fi
  fi

  # Extract authors/venue/year line
  AVY_LINE="$(grep -m1 '^\*\*Authors / Venue / Year:\*\*' "$VIBE_NOTES_FILE" || true)"
  AVY_VAL="$(printf '%s' "$AVY_LINE" | sed 's/^\*\*Authors \/ Venue \/ Year:\*\*[[:space:]]*//')"

  if [[ -n "$AVY_VAL" ]]; then
    esc_avy="$(escape_sed "$AVY_VAL")"
    # Put entire string into Authors; user can cleanly split if they want
    if grep -q '^\*\*Authors:\*\*' "$TARGET_FILE"; then
      "${SED_INPLACE[@]}" "s/^\\*\\*Authors:\\*\\*[[:space:]]*/**Authors:** ${esc_avy}/" "$TARGET_FILE"
    fi
    # Optionally also copy to Venue/Year if blank
    if grep -q '^\*\*Venue\/Year:\*\*' "$TARGET_FILE"; then
      "${SED_INPLACE[@]}" "s/^\\*\\*Venue\/Year:\\*\\*[[:space:]]*/**Venue\/Year:** ${esc_avy}/" "$TARGET_FILE"
    fi
  fi

  # Extract link
  LINK_LINE="$(grep -m1 '^\*\*Link:\*\*' "$VIBE_NOTES_FILE" || true)"
  LINK_VAL="$(printf '%s' "$LINK_LINE" | sed 's/^\*\*Link:\*\*[[:space:]]*//')"

  if [[ -n "$LINK_VAL" ]]; then
    esc_link="$(escape_sed "$LINK_VAL")"
    if grep -q '^\*\*Link:\*\*' "$TARGET_FILE"; then
      "${SED_INPLACE[@]}" "s/^\\*\\*Link:\\*\\*[[:space:]]*/**Link:** ${esc_link}/" "$TARGET_FILE"
    fi
  fi

  IMPORT_NOTE=" (templated from vibe)"
else
  IMPORT_NOTE=""
fi

# --- Done ---

echo "Created new $MODE paper${IMPORT_NOTE}:"
echo "  Name:   $PAPER_NAME"
echo "  Path:   $TARGET_FILE"
echo "  Source: $TEMPLATE_FILE"
if [[ "$MODE" == "read" && -f "$VIBE_NOTES_FILE" ]]; then
  echo "  Vibe source: $VIBE_NOTES_FILE"
fi
echo "You can start editing it now!"
