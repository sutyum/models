#!/usr/bin/env bash
set -euo pipefail

# CLI to manage paper reading notes in "vibe" and "read" modes.
#
# Usage:
#   new.sh                     # show usage + list existing vibe/read notes
#   new.sh --vibe <PaperName>  # create vibe/<PaperName>.md from VIBE.md
#   new.sh --read <PaperName>  # create read/<PaperName>.md from READ.md (templated from vibe if available)
#   new.sh <PaperName>         # shorthand for: new.sh --vibe <PaperName>
#
# Behavior:
#   - Name is mandatory for creating a file.
#   - READ.md and VIBE.md are markdown templates stored next to this script.
#   - For --read:
#       * Uses READ.md as template.
#       * If vibe/<PaperName>.md exists, pulls basic fields from it:
#           - [Title] (from "# Quick Screen: ...")
#           - **Authors:** (from "**Authors / Venue / Year:**")
#           - **Link:** (from "**Link:**")
#   - For both modes, if [Title] appears in the template, it is replaced
#     with a chosen title (PaperName or title from vibe).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Portable helpers: sed -i, stat, date ------------------------------------

# sed -i (GNU vs BSD)
if sed --version >/dev/null 2>&1; then
  SED_INPLACE=(sed -i)
else
  SED_INPLACE=(sed -i '')
fi

# stat mtime (GNU vs BSD)
if stat -c '%Y' "$SCRIPT_DIR" >/dev/null 2>&1; then
  STAT_MTIME() { stat -c '%Y' "$1"; }
else
  STAT_MTIME() { stat -f '%m' "$1"; }
fi

# date from epoch (GNU vs BSD)
if date -d @0 +%Y >/dev/null 2>&1; then
  FORMAT_EPOCH() { date -d "@$1" '+%Y-%m-%d %H:%M'; }
else
  FORMAT_EPOCH() { date -r "$1" '+%Y-%m-%d %H:%M'; }
fi

escape_sed() {
  # Escape & and / for sed replacement
  printf '%s' "$1" | sed -e 's/[&/]/\\&/g'
}

extract_year() {
  # Extract year from markdown file
  # Looks for patterns like: **Authors / Venue / Year:** ... 2024
  # or **Authors:** ... 2024
  local file="$1"
  local year=""

  # Try to find year in Authors/Venue/Year line or Authors line
  year=$(grep -m1 '^\*\*Authors' "$file" 2>/dev/null | grep -oE '20[0-9]{2}' | head -1 || true)

  # If not found, look anywhere in the first 30 lines for a 4-digit year starting with 20
  if [[ -z "$year" ]]; then
    year=$(head -30 "$file" 2>/dev/null | grep -oE '20[0-9]{2}' | head -1 || true)
  fi

  printf '%s' "${year:-—}"
}

days_ago() {
  # Calculate days between epoch timestamp and now
  local mtime="$1"
  local now
  now=$(date +%s)
  local diff=$((now - mtime))
  local days=$((diff / 86400))

  if [[ $days -eq 0 ]]; then
    printf "today"
  elif [[ $days -eq 1 ]]; then
    printf "1 day"
  else
    printf "%dd" "$days"
  fi
}

print_usage() {
  cat <<EOF
Usage:
  $(basename "$0")                     Show this help and list existing vibe/read notes
  $(basename "$0") --vibe <PaperName>  Create vibe notes for a paper
  $(basename "$0") --read <PaperName>  Create deep-read notes for a paper (templated from vibe if available)
  $(basename "$0") <PaperName>         Shorthand for: $(basename "$0") --vibe <PaperName>

Templates:
  VIBE.md  Quick-screen template for first-pass reads
  READ.md  Deep analysis template for serious reads
EOF
}

print_listing() {
  # Build a list of unique paper names and their metadata
  # Format: mtime|name|vibe_exists|vibe_year|read_exists|read_year
  local entries=()
  local seen_papers=()

  shopt -s nullglob

  # First pass: collect all unique paper names
  local all_names=()

  local vibe_dir="$SCRIPT_DIR/vibe"
  if [[ -d "$vibe_dir" ]]; then
    for f in "$vibe_dir"/*.md; do
      local base="${f##*/}"
      local name="${base%.md}"
      all_names+=("$name")
    done
  fi

  local read_dir="$SCRIPT_DIR/read"
  if [[ -d "$read_dir" ]]; then
    for f in "$read_dir"/*.md; do
      local base="${f##*/}"
      local name="${base%.md}"
      # Check if already in list
      local found=false
      for n in "${all_names[@]}"; do
        if [[ "$n" == "$name" ]]; then
          found=true
          break
        fi
      done
      $found || all_names+=("$name")
    done
  fi

  shopt -u nullglob

  echo
  echo "Existing papers (most recent first):"
  echo

  if [[ ${#all_names[@]} -eq 0 ]]; then
    echo "  (none yet – create one with: $(basename "$0") --vibe DeepMimic)"
    echo
    return
  fi

  # Second pass: for each paper, gather metadata
  for name in "${all_names[@]}"; do
    local vibe_file="$vibe_dir/${name}.md"
    local read_file="$read_dir/${name}.md"

    local vibe_exists=0 vibe_year="—" vibe_mtime=0
    local read_exists=0 read_year="—" read_mtime=0

    if [[ -f "$vibe_file" ]]; then
      vibe_exists=1
      vibe_mtime="$(STAT_MTIME "$vibe_file")"
      vibe_year="$(extract_year "$vibe_file")"
    fi

    if [[ -f "$read_file" ]]; then
      read_exists=1
      read_mtime="$(STAT_MTIME "$read_file")"
      read_year="$(extract_year "$read_file")"
    fi

    # Use most recent mtime for sorting
    local latest_mtime=$vibe_mtime
    [[ $read_mtime -gt $latest_mtime ]] && latest_mtime=$read_mtime

    entries+=("$latest_mtime|$name|$vibe_exists|$vibe_year|$vibe_mtime|$read_exists|$read_year|$read_mtime")
  done

  # ANSI color codes
  local COLOR_VIBE='\033[36m'      # cyan - vibe only
  local COLOR_READ='\033[32m'      # green - read only
  local COLOR_BOTH='\033[1;33m'    # bold yellow - both
  local COLOR_RESET='\033[0m'

  printf "%-11s | %-28s | %-6s | %-8s | %s\n" "Status" "Paper" "Year" "Age" "Files"
  printf "%s\n" "------------+------------------------------+--------+----------+-------------------------"

  # Sort and display
  printf '%s\n' "${entries[@]}" | sort -t'|' -k1,1nr | while IFS='|' read -r latest_mtime name vibe_exists vibe_year vibe_mtime read_exists read_year read_mtime; do
    local status year age files color

    # Determine status based on what exists
    if [[ "$vibe_exists" == "1" && "$read_exists" == "1" ]]; then
      status="vibe + read"
      color="$COLOR_BOTH"
      # Prefer read year, fall back to vibe year
      year="$read_year"
      [[ "$year" == "—" ]] && year="$vibe_year"
      files="vibe/$name.md, read/$name.md"
      # Age from most recent
      local recent_mtime=$vibe_mtime
      [[ $read_mtime -gt $recent_mtime ]] && recent_mtime=$read_mtime
      age="$(days_ago "$recent_mtime")"
    elif [[ "$vibe_exists" == "1" ]]; then
      status="vibe"
      color="$COLOR_VIBE"
      year="$vibe_year"
      files="vibe/$name.md"
      age="$(days_ago "$vibe_mtime")"
    else
      status="read"
      color="$COLOR_READ"
      year="$read_year"
      files="read/$name.md"
      age="$(days_ago "$read_mtime")"
    fi

    printf "${color}%-11s${COLOR_RESET} | %-28s | %-6s | %-8s | %s\n" "$status" "$name" "$year" "$age" "$files"
  done

  echo
}

# --- Argument parsing --------------------------------------------------------

MODE=""        # "vibe" or "read"
PAPER_NAME=""

if [[ $# -eq 0 ]]; then
  print_usage
  print_listing
  exit 0
fi

case "${1-}" in
  --read)
    MODE="read"
    shift
    ;;
  --vibe)
    MODE="vibe"
    shift
    ;;
  -h|--help)
    print_usage
    print_listing
    exit 0
    ;;
esac

if [[ $# -eq 0 ]]; then
  echo "Error: paper name is required." >&2
  echo
  print_usage
  exit 1
fi

PAPER_NAME="$*"
SAFE_NAME="${PAPER_NAME// /_}"

# Default mode if not explicitly set: vibe
if [[ -z "$MODE" ]]; then
  MODE="vibe"
fi

READ_TEMPLATE="$SCRIPT_DIR/READ.md"
VIBE_TEMPLATE="$SCRIPT_DIR/VIBE.md"

TARGET_DIR="$SCRIPT_DIR/$MODE"
mkdir -p "$TARGET_DIR"
TARGET_FILE="$TARGET_DIR/${SAFE_NAME}.md"

VIBE_NOTES_DIR="$SCRIPT_DIR/vibe"
VIBE_NOTES_FILE="$VIBE_NOTES_DIR/${SAFE_NAME}.md"

# --- Pick template and basic checks -----------------------------------------

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

# --- Resolve title to insert into [Title] ------------------------------------

TITLE="$PAPER_NAME"

# If we're in read mode and a vibe file exists, prefer the title from vibe notes
if [[ "$MODE" == "read" && -f "$VIBE_NOTES_FILE" ]]; then
  V_TITLE_LINE="$(grep -m1 '^# Quick Screen:' "$VIBE_NOTES_FILE" || true)"
  V_TITLE_VAL="$(printf '%s' "$V_TITLE_LINE" | sed 's/^# Quick Screen:[[:space:]]*//')"
  if [[ -n "$V_TITLE_VAL" && "$V_TITLE_VAL" != "[Title]" ]]; then
    TITLE="$V_TITLE_VAL"
  fi
fi

# --- Create base file from template ------------------------------------------

cp "$TEMPLATE_FILE" "$TARGET_FILE"

# Replace [Title] (if present) with resolved title
if grep -q '\[Title\]' "$TARGET_FILE"; then
  esc_title="$(escape_sed "$TITLE")"
  "${SED_INPLACE[@]}" "s/\\[Title\\]/${esc_title}/g" "$TARGET_FILE"
fi

# --- If read mode, template authors/link from vibe if available --------------

if [[ "$MODE" == "read" && -f "$VIBE_NOTES_FILE" ]]; then
  # Authors / Venue / Year
  AVY_LINE="$(grep -m1 '^\*\*Authors / Venue / Year:\*\*' "$VIBE_NOTES_FILE" || true)"
  AVY_VAL="$(printf '%s' "$AVY_LINE" | sed 's/^\*\*Authors \/ Venue \/ Year:\*\*[[:space:]]*//')"

  if [[ -n "$AVY_VAL" ]]; then
    esc_avy="$(escape_sed "$AVY_VAL")"
    if grep -q '^\*\*Authors:\*\*' "$TARGET_FILE"; then
      "${SED_INPLACE[@]}" "s/^\\*\\*Authors:\\*\\*[[:space:]]*/**Authors:** ${esc_avy}\n/" "$TARGET_FILE"
    fi
  fi

  # Link
  LINK_LINE="$(grep -m1 '^\*\*Link:\*\*' "$VIBE_NOTES_FILE" || true)"
  LINK_VAL="$(printf '%s' "$LINK_LINE" | sed 's/^\*\*Link:\*\*[[:space:]]*//')"

  if [[ -n "$LINK_VAL" ]]; then
    esc_link="$(escape_sed "$LINK_VAL")"
    if grep -q '^\*\*Link:\*\*' "$TARGET_FILE"; then
      "${SED_INPLACE[@]}" "s/^\\*\\*Link:\\*\\*[[:space:]]*/**Link:** ${esc_link}\n/" "$TARGET_FILE"
    fi
  fi

  IMPORT_NOTE=" (templated from vibe)"
else
  IMPORT_NOTE=""
fi

# --- Final output ------------------------------------------------------------

echo "Created new $MODE note${IMPORT_NOTE}:"
echo "  Paper: $PAPER_NAME"
echo "  File:  $TARGET_FILE"
echo "  From:  $TEMPLATE_FILE"
if [[ "$MODE" == "read" && -f "$VIBE_NOTES_FILE" ]]; then
  echo "  Vibe:  $VIBE_NOTES_FILE"
fi
