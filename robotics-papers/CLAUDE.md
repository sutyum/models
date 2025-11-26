# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a paper reading and note-taking system for robotics research papers. The repository uses a two-stage reading workflow:

1. **VIBE mode** (quick screening): Initial 5-10 minute pass to determine if a paper is worth reading deeply
2. **READ mode** (deep analysis): Comprehensive 2+ hour analysis with structured questions and connections

## Key Commands

### Creating new paper notes

```bash
# Quick screen (VIBE mode)
./new.sh <PaperName>
./new.sh --vibe <PaperName>

# Deep read (READ mode) - auto-imports metadata from vibe if available
./new.sh --read <PaperName>

# List all existing notes
./new.sh
```

### Git operations

```bash
# Check status
git status

# Add and commit changes
git add .
git commit -m "message"

# Push to remote
git push
```

## Architecture

### Directory Structure

- `vibe/` - Quick screening notes for papers (VIBE template)
- `read/` - Deep analysis notes for papers (READ template)
- `VIBE.md` - Template for quick paper screening (checklist-style)
- `READ.md` - Template for comprehensive paper analysis (10-section structured format)
- `new.sh` - CLI tool for creating and managing paper notes
- `README.md` - Reading list organized by topic

### Workflow

1. When encountering a new paper, create a vibe note using `./new.sh <PaperName>`
2. Fill out the VIBE template to decide if the paper warrants deep reading
3. If worthy, create a READ note using `./new.sh --read <PaperName>` which:
   - Uses READ.md as the template
   - Auto-imports title, authors, and link from the corresponding vibe note if it exists
4. The `new.sh` script handles:
   - Template instantiation
   - Title replacement in `[Title]` placeholders
   - Metadata extraction from vibe notes when creating read notes
   - Safe filename generation (spaces → underscores)
   - Listing existing notes sorted by modification time

### Reading List Categories (README.md)

Papers are organized by topic:
- **Motion Imitation**: DeepMimic, AMP, GAIL
- **Generative Models**: DDPM, Flow Matching, Conditional Flow Matching
- **VLA (Vision-Language-Action)**: Groot N1/N1.5, FAST, Pi0 variants, SmolVLA
- **WBC (Whole Body Control)**: TWIST, TWIST 2, SONIC, BeyondMimic, OmniRetarget
- **Retargeting**: GMR, GVHMR
- **High-Level Control**: Hi Robot
- **Data Collection**: UMI

### Templates

**VIBE.md** (Quick Screen):
- 30-second skim section
- Relevance filter table
- Quality signals checklist
- Verdict: DEEP READ / SKIM / ARCHIVE / SKIP

**READ.md** (Deep Analysis):
10 structured sections covering:
1. The claim (problem, core bet, impact)
2. Architecture (system diagram, information flow)
3. Math (equations, loss decomposition, gradients)
4. Data (training data, sim2real gap)
5. Evaluation (metrics, baselines, ablations)
6. Honesty check (what they don't show)
7. Deep understanding tests (Feynman, prediction, extension, breaking)
8. Connections (to other papers, to your work)
9. Innovation extraction (generalizable insight, gaps, variants)
10. Spaced repetition hooks (quiz questions, one-liner)

## Key Implementation Details

### new.sh Script

- Portable across GNU and BSD systems (handles `sed -i`, `stat`, `date` differences)
- Creates parent directories automatically
- Prevents overwriting existing notes
- Uses modification time for sorting in listings
- Escapes special characters when doing sed replacements
- When creating READ notes from vibe:
  - Extracts title from `# Quick Screen: [Title]` line
  - Extracts authors from `**Authors / Venue / Year:**` line
  - Extracts link from `**Link:**` line

### File Naming

Paper names with spaces are converted to underscores for safe filenames. For example:
- `./new.sh Deep Mimic` creates `vibe/Deep_Mimic.md`

## Research Focus

Based on the notes, the researcher is working on:
- VLA (Vision-Language-Action) systems
- Humanoid control (motor control / low-level)
- Motion tracking and retargeting
- Sim-to-real transfer
- TEMPEQ (referenced in READ template)

Key interests visible from vibe notes:
- Reducing artifacts in human-to-humanoid motion retargeting
- Extensive reward engineering challenges
- Co-design of hardware and controllers for agility
- Centroidal dynamics modeling (CII, SLIP, centroidal momentum matrix)
