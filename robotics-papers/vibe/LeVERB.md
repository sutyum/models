# Quick Screen: LeVERB - Humanoid whole-body control with latent visual-language instruction

**Authors / Venue / Year:** Haoru Xue, ..., Shankar Satry / Arxiv / Sep 2025
**Link:** https://arxiv.org/pdf/2506.13751

---

## 30-Second Skim (Abstract + Figures)

**What's the claim?** (One line)
Introduces a benchmark for instruction following in the whole body control context. Also uses latent space to encode actions before they are turned into low level motor commands.

**The figure that tells the story:** (Figure #)
Figure 1 shows the System1/System2 type of cascade VLA architecture.

**Task domain:** [x] Manipulation  [x] Locomotion  [ ] Navigation  [x] Multi-task  [ ] Other: ___

---

## Relevance Filter

| Relevant to... | Yes/No | How? |
|----------------|--------|------|
| VLA / embodied AI | Yes  | VLA applied to whole body control |
| Motor control / low-level |  |  |
| Sim2Real |  |  |
| Data efficiency |  |  |
| My current problem: ___ |  |  |

---

## Quality Signals (Flip through)

- [x] Real robot results (not just sim)
- [x] Ablations present
- [ ] Code released
- [ ] Comparisons to recent (<2 yr) baselines
- [x] Clear failure cases shown

**Red flags spotted:**
If there is no code and so it may not be feasible to replicate their results. The latent space, 2 system idea can work but the literature seems to be bent towards single model approaches.

---

## Gut Check

**Novelty:** [ ] Incremental  [x] Solid contribution  [ ] Potentially big idea

**Could I explain the core idea right now?** [x] Yes  [ ] Vaguely  [ ] No (complexity signal)

**One thing that made me curious:**
Their demo videos and the possibility of latent space being "enough" for modeling VLAs.

---

## Verdict

[ ] **DEEP READ** - High relevance, worth 2+ hours
[x] **SKIM** - Grab the technique, skip details  
[ ] **ARCHIVE** - Reference later if needed
[ ] **SKIP** - Not for me right now

**If reading, focus on Section(s):**
