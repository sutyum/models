# Quick Screen: OpenVLA - An open-source vision-language-action model

**Authors / Venue / Year:** Moo Jin Kim, ..., <u>Sergey Levine</u>, Percy Liang, <u>Chelsea Finn</u>
**Link:** https://arxiv.org/pdf/2406.09246

---

## 30-Second Skim (Abstract + Figures)

**What's the claim?** (One line)
Open weight and code for a VLA beating 2024 closed source SOTA - Google's RT-2-X.

**The figure that tells the story:** (Figure #)
Figure 2

**Task domain:** [x] Manipulation  [ ] Locomotion  [ ] Navigation  [ ] Multi-task  [ ] Other: ___

---

## Relevance Filter

| Relevant to... | Yes/No | How? |
|----------------|--------|------|
| VLA / embodied AI | Yes | An open source VLA |
| Motor control / low-level |  |  |
| Sim2Real |  |  |
| Data efficiency |  |  |
| My current problem: ___ |  |  |

---

## Quality Signals (Flip through)

- [x] Real robot results (not just sim)
- [ ] Ablations present
- [x] Code released
- [x] Comparisons to recent (<2 yr) baselines
- [x] Clear failure cases shown

**Red flags spotted:**
No ablation, so would have to be careful to draw conclusions about the 2 encoder concept (DINOv2 + SigLIP).

---

## Gut Check

**Novelty:** [ ] Incremental  [x] Solid contribution  [ ] Potentially big idea

**Could I explain the core idea right now?** [x] Yes  [ ] Vaguely  [ ] No (complexity signal)

**One thing that made me curious:**
The two image encoders and also action tokenization.

---

## Verdict

[x] **DEEP READ** - High relevance, worth 2+ hours
[ ] **SKIM** - Grab the technique, skip details  
[ ] **ARCHIVE** - Reference later if needed
[ ] **SKIP** - Not for me right now

**If reading, focus on Section(s):**
