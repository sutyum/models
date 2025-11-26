# Quick Screen: Sampling-based System Identification with Active Exploration for legged robot Sim2Real Learning

**Authors / Venue / Year:** Nikhil Sobanbabu, Guanqi He, Tairan He, Yuxiang Yang, Guanya Shi / Arxiv / 2025
**Link:** https://arxiv.org/pdf/2505.14266

---

## 30-Second Skim (Abstract + Figures)

**What's the claim?** (One line)
Maximizing fisher information of the actively collected data in the real world improves sim2real transfer for legged robots.


**The figure that tells the story:** (Figure #)
Figure 2: Shows the 2 stage process of parameter identification followed by active exploration to improve sim2real transfer.


**Task domain:** [ ] Manipulation  [x] Locomotion  [ ] Navigation  [ ] Multi-task  [ ] Other: ___

---

## Relevance Filter

| Relevant to... | Yes/No | How? |
|----------------|--------|------|
| VLA / embodied AI |  |  |
| Motor control / low-level |  |  |
| Sim2Real | Yes | Active sampling to learn better parameters improves Sim2Real transfer |
| Data efficiency |  |  |
| My current problem: ___ |  |  |

---

## Quality Signals (Flip through)

- [x] Real robot results (not just sim)
- [x] Ablations present
- [x] Code released
- [x] Comparisons to recent (<2 yr) baselines
- [x] Clear failure cases shown

**Red flags spotted:**
Active exploration is only showed for quadruped robots, not bipedal. Use with humanoids seems to improve performance but limited to stage 1 of the 2-stage process.


---

## Gut Check

**Novelty:** [x] Incremental  [ ] Solid contribution  [ ] Potentially big idea

**Could I explain the core idea right now?** [x] Yes  [ ] Vaguely  [ ] No (complexity signal)

**One thing that made me curious:**
The use of fisher information reward to guide system identification is interesting, wonder how this reward can be used in other contexts.


---

## Verdict

[ ] **DEEP READ** - High relevance, worth 2+ hours
[x] **SKIM** - Grab the technique, skip details  
[ ] **ARCHIVE** - Reference later if needed
[ ] **SKIP** - Not for me right now

**If reading, focus on Section(s):**
