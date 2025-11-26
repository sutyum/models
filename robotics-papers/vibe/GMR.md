# Quick Screen: Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking

**Authors / Venue / Year:** Joao Araujo, <u>Yanjie Ze</u>, Pei Xu, Jianjun Wu, C. Karen Liu / Arxiv / 2025
**Link:** https://arxiv.org/pdf/2510.02252

---

## 30-Second Skim (Abstract + Figures)

**What's the claim?** (One line)
Retargeting human motion to humanoid embodiement introduces artifacts which are adressed by extensive reward engineering and domain randomization. They introduce a pipeline that reduces artifacts and improves policy success rate.


**The figure that tells the story:** (Figure #)
Figure 2: General Motion Retargeting Pipeline


**Task domain:** [ ] Manipulation  [x] Locomotion  [ ] Navigation  [ ] Multi-task  [x] Other: Motion Tracking

---

## Relevance Filter

| Relevant to... | Yes/No | How? |
|----------------|--------|------|
| VLA / embodied AI |  |  |
| Motor control / low-level |  |  |
| Sim2Real | Yes |  |
| Data efficiency |  |  |
| My current problem: Extensive reward engineering | Yes | Data processing pipeline which reduces artifacts |

---

## Quality Signals (Flip through)

- [ ] Real robot results (not just sim)
- [ ] Ablations present
- [x] Code released
- [x] Comparisons to recent (<2 yr) baselines
- [x] Clear failure cases shown

**Red flags spotted:**


---

## Gut Check

**Novelty:** [x] Incremental  [ ] Solid contribution  [ ] Potentially big idea

**Could I explain the core idea right now?** [ ] Yes  [x] Vaguely  [ ] No (complexity signal)

**One thing that made me curious:**
The evaluation method based on showing participants videos of different methods and asking them to rate the naturalness of the motion.

---

## Verdict

[x] **DEEP READ** - High relevance, worth 2+ hours
[ ] **SKIM** - Grab the technique, skip details  
[ ] **ARCHIVE** - Reference later if needed
[ ] **SKIP** - Not for me right now

**If reading, focus on Section(s):**
Section 2-4
