# Jet-free Transformer for HH->4b Analysis with LHC Run 2 and Run 3 Data

## TODO

- [x] Figure out which data format to copy to pvc: compressed parquet (already sharded)
- [ ] Create PVC on Nautilus (or use existing)
- [ ] Current pvc has only 0.5 TB, estimate how much we'd need
- [ ] Create preprocessing to prepare files for PVC (no selections, just cleaning)
- [ ] Get/generate Run 2 data + any missing samples
- [ ] Move run 2 and run 3 data to pvc
- [ ] Create configurable model with callbacks (include WandB for tracking)
- [ ] Create mock training data & unit/integration tests
- [ ] Create trainer
- [ ] Create inference module
- [ ] Create/integrate evaluation plots
- [ ] Implement training on 4 GPUs



# Notes

### References

- Yang, T. and Li, C. (2025). *Potential of di-Higgs observation via a calibratable jet-free HH → 4b framework*. [arXiv:2508.15048](https://arxiv.org/pdf/2508.15048)
- Qu, H., Li, C., and Qian, S. (2024) *Particle Transformer for Jet Tagging*. [arXiv:2202.03772v3](https://arxiv.org/pdf/2202.03772)

### Original ParT Architecture (with Jets)
This implementation removes jets and has 8 overall attention blocks instead of 6 + 2 particle/class blocks

<img src="figures/legacy-arch.png" alt="Legacy architecture" width="600" title="Legacy architecture diagram">

## Hyperparameters

**Architecture:**
 * Attention Layers: 8 
 * Embedding Dimension: 256
 * Attention Heads: 16 (double check, thought it was 8 in other paper)
 * Final MLP: one hidden layer with 1024 units
 * Softmax over classes

 **Regularization:** 
 * Dropout: 0.1

 ## Task

Multiclass classifier. 136 signal classes (?) with discretized (m_h1, m_h2), BG classes: (136 QCD, 137 ttbar)(where are the others included?). Double check this 
 
 ### Produce: 
 * mass de-correlated HH discriminant
 * event-level (m_h1, m_h2) estimator (localized fit over per-class posteriors (?))

 ## Samples:

 * ggF HH, VBF HH
 * QCD multijet
 * Z -> qq
 * ttbar, single top, tW, ttW, ttZ, ttH
 * WW, WZ, ZZ
 * ggF H, VBF H, WplusH, WmiunusH
 * ZH

 ## Inputs 
single particle-candidate stream with masking

 **Particle Features:**

 up to 256 particles
  * Kinematic Variables:  ln(pT), ln(E), ln(pT^rel), ln(E^rel), ∆R, ∆η, ∆φ, η, φ
  * Track parameters: d0, d0^err, dz, dz^err  (hyperbolic tangent transformation for impact params (?))
  * Particle ID: Binary flags for charged hadrons, neutrol hadrons, photons, electrons, muons, continuous charge information (?)

Variable standardization employs manually optimized parameters: logarithmic pT and energy variables are centered at 1.7 and 2.0 respectively with scale factors of 0.7, while relative momentum variables are centered at −4.7 with identical scaling. Angular variables use specialized normalization with ∆R centered at 0.2 and scaled by 4.0.

**Lorentz vectors:**
 * (px, py, pz, E) for each particle

**Paricle masks:**
 * binary masks to distinguish genuine particles from zero padding

 ## Attention Biases
 **Pairwise features** embedded as biases: {ln(k_T), ln(z), ln(∆R), ln(m_ij)}
 (clustering metric, momentum fraction, angular separation, dijet mass)

 ## Training Setup

  * Ranger Optimizer (RAdam + Lookahead)
  * Initial LR: 2 * 10^-3
  * Automatic mixed precision
  * Epochs: 80
  * Batch size: 512
  * GPUs: 4
  * Per-epoch sample counts specified for train/val
  * Labels: 136 signal indices, events outside 40-200 GeV mapped to BG
  * No class reweighting
  * All generated samples used
  * Total Events: 140 M












