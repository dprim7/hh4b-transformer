# Implementation plan for jet-less transformer for HH4b analysis

## Tasks

- [ ] Figure out which data format to copy to pvc
- [ ] Create PVC on Nautilus (or use existing)
- [ ] Create preprocessing to prepare files for PVC
- [ ] Get/generate Run 2 data
- [ ] Create configurable model with callbacks (include WandB)
- [ ] Create mock training data & unit tests
- [ ] Create trainer
- [ ] Create/integrate evaluation plots


# Notes
Reference: 
```
"Potential of di-Higgs observation via a calibratable jet-free HH -> 4b framework" https://arxiv.org/pdf/2508.15048 
```

## Hyperparameters

Architecture:
 * Attention Layers: 8 (meaning?)
 * Embedding Dimension: 256
 * Attention Heads: 16 (double check, thought it was 8 in other paper)
 * Final MLP: one hidden layer with 1024 units
 * Softmax over classes

 Regularization: 
 * Dropout: 0.1

 ## Task

 138-class multiclass. 136 signal classes (?) with discretized (m_h1, m_h2), 2 BG classes (QCD, ttbar)
 
 ==> Produce: 
 * mass de-correlated HH discriminant
 * event-level (m_h1, m_h2) estimator (localized fit over per-class posteriors (?))



