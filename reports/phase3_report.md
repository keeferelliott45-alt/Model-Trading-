# Phase 3 (funding) report

```
Tests: 144
Raw p<0.05: 6 (~7.2 by chance)
Survive FDR: 0
Survive Bonferroni: 0
Raw-sig reversion: 0, continuation: 6
```

Top 12:

```
  asset          model          subset  lookback  horizon  events  mean_rev  control    edge  p_value  p_fdr  p_bonf    direction
ETH-USD regime+funding         flagged        30        5     169   -0.0180   0.0034 -0.0214   0.0036 0.5205  0.5205 continuation
ETH-USD regime+funding         flagged        30       10     127   -0.0296   0.0041 -0.0338   0.0079 0.5721  1.0000 continuation
BTC-USD regime+funding         flagged        90        5     117   -0.0189  -0.0003 -0.0186   0.0119 0.5728  1.0000 continuation
ETH-USD regime+funding         flagged        30        3     199   -0.0098   0.0014 -0.0112   0.0275 0.8548  1.0000 continuation
ETH-USD regime+funding         flagged        90        5     129   -0.0186   0.0005 -0.0192   0.0324 0.8548  1.0000 continuation
ETH-USD regime+funding         flagged        90       10     104   -0.0302   0.0011 -0.0313   0.0356 0.8548  1.0000 continuation
ETH-USD     price_only flagged_crowded       180        1      24    0.0233   0.0007  0.0226   0.0604 0.8945  1.0000    reversion
BTC-USD regime+funding         flagged        60        5     125   -0.0133   0.0003 -0.0136   0.0616 0.8945  1.0000 continuation
ETH-USD regime+funding         flagged        60       10     106   -0.0257   0.0017 -0.0274   0.0624 0.8945  1.0000 continuation
ETH-USD regime+funding         flagged        60        5     134   -0.0150   0.0019 -0.0169   0.0678 0.8945  1.0000 continuation
ETH-USD regime+funding flagged_crowded       180        1      26    0.0188   0.0006  0.0182   0.0683 0.8945  1.0000    reversion
ETH-USD regime+funding flagged_crowded        30        5      37   -0.0221   0.0034 -0.0255   0.0917 0.9150  1.0000 continuation
```
