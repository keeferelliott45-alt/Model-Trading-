# Phase 2 backtest report

Generated 2026-11-26T22:01:21.278270+00:00

Assets used: ['AMD', 'AVGO', 'BTC-USD', 'CEG', 'CRWV', 'DLR', 'ETH-USD', 'FCX', 'GLD', 'MU', 'SLV']
Dropped: []

```
Tests run: 261
Significant raw (p<0.05): 16 (~13.1 expected by chance)
Survive FDR (BH): 0
Survive Bonferroni: 0
Raw-significant reversion: 6, continuation: 10
```

Top 15 by raw p-value:

```
  asset  lookback  horizon  events  mean_reversion_ret  control_ret    edge  t_stat  p_value  folds_used  folds_positive  p_fdr  p_bonf  sig_raw  sig_fdr  sig_bonf    direction
ETH-USD        60        5     155             -0.0234       0.0008 -0.0242 -2.8466   0.0050           4               0 0.6070  1.0000     True    False     False continuation
ETH-USD        30        5     187             -0.0182       0.0009 -0.0192 -2.7528   0.0064           4               0 0.6070  1.0000     True    False     False continuation
BTC-USD        60        5     145             -0.0170      -0.0012 -0.0158 -2.3550   0.0198           4               0 0.6070  1.0000     True    False     False continuation
    DLR        30        5     196              0.0060      -0.0005  0.0065  2.3154   0.0215           4               4 0.6070  1.0000     True    False     False    reversion
ETH-USD        60       60      39             -0.1537      -0.0032 -0.1505 -2.3876   0.0219           4               0 0.6070  1.0000     True    False     False continuation
     MU       180        1     199              0.0064       0.0009  0.0055  2.2861   0.0232           4               4 0.6070  1.0000     True    False     False    reversion
    FCX        60        5     184             -0.0065       0.0020 -0.0085 -2.1890   0.0297           4               1 0.6070  1.0000     True    False     False continuation
    DLR       180       20      79              0.0172      -0.0005  0.0178  2.1991   0.0307           4               3 0.6070  1.0000     True    False     False    reversion
ETH-USD        60        3     178             -0.0132      -0.0003 -0.0128 -2.1619   0.0318           4               0 0.6070  1.0000     True    False     False continuation
    DLR        60        5     159              0.0061      -0.0007  0.0068  2.1356   0.0341           4               4 0.6070  1.0000     True    False     False    reversion
ETH-USD        30       10     142             -0.0233       0.0011 -0.0244 -2.1338   0.0344           4               0 0.6070  1.0000     True    False     False continuation
     MU        30        1     259              0.0049       0.0005  0.0044  2.1051   0.0361           4               4 0.6070  1.0000     True    False     False    reversion
    DLR        30        3     224              0.0041      -0.0005  0.0046  2.0706   0.0394           4               4 0.6070  1.0000     True    False     False    reversion
ETH-USD        90        5     142             -0.0181      -0.0001 -0.0180 -2.0614   0.0410           4               1 0.6070  1.0000     True    False     False continuation
    GLD        30        5     209             -0.0025       0.0006 -0.0030 -2.0494   0.0415           4               1 0.6070  1.0000     True    False     False continuation
```
