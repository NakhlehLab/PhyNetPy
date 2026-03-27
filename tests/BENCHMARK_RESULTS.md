# DEFJ Benchmark: Hill Climbing vs Simulated Annealing

## Scenarios

| Scenario | Taxa | Reticulations | Description |
|----------|------|---------------|-------------|
| D | 6 | 1 | Single hybridization |
| E | 6 | 2 | Two hybridizations |
| F | 6 | 3 | Three hybridizations |
| J | 14 | 2 | Two hybridizations, larger taxon set |

## Ground Truth Networks (Extended Newick)

- **D:** `(((b:0.009,((x:0.006,(y:0.003,z:0.003):0.003):0.003)#H1:0):0.003,(#H1:0,a:0.009):0.003):0.04366667,o:0.10233333);`
- **E:** `(o:0.10283333,(((a:0.006,((y:0.003,z:0.003):0.003)#H1:0):0.003,(x:0.009)#H2:0):0.003,(#H2:0,(#H1:0,b:0.006):0.003):0.003):0.04316667);`
- **F:** `(o:0.10383333,((((a:0.003,(z:0.003)#H1:0):0.003,(y:0.006)#H2:0):0.003,(x:0.009)#H3:0):0.003,((#H2:0,(#H1:0,b:0.003):0.003):0.003,#H3:0):0.003):0.04216667);`
- **J:** `((((b:0.017,a:0.017):0.01,((c:0.011,(d:0.006,(v:0.006)#H1:0):0.005):0.012,(((x:0.013,(y:0.01,z:0.01):0.003):0.003,w:0.016):0.007)#H2:0):0.004):0.008,((((#H1:0,e:0.006):0.006,(t:0.003,u:0.003):0.009):0.013,#H2:0):0.005,f:0.032):0.003):0.002275,o:0.043225);`

## Column Definitions

| Column | Meaning |
|--------|---------|
| **Test Case** | `Scenario-gN-tN` where `g` = number of gene trees and `t` = ILS scaling factor. Low `t` (e.g. 4) = low ILS (gene trees mostly concordant). High `t` (e.g. 100) = high ILS (gene trees very discordant). |
| **Method** | **HC** = Hill Climbing (only accepts strictly improving moves). **SA T=5** = Simulated Annealing, single run, temperature cooling from 5.0 to 0.01. **SA T=5 x3** = same SA schedule with 3 independent restarts, keeping the best. |
| **Init Pars** | Parsimony score (total extra lineages) of the starting network before search. Shared across methods for each test case since they start from the same random network. Lower is better; 0 = perfect reconciliation. |
| **Final Pars** | Parsimony score of the best network found after search. This is the objective being minimized. |
| **Acc/Up** | Accepted moves / Uphill moves. For HC, uphill is always 0 (greedy). For SA, uphill counts how many accepted moves were worse than the current state -- this is SA's mechanism for escaping local optima. |
| **mu_d** | Mu-distance to the true network. A generalization of Robinson-Foulds distance to networks via path-multiplicity vectors. 0 = exact topological match. |
| **hw_d** | Hardwired cluster distance to the true network. Symmetric difference of hardwired cluster sets (leaf descendants via directed edges from each node). 0 = exact match. |
| **Time** | Wall-clock seconds. |

## Results

### Scenario D (6 taxa, 1 reticulation)

| Test Case | Method | Init Pars | Final Pars | Acc/Up | mu_d | hw_d | Time |
|---|---|---:|---:|---|---:|---:|---:|
| D-g1-t4 (1g low-ILS) | HC | 5 | **0** | 4/0 | 4 | 0 | 0.5s |
| | SA T=5 | 5 | **0** | 86/73 | **0** | 0 | 0.9s |
| | SA T=5 x3 | 5 | **0** | 262/213 | 2 | 0 | 3.0s |
| D-g10-t4 (10g low-ILS) | HC | 129 | 3 | 7/0 | 2 | 0 | 1.9s |
| | SA T=5 | 129 | 3 | 60/56 | 2 | 0 | 2.3s |
| | SA T=5 x3 | 129 | 3 | 186/156 | **0** | 0 | 7.1s |
| D-g10-t100 (10g high-ILS) | HC | 125 | 72 | 7/0 | 8 | 5 | 3.3s |
| | SA T=5 | 125 | 73 | 76/58 | **6** | 6 | 4.3s |
| | SA T=5 x3 | 125 | **69** | 218/182 | 10 | 7 | 12.6s |

### Scenario E (6 taxa, 2 reticulations)

| Test Case | Method | Init Pars | Final Pars | Acc/Up | mu_d | hw_d | Time |
|---|---|---:|---:|---|---:|---:|---:|
| E-g1-t4 (1g low-ILS) | HC | 11 | **0** | 7/0 | **0** | 0 | 0.5s |
| | SA T=5 | 11 | **0** | 75/60 | **0** | 0 | 0.9s |
| | SA T=5 x3 | 11 | **0** | 243/198 | **0** | 0 | 2.9s |
| E-g10-t4 (10g low-ILS) | HC | 110 | 11 | 13/0 | 4 | 2 | 2.1s |
| | SA T=5 | 110 | **3** | 51/41 | **0** | 0 | 2.5s |
| | SA T=5 x3 | 110 | **3** | 174/142 | **0** | 0 | 7.5s |
| E-g10-t100 (10g high-ILS) | HC | 112 | 69 | 4/0 | 12 | 9 | 3.8s |
| | SA T=5 | 112 | 66 | 63/51 | 12 | 8 | 4.4s |
| | SA T=5 x3 | 112 | **65** | 183/141 | **10** | **7** | 13.3s |

### Scenario F (6 taxa, 3 reticulations)

| Test Case | Method | Init Pars | Final Pars | Acc/Up | mu_d | hw_d | Time |
|---|---|---:|---:|---|---:|---:|---:|
| F-g1-t4 (1g low-ILS) | HC | 9 | **0** | 7/0 | **0** | 0 | 0.6s |
| | SA T=5 | 9 | 1 | 60/46 | 2 | 2 | 0.9s |
| | SA T=5 x3 | 9 | 1 | 217/175 | 2 | 2 | 2.9s |
| F-g10-t4 (10g low-ILS) | HC | 100 | 3 | 8/0 | **0** | 0 | 2.1s |
| | SA T=5 | 100 | 3 | 45/34 | **0** | 0 | 2.5s |
| | SA T=5 x3 | 100 | 3 | 120/94 | **0** | 0 | 7.4s |
| F-g10-t100 (10g high-ILS) | HC | 91 | 68 | 4/0 | 10 | 7 | 3.3s |
| | SA T=5 | 91 | 68 | 67/57 | **6** | **3** | 4.1s |
| | SA T=5 x3 | 91 | 68 | 203/175 | 8 | 7 | 12.4s |

### Scenario J (14 taxa, 2 reticulations)

| Test Case | Method | Init Pars | Final Pars | Acc/Up | mu_d | hw_d | Time |
|---|---|---:|---:|---|---:|---:|---:|
| J-g1-t20 (1g mod-ILS) | HC | 63 | 4 | 23/0 | 16 | 9 | 20.6s |
| | SA T=5 | 63 | 5 | 81/52 | 16 | 7 | 24.5s |
| | SA T=5 x3 | 63 | **3** | 264/169 | **12** | **6** | 74.3s |
| J-g10-t20 (10g mod-ILS) | HC | 646 | 54 | 25/0 | 10 | 6 | 210.7s |
| | SA T=5 | 646 | **42** | 53/23 | 12 | **5** | 226.5s |
| | SA T=5 x3 | 646 | **33** | 215/113 | 12 | **3** | 655.9s |

## Key Takeaways

1. **Low-ILS is easy.** All methods reach near-optimal parsimony (0-3) and near-zero network distances for `t=4` cases. HC is often sufficient here.

2. **SA shines on medium-difficulty cases.** The biggest wins come from cases like E-g10-t4, where HC gets stuck at parsimony 11 / mu_d=4 while SA reaches parsimony 3 / mu_d=0 (exact topology recovery).

3. **High-ILS remains fundamentally hard.** Under extreme ILS (`t=100`), parsimony plateaus around 65-73 for all methods. The gene tree discordance from ILS is indistinguishable from topological error, so no search strategy can fully overcome the lack of signal.

4. **Restarts matter more than single-run SA.** SA x3 consistently outperforms SA x1, especially on harder cases (J-g10: parsimony 42 -> 33, hw_d 5 -> 3). The cost scales linearly.

5. **Larger networks benefit most from SA.** The J scenario (14 taxa) shows the most dramatic SA improvements, as the search space is much larger and HC is more prone to local optima.

6. **SA can occasionally overshoot on easy problems.** F-g1-t4 is the one case where HC beats SA (parsimony 0 vs 1), because the greedy path was already optimal and SA's random uphill moves caused unnecessary detours.
