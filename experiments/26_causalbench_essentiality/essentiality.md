# Experiment 26: perturbation essentiality and cascade position (RPE1)

## Question

Most knockouts of essential genes trigger the same convergent cell-cycle response. Rather than removing that dominant signal, this experiment uses it: rank genes by how strongly their knockout disrupts the system (a data-derived essentiality), and order genes by their position in the resulting cascade.

## Method

Four essentiality measures per perturbed gene g, on the 651-gene response matrix:
- magnitude: ||Δ_g||, the size of the whole-transcriptome response to knocking out g.
- cascade: |Δ_g · cascade_axis|, projection onto the dominant response component.
- breadth: number of genes responding to perturbing g above a control-null threshold.
- centrality: connectivity of g in the control-cell correlation network.

Cascade position: net_out(g) = mean over h of (|Δ_g[h]| − |Δ_h[g]|), the degree to which perturbing g affects others more than they affect g. High net_out is upstream. This reuses the directional asymmetry that was reproducible in experiment 21.

Validation uses no external labels: cross-measure rank agreement, split-half reproducibility across independent cell halves, and the relation to the number of surviving cells per perturbation.

## Part 1: essentiality

Cross-measure rank agreement (Spearman):

| | magnitude | cascade | breadth | centrality |
| --- | --- | --- | --- | --- |
| magnitude | 1.00 | 0.80 | 0.89 | 0.26 |
| cascade | 0.80 | 1.00 | 0.79 | 0.12 |
| breadth | 0.89 | 0.79 | 1.00 | 0.12 |
| centrality | 0.26 | 0.12 | 0.12 | 1.00 |

- The three interventional measures (magnitude, cascade, breadth) agree strongly. The observational correlation-network centrality is largely independent of them, consistent with the earlier finding that observational structure does not track interventional response.
- Split-half reproducibility: magnitude ranking Spearman 0.970, breadth ranking 0.915.
- Essentiality (magnitude) versus surviving cells per perturbation: Spearman -0.348. Stronger responses come from more-depleted perturbations, consistent with essential-gene knockouts being selected against.
- Top 20 by mean rank: SFPQ, HSPA9, SNRNP200, RPL35A, RPL7, CSE1L, RPL23, HSPE1, PSMC5, RUVBL1, CDT1, NUP98, NUP107, SF3B3, KPNB1, SRSF7, DDX23, RPA3, SNRPC, EIF3B. These are recognizable core machinery: ribosomal proteins (RPL7, RPL23, RPL35A), spliceosome (SNRNP200, SF3B3, SNRPC), proteasome (PSMC5), nuclear pore (NUP98, NUP107), translation initiation (EIF3B), nuclear transport (KPNB1, CSE1L), and replication (CDT1, RPA3).

## Part 2: cascade position

- net_out split-half reproducibility: Spearman 0.986.
- Essentiality (magnitude) versus upstream score (net_out): Spearman 0.463. The two axes are related but distinct.
- Most upstream (high net_out): SART1, PSMB5, EIF2B3, SNRNP200, PRPF3, EIF2B5, PRPF31, THAP1, EFTUD2, EIF2B2, DDX23, TRAPPC11, RBMX2, MTOR, WDR70. These are translation initiation (EIF2B subunits), splicing (PRPF3, PRPF31, EFTUD2, SART1), proteasome (PSMB5), and growth control (MTOR).
- Most downstream (low net_out): H2AFZ, TUBA1B, HMGB1, PTMA, NASP, TUBB, RRM2, CCNB1, DTYMK, H2AFX, DNMT1, HSP90B1, CENPW, HNRNPM, ENO1. These are cell-cycle and structural effectors: cyclins (CCNB1), tubulins (TUBA1B, TUBB), histones (H2AFZ, H2AFX), replication (RRM2, DNMT1, DTYMK).

## Summary

- The four essentiality measures agree at mean Spearman 0.497, and the ranking is reproducible across cell halves at 0.970. A single data-derived essentiality axis is well-defined on this data.
- The data-derived essentiality recovers recognizable essential machinery (ribosome, spliceosome, proteasome, nuclear pore, translation).
- Cascade position is highly reproducible (0.986) and separates upstream information-processing machinery (translation initiation, splicing, proteasome, MTOR) from downstream cell-cycle and structural effectors (cyclins, tubulins, histones, replication). It is moderately correlated with essentiality (0.46) but distinct.

## Limitations

- Both axes partly recover known biology (essential machinery is annotated, and the cell-cycle program is known), so this validates that the data and measures behave sensibly more than it discovers new structure.
- External validation against annotated essentiality (for example DepMap gene-effect scores) is the natural next step and is not done here (no external download).
- The upstream/downstream ordering is a coarse partial order from net effect, not a verified causal sequence.

Artifacts (git-ignored `results/`): `causalbench_essentiality_genes.csv`, `causalbench_essentiality_summary.csv`, `causalbench_essentiality_debug_report.md`, `results/figures/causalbench_essentiality.png`.
