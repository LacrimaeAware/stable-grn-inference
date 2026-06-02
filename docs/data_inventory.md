# DREAM4 Data Inventory

Raw local data root:

```text
data/raw/dream4/
+-- DREAM4_InSilico_Size10/
|   +-- insilico_size10_1/
|   +-- insilico_size10_2/
|   +-- insilico_size10_3/
|   +-- insilico_size10_4/
|   +-- insilico_size10_5/
+-- DREAM4_InSilico_Size100/
|   +-- insilico_size100_1/
|   +-- insilico_size100_2/
|   +-- insilico_size100_3/
|   +-- insilico_size100_4/
|   +-- insilico_size100_5/
+-- DREAM4_InSilico_Size100_Multifactorial/
+-- DREAM4_InSilicoNetworks_GoldStandard/
|   +-- DREAM4_Challenge2_GoldStandards/
```

The inspected expression files are tab-delimited TSV files with quoted headers. The inspected gold-standard network files are tab-delimited, headerless edge tables.

## File Groups

| Location | File pattern | Likely type | Shape per file | Delimiter |
|---|---|---|---:|---|
| `DREAM4_InSilico_Size10/insilico_size10_*/` | `*_multifactorial.tsv` | Size10 multifactorial expression | 10 x 10 | tab |
| `DREAM4_InSilico_Size10/insilico_size10_*/` | `*_knockouts.tsv` | Size10 knockout expression | 10 x 10 | tab |
| `DREAM4_InSilico_Size10/insilico_size10_*/` | `*_knockdowns.tsv` | Size10 knockdown expression | 10 x 10 | tab |
| `DREAM4_InSilico_Size10/insilico_size10_*/` | `*_timeseries.tsv` | Size10 time-series expression | 105 x 11 | tab |
| `DREAM4_InSilico_Size10/insilico_size10_*/` | `*_wildtype.tsv` | Size10 wildtype expression | 1 x 10 | tab |
| `DREAM4_InSilico_Size10/insilico_size10_*/` | `*_dualknockouts_indexes.tsv` | Dual-knockout index pairs | 5 x 2 | tab |
| `DREAM4_InSilico_Size100/insilico_size100_*/` | `*_knockouts.tsv` | Size100 knockout expression | 100 x 100 | tab |
| `DREAM4_InSilico_Size100/insilico_size100_*/` | `*_knockdowns.tsv` | Size100 knockdown expression | 100 x 100 | tab |
| `DREAM4_InSilico_Size100/insilico_size100_*/` | `*_timeseries.tsv` | Size100 time-series expression | 210 x 101 | tab |
| `DREAM4_InSilico_Size100/insilico_size100_*/` | `*_wildtype.tsv` | Size100 wildtype expression | 1 x 100 | tab |
| `DREAM4_InSilico_Size100/insilico_size100_*/` | `*_dualknockouts_indexes.tsv` | Dual-knockout index pairs | 20 x 2 | tab |
| `DREAM4_InSilico_Size100_Multifactorial/` | `*_multifactorial.tsv` | Size100 multifactorial expression | 100 x 100 | tab |
| `DREAM4_InSilicoNetworks_GoldStandard/DREAM4_Challenge2_GoldStandards/Size 10/` | `DREAM4_GoldStandard_InSilico_Size10_*.tsv` | Size10 gold-standard directed edge labels | 90 x 3 | tab |
| `DREAM4_InSilicoNetworks_GoldStandard/DREAM4_Challenge2_GoldStandards/Size 100/` | `DREAM4_GoldStandard_InSilico_Size100_*.tsv` | Size100 gold-standard directed edge labels | 9900 x 3 | tab |
| `DREAM4_InSilicoNetworks_GoldStandard/DREAM4_Challenge2_GoldStandards/Size 100 multifactorial/` | `DREAM4_GoldStandard_InSilico_Size100_multifactorial_*.tsv` | Size100 multifactorial gold-standard directed edge labels | 9900 x 3 | tab |

## Size10 Files

Each of the five Size10 network folders contains:

- `*_dualknockouts_indexes.tsv`
- `*_knockdowns.tsv`
- `*_knockouts.tsv`
- `*_multifactorial.tsv`
- `*_timeseries.tsv`
- `*_wildtype.tsv`

The simplest file for the first expression-loading step is:

```text
data/raw/dream4/DREAM4_InSilico_Size10/insilico_size10_1/insilico_size10_1_multifactorial.tsv
```

It is a plain 10-row by 10-column expression matrix with gene columns `G1` through `G10`.

First rows:

```text
"G1"      "G2"      "G3"      "G4"      "G5"      "G6"      "G7"      "G8"      "G9"      "G10"
0.7472648 0.0145174 0.2768379 0.5924924 0.1137333 0.4199753 0.6704619 0.7873434 0.5851015 0.6379321
0.1952416 0.0666675 0.4668403 0.7665067 0.5201448 0.2495804 0.3279500 0.7170845 0.7695890 0.6548019
0.7684065 0.1396432 0.1299352 0.6305152 0.0016216 0.2795217 0.5784058 0.6209451 0.5909760 0.6678004
```

The Size10 time-series files are also clear expression files. They contain a `Time` column plus 10 gene columns. Pandas reads each file as 105 rows by 11 columns, which appears consistent with five 21-point trajectories separated by blank lines.

First rows of `insilico_size10_1_timeseries.tsv`:

```text
"Time" "G1"      "G2"      "G3"      "G4"      "G5"      "G6"      "G7"      "G8"      "G9"      "G10"
0.0    0.6665114 0.1272186 0.3550646 0.7745716 0.1004299 0.2754930 0.6067846 0.7430983 0.6656366 0.6950638
50.0   0.3257748 0.1218223 0.3464115 0.7229108 0.1924591 0.3107637 0.6096963 0.7567515 0.5554138 0.7327167
100.0  0.1775012 0.0443587 0.5712888 0.5868280 0.2333497 0.3569736 0.4647324 0.6656993 0.7211032 0.6717156
```

The dual-knockout index files are not gold-standard edge files. They appear to identify perturbed gene pairs:

```text
"G_i" "G_j"
7     10
1     10
1     7
3     9
```

## Expression Data Candidates

Expression data candidates are:

- Size10: `multifactorial`, `knockouts`, `knockdowns`, `timeseries`, and `wildtype` files in each `insilico_size10_*` folder.
- Size100: `knockouts`, `knockdowns`, `timeseries`, and `wildtype` files in each `insilico_size100_*` folder.
- Size100 multifactorial: five top-level `insilico_size100_*_multifactorial.tsv` files.

For the first baseline, the simplest expression file is the Size10 network 1 multifactorial matrix listed above.

## Gold-Standard Edge Files

Gold standards are the answer keys for network topology: each row is a directed candidate regulator-target pair plus a binary label indicating whether that edge exists in the reference network. The expression files are the input data used to infer or rank candidate edges.

The Size10 gold-standard files are:

- `DREAM4_GoldStandard_InSilico_Size10_1.tsv`
- `DREAM4_GoldStandard_InSilico_Size10_2.tsv`
- `DREAM4_GoldStandard_InSilico_Size10_3.tsv`
- `DREAM4_GoldStandard_InSilico_Size10_4.tsv`
- `DREAM4_GoldStandard_InSilico_Size10_5.tsv`

The matching gold-standard file for `insilico_size10_1_multifactorial.tsv` is:

```text
data/raw/dream4/DREAM4_InSilicoNetworks_GoldStandard/DREAM4_Challenge2_GoldStandards/Size 10/DREAM4_GoldStandard_InSilico_Size10_1.tsv
```

This file is a `90 x 3` tab-delimited table with no header. For 10 genes, 90 rows corresponds to all directed non-self gene pairs. The columns are interpreted as:

```text
source target is_true
```

First rows:

```text
G1 G2 1
G1 G3 1
G1 G4 1
G1 G5 1
G3 G4 1
G3 G7 1
G4 G3 1
G6 G2 1
```

The Size100 gold-standard files have the same headerless three-column format with `9900` rows, corresponding to all directed non-self pairs for 100 genes.

The `Size 10 bonus round` dual-knockout files are not the primary topology answer keys for the first baseline. They are kept separate from the gold-standard directed edge label files.
