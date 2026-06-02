# GeneNetWeaver (GNW) Setup

How GNW is built and run in this repo, reproducibly, from a clean GNW 3.0 source
download. This is a setup + smoke-test record; no large simulations were run.

## What was found

The download under `tools/gnw-3.0-src/` is the **GNW 3.0 source distribution**, not
a runnable application:

```
tools/gnw-3.0-src/
├── README, NEWS, LICENSE        # NEWS: 3.0 added a "Command-line interface (standalone version)"
├── bin/                         # EMPTY (nothing prebuilt)
├── lib/                         # ~28 dependency jars (JSAP, JUNG, colt, sbml2, log4j, ...)
│   └── gnw-networks.jar         # bundled DREAM3/DREAM4 network XMLs (data, NOT the app)
└── src/                         # 126 .java files (ch.epfl.lis.*, com.swtdesigner)
```

- **No runnable jar / jnlp / bat / shell script exists.** `bin/` is empty and no
  `lib/*.jar` is the GNW application (they are dependencies + bundled network data).
- There is **no `build.xml` / `pom.xml` / `Makefile`** - GNW 3.0 was an Eclipse
  project (note the `com.swtdesigner` package). So it must be compiled manually.
- Entry points (from `grep "static void main"`):
  - **CLI / standalone:** `ch.epfl.lis.gnw.GnwMain`
  - **GUI:** `ch.epfl.lis.gnwgui.Main`

### Why it is source, not a jar

The "standalone" download is the source tree; the prebuilt application jar is a
separate artifact that was not downloaded. Because only source was provided, GNW
must be compiled before it can run.

## Java

- `java` / `javac` are **not on PATH** and `JAVA_HOME` is **not set**.
- A usable **JDK 17** was found and is auto-discovered by the wrapper:
  `C:\Program Files (x86)\Android\openjdk\jdk-17.0.14` (OpenJDK 17.0.14 LTS, has `javac`).

## JDK 17 compatibility patches (and why)

GNW 3.0 (2010) uses two JDK-internal APIs **removed in Java 9+**, so it does not
compile as-is on JDK 17. Both are in GUI export code, unused by the CLI:

| File | Removed API | Replacement |
|---|---|---|
| `gnwgui/NetworkGraph.java` | `com.sun.image.codec.jpeg.{JPEGCodec,JPEGImageEncoder}` | `javax.imageio.ImageIO` |
| `gnwgui/NetworkDesktopMap.java` | `com.sun.org.apache.xerces.internal.*`, `...xml.internal.serialize.{OutputFormat,XMLSerializer}` | `javax.xml.parsers` + `javax.xml.transform` |

After these replacements the full source compiles cleanly on JDK 17
(`javac --release 8`, **301 classes, 0 errors**). Building instead with a **JDK 8**
needs no patches (the `com.sun.*` internals still exist there).

### Reproducibility: patches are tracked and auto-applied

The GNW source tree (`tools/gnw-3.0-src/`) stays **git-ignored** (external download,
includes `.svn` metadata + build artifacts). The patches are therefore kept
**outside** the ignored tree and applied at build time:

- `tools/gnw_patches/apply_jdk17_patches.ps1` - tracked, **idempotent**, whitespace-
  tolerant. It rewrites the two GUI files from the original `com.sun.*` code to the
  standard APIs. Run on an already-patched tree it matches nothing and is a no-op.
- `tools/run_gnw.ps1 -Build` calls this patcher automatically before `javac`.

Verified end-to-end: reverting the two files to their pristine (com.sun) state and
running `.\tools\run_gnw.ps1 -Build` re-applies the patches (`NetworkGraph.java:
patched`, `NetworkDesktopMap.java: patched`) and compiles to 301 classes; a second
`-Build` reports `already patched / no change` and still compiles. So a clean
re-download + `-Build` reproduces a working GNW.

## How to rebuild GNW from a clean source download

1. Place the GNW 3.0 source under `tools/gnw-3.0-src/` (so `tools/gnw-3.0-src/src`
   and `tools/gnw-3.0-src/lib` exist).
2. Run:
   ```powershell
   .\tools\run_gnw.ps1 -Build
   ```
   This auto-applies `tools/gnw_patches/apply_jdk17_patches.ps1`, compiles with the
   discovered JDK (`javac --release 8`), and copies non-source resources (e.g.
   `gnwguiLogSettings.txt`, needed at startup) into `tools/gnw-3.0-src/build/classes`.

No `JAVA_HOME`/PATH setup is required - the wrapper finds the JDK. (If you prefer a
specific JDK, set `JAVA_HOME`; building needs a full JDK, not just a JRE.)

## How to run the wrapper

```powershell
.\tools\run_gnw.ps1            # safe smoke test: prints CLI usage (default --help)
.\tools\run_gnw.ps1 -Build     # (re)build, then print usage
.\tools\run_gnw.ps1 -Gui       # launch the GUI (opens a window; not run in this smoke test)
.\tools\run_gnw.ps1 --settings tools\gnw_settings\smoke_size10_timeseries.properties   # see caveat below
```

Java discovery, the safe `--help` default, and the CLI smoke test were confirmed:

```
Usage: java -jar gnw.jar
                [-h|--help] [(-s|--settings) <file>] [--regulondb <file>]
```

CLI flags (`ch.epfl.lis.gnw.GnwMain`, JSAP-based): `-h/--help`, `-s/--settings <file>`,
`--regulondb <file>`. The `Could not load settings file.` line is a harmless warning
when no `-s` file is given; usage still prints and the process exits 0.

## Why no-argument `GnwMain` is dangerous

`GnwMain.run()` does: if `--help` -> print usage; else if `--regulondb` -> parse a
RegulonDB file; **else -> `dream5()`**, which calls
`BenchmarkGeneratorDream5.generateBenchmark()`. That method is hardcoded to
`generateEcoliInsilicoCompendium("net1")` - it loads the full E. coli RegulonDB
transcriptional network and generates a **DREAM5-scale in-silico compendium**.

So running `GnwMain` with **no arguments** (or with only `-s settings`) starts a
large generation. `tools/run_gnw.ps1` guards against this by defaulting to `--help`
when no CLI arguments are supplied; pass explicit arguments for real runs.

## Tiny smoke generation: SKIPPED (documented, not forced)

A tiny CLI generation was **not run**, on purpose, because GNW 3.0's command-line
tool cannot do one safely:

- The only generation path in `GnwMain` is the **hardcoded DREAM5 E. coli
  compendium** (large). The `-s/--settings` flag only loads global simulation
  parameters (`GnwSettings`); it does **not** select a smaller/different source
  network or change the generation target. There is no CLI action of the form
  "take this small network + these settings -> emit a tiny benchmark".
- Even attempting it would fail here: the DREAM5 inputs it expects
  (`dream5/ecoli/ecoli_experiments.tsv`, `..._experiment_defs.tsv`, the RegulonDB
  network) are **not shipped** in this source download.
- Flexible, size-controlled generation (extract an N-gene subnetwork, choose
  time-series length, simulate, export) is a **GUI workflow** in GNW 3.0
  (`ch.epfl.lis.gnwgui.Main`), not a CLI feature.

What WAS produced/verified instead: a clean reproducible build, Java discovery, the
safe `--help` default, and the CLI usage smoke test (all above).

### Settings format (known) and the tiny profile

The settings format is **not** the blocker - it is a plain Java `.properties` file
(defaults in `src/ch/epfl/lis/gnw/rsc/settings.txt`). A minimal tiny time-series
profile is provided at:

```
tools/gnw_settings/smoke_size10_timeseries.properties
```

(1 short DREAM4-style trajectory of 3 points, ODE-only, no noise). It is ready for
the **GUI** workflow or a future custom CLI entry point, but the bundled GNW 3.0 CLI
will not consume it for a standalone tiny generation (see above).

### What is missing / manual next steps for real GNW generation

To generate tiny networks/time-series with GNW 3.0, one of:

1. **GUI:** `.\tools\run_gnw.ps1 -Gui` -> open/extract a small (e.g. 10-gene)
   subnetwork from a source network (GNW ships DREAM3/DREAM4 networks in
   `lib/gnw-networks.jar`), load `tools/gnw_settings/smoke_size10_timeseries.properties`,
   run DREAM4-style time series, and export to an output folder. Output (when
   produced) should go to `results/gnw_smoke/` (git-ignored).
2. **Custom small CLI entry point:** add a tiny Java main that loads a source
   network + a settings file and calls the existing simulation code, then build via
   `-Build`. (Not done here; would be a code addition to GNW, beyond setup.)
3. **Prebuilt jar:** download the GNW standalone application jar and point the
   wrapper at it (`java -jar gnw.jar ...`). The GUI workflow is still required for
   custom small networks.

The expected smoke output folder `results/gnw_smoke/` is git-ignored (under the
existing `/results/` ignore) and was not created because no generation ran.

## GUI status

`ch.epfl.lis.gnwgui.Main` compiles and is launchable via `.\tools\run_gnw.ps1 -Gui`,
but it was **not** started in this smoke test (it opens a blocking Swing window). Run
it interactively on a desktop session to use the generation workflow.

## What is tracked vs ignored

- Tracked: `tools/run_gnw.ps1`, `tools/gnw_patches/apply_jdk17_patches.ps1`,
  `tools/gnw_settings/smoke_size10_timeseries.properties`, this document.
- Ignored: `tools/gnw-3.0-src/` (the external source + `build/` artifacts) and
  `results/` (any generated output).
- Re-applying patches after a fresh download is automatic via `-Build`; the patch
  script is idempotent.
