#!/usr/bin/env python3
"""Preserve submitted inputs and prepare an explicitly unfinished paper copy."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
ROUND2 = ROOT / 'DASIP_round2'
SOURCE = ROOT / 'DASIP paper'


ABSTRACT_SOURCE = ROUND2 / 'paper_round2_placeholder/abstracts/ABSTRACT_2026-10-20.txt'
ABSTRACT = '\\begin{abstract}\n' + ABSTRACT_SOURCE.read_text().strip().replace('%', r'\%') + r'''
\keywords{Neural RGB-D SLAM \and Embedded GPUs \and Hash-Grid Encoding
\and Depth-Aware Point Pruning}
\end{abstract}'''

INTRODUCTION = r'''\section{Introduction}
\label{sec:introduction}
Co-SLAM~\cite{wang2023coslam} and EC-SLAM~\cite{li2026ecslam} jointly estimate
camera motion and scene geometry using multiresolution hash grids and small
neural decoders. Repeated sampling, field queries, rendering and optimisation
make query locality and query count relevant to desktop and embedded GPUs.

Complete execution also includes ray preparation, backward propagation,
regularization and pose finalization. Their costs vary across systems, phases
and power modes; an isolated kernel improvement need not accelerate a complete
call. We therefore distinguish kernel profiles, optimizer-iteration CUDA ranges
and enclosing synchronized host-wall measurements.

Memory-coherent execution combines Morton point ordering and hash mapping.
Our conservative Co-SLAM pruning instead selects buffered support on the
native point lattice and periodically restores complete support. It retains
native rays, BA constraints and geometric supervision but approximates
rendering, requiring explicit speed--quality evaluation.

The scoped contributions are:
\begin{enumerate}
\item A kernel-to-call study across two neural SLAM systems, a desktop GPU
and two embedded power modes, with explicit measurement boundaries.
\item Integration of established Morton mechanisms with native precision and
capacity settings; engineering changes remain a separate ablation.
\item Conservative Co-SLAM pruning evaluated across Replica, Synthetic, TUM
and ScanNet using paired seeds and dataset-appropriate quality metrics,
including negative outcomes and individual-scene exceptions.
\end{enumerate}

'''

CHARACTERIZATION = r'''\section{Shared Workloads and Cross-System Bottleneck Characterisation}
\label{sec:characterization}
\subsection{Shared Optimisation Structure}
\label{sec:coslam_pipeline}
Both systems alternate camera tracking with scene updates: global bundle
adjustment in Co-SLAM and mapping in EC-SLAM. An iteration denotes one
optimizer update, not a complete frame or an enclosing estimator call.
Both phases select observations, sample points, query encodings and decoders,
render predictions, evaluate losses, backpropagate and update parameters.
Native Co-SLAM scene configurations are preserved: the selected Replica,
Synthetic and ScanNet settings use ten BA updates, whereas TUM uses twenty;
tracking uses ten. BA and keyframe insertion occur every fifth frame.

The point-pruning extension is restricted to Co-SLAM. EC-SLAM's inspected
native renderer already places depth-conditioned free-space samples and
surface-centred points. Our extension instead selects a subset of Co-SLAM's
original lattice, preserves its geometry-loss normalization, and periodically
restores its complete rendering support. We do not port the Co-SLAM pruning
module into EC-SLAM.

\subsection{Profiling Methodology}
\label{sec:profiling_method}
NVIDIA Nsight Compute characterises selected kernels through cycle counts,
memory traffic, throughput and coalescence indicators~\cite{nvidia_nsight_compute}.
Inherited CUDA-event iteration/component measurements remain available.
Enclosing events and synchronized host-wall timers additionally measure
complete tracking and BA calls, from selection/preprocessing through pose
finalization~\cite{nvidia_cuda_events}. Iteration sums are labelled separately;
they are not substituted for complete-call wall time. Primary stage means
exclude frames 0--19, with all-call means retained as a sensitivity check.
Here ms/call covers all stage iterations, setup and pose finalization; it is
distinct from per-iteration, complete-frame or full-scene elapsed time. EC active
worker calls exclude handshake waits and may overlap; they cannot be summed
to infer frame latency or FPS.

The hardware comparison includes RTX\,4070 and Jetson AGX Orin in MAXN and
15\,W modes~\cite{nvidia_orin_power_modes}. MAXN is evaluated first; the author
changes the mode manually before repeating matching workloads at 15\,W. Each
queue verifies a single mode and records clock policy. Kernel profiling and
mesh extraction do not overlap a timed GPU job.

\subsection{Cross-Regime Bottleneck Analysis}
\label{sec:regime_analysis}
Table~\ref{tab:profiling_regimes} reserves the component decomposition.
Absolute component times and shares must come from compatible reference
settings and disjoint ranges. The former submission's percentages are not
carried forward as measurements under restored precision/capacity settings.
For Co-SLAM, office0 at H16/H19 isolates capacity within one Replica scene;
fr1\_desk at native H16 provides a real-sensor/native-configuration comparison.
Complete sequences preserve online map history. After frames 0--19, each
chronological third contributes 100 evenly spread tracking calls and 20 BA
calls. Third means are weighted by eligible native call counts. Co profiles
use three RTX seeds and one run per Orin mode, without across-seed SD there.
EC office0 H16/H19 follows the same temporal procedure with native concurrent
workers and eight tracking/ten mapping updates. Supporting iteration and call
partitions are collected separately from forward shares.

\begin{table}[htbp]\centering\scriptsize
\caption{Planned forward decomposition: component CUDA-event ms and percentage
of the same complete forward interval; total is ms per forward evaluation.
Co parent: JointEncoding.forward including its internal supervised terms;
Other includes positional encoding, normalization/glue and internal loss
assembly. External smoothness, backward/optimizer steps and call
setup/finalization are excluded. Co workloads and temporal sampling are
specified above; existing EC baseline placeholders remain separate. All
unmeasured cells are \pending{}, not zero.}
\label{tab:profiling_regimes}
\resizebox{\linewidth}{!}{\begin{tabular}{llllrrrrr}
\toprule System&Platform&Workload&Stage&Hash grid&Decoder&\shortstack{Sampling/\\rendering}&Other&Total\\\midrule
\multirow{18}{*}{Co-SLAM}&\multirow{6}{*}{RTX\,4070}&office0 H16&TR&\pending&\pending&\pending&\pending&\pending\\
&&&BA&\pending&\pending&\pending&\pending&\pending\\
&&office0 H19&TR&\pending&\pending&\pending&\pending&\pending\\
&&&BA&\pending&\pending&\pending&\pending&\pending\\
&&fr1\_desk H16&TR&\pending&\pending&\pending&\pending&\pending\\
&&&BA&\pending&\pending&\pending&\pending&\pending\\\addlinespace
&\multirow{6}{*}{MAXN}&office0 H16&TR&\pending&\pending&\pending&\pending&\pending\\
&&&BA&\pending&\pending&\pending&\pending&\pending\\
&&office0 H19&TR&\pending&\pending&\pending&\pending&\pending\\
&&&BA&\pending&\pending&\pending&\pending&\pending\\
&&fr1\_desk H16&TR&\pending&\pending&\pending&\pending&\pending\\
&&&BA&\pending&\pending&\pending&\pending&\pending\\\addlinespace
&\multirow{6}{*}{15\,W}&office0 H16&TR&\pending&\pending&\pending&\pending&\pending\\
&&&BA&\pending&\pending&\pending&\pending&\pending\\
&&office0 H19&TR&\pending&\pending&\pending&\pending&\pending\\
&&&BA&\pending&\pending&\pending&\pending&\pending\\
&&fr1\_desk H16&TR&\pending&\pending&\pending&\pending&\pending\\
&&&BA&\pending&\pending&\pending&\pending&\pending\\\midrule
\multirow{6}{*}{EC-SLAM}&\multirow{2}{*}{RTX\,4070}&Replica H16&TR&\pending&\pending&\pending&\pending&\pending\\
&&&Map&\pending&\pending&\pending&\pending&\pending\\
&\multirow{2}{*}{MAXN}&Replica H16&TR&\pending&\pending&\pending&\pending&\pending\\
&&&Map&\pending&\pending&\pending&\pending&\pending\\
&\multirow{2}{*}{15\,W}&Replica H16&TR&\pending&\pending&\pending&\pending&\pending\\
&&&Map&\pending&\pending&\pending&\pending&\pending\\\bottomrule
\end{tabular}}
\end{table}

\subsection{Hash-Grid Kernel Characterisation}
\label{sec:hashgrid_profile}
Spatially neighbouring queries can gather features from distant table entries.
This motivates measuring cache/memory behaviour rather than inferring it from
nominal hardware throughput~\cite{li2023instant3d}. Kernel ablations retain
four conditions: A, prime hash/native order; B, Morton hash/native order; C,
prime hash/Morton order; D, Morton hash/Morton order. Complete-pipeline
comparisons use A versus D, labelled BASE versus MC. Supporting cycle/traffic
records, including the submitted 15\,W claim, remain \pending{} until recovered
or remeasured with their actual configuration metadata.

'''

METHOD = r'''\subsection{Conservative Depth-Point Pruning}
\label{sec:depth_reduction}
We replace the earlier point redistribution and depth-homogeneous BA buckets
with buffered support selection on the \emph{native} stratified lattice. The
native ray batches, mixed keyframe/current-frame draws, optimized pose
population, loss weights and dense feature smoothness remain unchanged.
The method applies to field-query points within rays; it does not prune whole
observations or keyframe constraints.

For observed depth $d_r$, native sample index $j$ has stratified interval
$[\ell_{rj},u_{rj}]$ and realized position
$z_{rj}=\ell_{rj}+\xi_{rj}(u_{rj}-\ell_{rj})$.
We generate the complete native lattice and random draws before selection.
On an intermediate pruned update we retain
\begin{equation}
 m_{rj}=\mathbf1\!\left[
 \neg\operatorname{valid}(d_r)\ \lor\ j<2\ \lor\
 \ell_{rj}\leq d_r+\tau+b\right],
 \label{eq:depth_sample_count}
\end{equation}
where indices are zero-based, $\operatorname{valid}(d)=\operatorname{finite}(d)
\land d>0$, $\tau$ is native geometric truncation in scene units, and $b=0.15$
is a shared buffer. Selected configurations have scale factor one. Finite
missing-depth rays retain their full native near--far support. Nonfinite-depth
retention is not a replacement for input sanitization.

Every valid sample with native free-space or SDF supervision obeys
$\ell_{rj}\leq z_{rj}\leq d_r+\tau$, so it is retained. Outputs are restored
to the full native layout before the unchanged geometry masks, balancing
factors and reductions are evaluated. For fixed inputs and field parameters,
this preserves those supervised terms mathematically, up to floating-point
effects. It does not preserve the entire subsequent optimization trajectory.
The policy does not necessarily preserve every depth-centred point: TUM's
half-window is 0.25 whereas $\tau+b=0.20$.

Rendering is approximated. Omitted slots receive zero weight, and an adjacent
SDF sign-crossing pair is admissible only when both native sample indices were
queried. Retained rendering weights are normalized with the native convention.
This can change colour/depth predictions and their pose/map gradients; no
opacity, ATE or convergence bound is asserted.

For a call with $T$ updates, full support is used at $i=0$, $i=T-1$, and
$i\bmod3=0$. Thus ten-update calls use dense indices $\{0,3,6,9\}$; first-frame
mapping remains dense throughout. Tracking may cache interval-support indices
for its fixed ray/depth batch while regenerating jitter and posed coordinates.
BA recomputes support for each native mixed draw. No depth bucket or pose cohort
is introduced.

The mechanism combines depth-informed support allocation and periodic dense
updates within the original Co-SLAM sampling objective. It differs from
EC-SLAM's native depth-conditioned point placement and does not reuse that
sampler. Its value for this revision is established only if complete-call
measurements and quality evaluation support the resulting trade-off.

'''

RESULTS = r'''\section{Results}
\label{sec:results}
\subsection{Experimental Setup and Evidence Status}
We restore pinned public settings, including requested FP32 encodings,
keyframe storage fractions and scene-specific hash capacities. Replica uses
exponent 16 in the native comparison; 19 is a labelled sensitivity condition.
Native rays, iterations, architecture, losses and smoothness are retained.
Source hashes, seed values, effective configurations and launch arguments
accompany every run. BASE, MC, DP and MC+DP distinguish dense reference,
Morton execution, conservative point pruning and their composition.
Engineering ray handling/pose commit is a separate BASE-versus-ENG ablation.

RTX quality uses three paired seeds (0, 1, 2), seeded once before construction
with ordinary within-run RNG evolution. Orin timing initially uses two paired
seeds (0, 1) and alternating condition order. We report means and sample SD;
frames are not treated as independent repetitions. The first launched RTX
matrix contains BASE/DP on two scenes per dataset. Additional Replica scenes,
MC/MC+DP, engineering and capacity conditions are not claimed complete.

TUM and ScanNet use ATE only. Replica and Synthetic also use checked
reconstruction metrics. Final meshes are exported after all stage timings;
metric scoring uses a common evaluator with isolated GT/reconstruction
rendering and versioned cache provenance. This scoring validation is still
pending. The primary Replica aggregate includes all eight scenes, including
office4; an exclusion mean is a labelled sensitivity, not the primary result.

\subsection{Kernel and Complete-Call Runtime}
\begin{table}[htbp]\centering\small
\caption{Isolated hash-kernel placeholders. A/B/C/D are kernel ablations only;
full-pipeline comparisons use BASE/MC. Report absolute values and metadata,
including system, scene, phase, precision, capacity, power/clock policy.}
\label{tab:hashgrid_ablation}
\begin{tabular}{lcccc}\toprule Metric&A&B&C&D\\\midrule
Cycles&\pending&\pending&\pending&\pending\\
L2 sectors&\pending&\pending&\pending&\pending\\
Memory throughput&\pending&\pending&\pending&\pending\\
Kernel time&\pending&\pending&\pending&\pending\\\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]\centering\small
\caption{Complete-call runtime placeholders. Each cell will contain tracking /
BA (Co-SLAM) or tracking / mapping (EC-SLAM) absolute wall times with SD; paired
reductions are derived from matched means. CUDA-event results are retained
alongside wall time. DP applies only to Co-SLAM.}
\label{tab:runtime_round2}
\begin{tabular}{llccc}\toprule System&Condition&RTX\,4070&Orin MAXN&Orin 15\,W\\\midrule
Co-SLAM&BASE&\pending&\pending&\pending\\
&MC&\pending&\pending&\pending\\
&DP&\pending&\pending&\pending\\
&MC+DP&\pending&\pending&\pending\\
EC-SLAM&BASE&\pending&\pending&\pending\\
&MC&\pending&\pending&\pending\\\bottomrule
\end{tabular}
\end{table}

Point-count reduction is reported separately from speed. Mask construction,
compaction/gather/scatter, rendering, dense feature smoothness and pose work
can offset field savings. Percentage time reduction is
$100(1-T_{\rm variant}/T_{\rm BASE})$; the speedup factor is separately
$T_{\rm BASE}/T_{\rm variant}$. Results must identify their scene coverage and
must not generalize a favourable single scene to a dataset-wide gain.

\begin{table}[htbp]\centering\small
\caption{Explicit capacity and engineering sensitivity placeholders. Native
Replica capacity is 16. Engineering benefit is not attributed to DP or MC.}
\label{tab:capacity_engineering}
\begin{tabular}{llcc}\toprule Comparison&Condition&Tracking&BA/mapping\\\midrule
Replica capacity 16&BASE versus MC&\pending&\pending\\
Replica capacity 19&BASE versus MC&\pending&\pending\\
Engineering isolation&BASE versus ENG&\pending&\pending\\
Claimed composition&interaction check&\pending&\pending\\\bottomrule
\end{tabular}
\end{table}

\subsection{ATE and Reconstruction Quality}
Table~\ref{tab:ate_round2} reserves per-scene BASE/DP comparisons. Cells remain
pending until all three paired seeds are available and the trajectory
convention is checked consistently. Direct and relative-pose-corrected ATE
are both saved; the final table must declare which is primary.

\begin{table}[htbp]\centering\small
\caption{ATE RMSE placeholders: three-seed mean $\pm$ sample SD, in cm.
Change is computed from the scene means. The initial two-scene-per-dataset
queue is running; the other six Replica scenes remain additional coverage.}
\label{tab:ate_round2}
\begin{tabular}{llccc}\toprule Dataset&Scene&BASE&DP&Change (\%)\\\midrule
Replica&office0&\pending&\pending&\pending\\
&office1&\pending&\pending&\pending\\
&office2&\pending&\pending&\pending\\
&office3&\pending&\pending&\pending\\
&office4&\pending&\pending&\pending\\
&room0&\pending&\pending&\pending\\
&room1&\pending&\pending&\pending\\
&room2&\pending&\pending&\pending\\\midrule
Synthetic&thin geometry&\pending&\pending&\pending\\
&complete kitchen&\pending&\pending&\pending\\\midrule
TUM&fr1\_desk&\pending&\pending&\pending\\
&fr3\_office&\pending&\pending&\pending\\\midrule
ScanNet&scene0106&\pending&\pending&\pending\\
&scene0000&\pending&\pending&\pending\\\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]\centering\small
\caption{Replica/Synthetic reconstruction placeholders for the common checked
evaluator. Fill separate dataset panels, explicit scene coverage and SD.
TUM/ScanNet have no reconstruction panel. Completion ratio uses percentage
points for absolute changes. Report the all-eight-scene Replica result and a
separately labelled office4-exclusion sensitivity.}
\label{tab:quality_round2}
\begin{tabular}{lcccccc}\toprule
&\multicolumn{4}{c}{Co-SLAM}&\multicolumn{2}{c}{EC-SLAM}\\
Metric&BASE&MC&DP&MC+DP&BASE&MC\\\midrule
Accuracy (cm)&\pending&\pending&\pending&\pending&\pending&\pending\\
Completion (cm)&\pending&\pending&\pending&\pending&\pending&\pending\\
Completion ratio (\%)&\pending&\pending&\pending&\pending&\pending&\pending\\
Depth L1 (cm)&\pending&\pending&\pending&\pending&\pending&\pending\\
\bottomrule\end{tabular}
\end{table}

The research target of low per-scene mean ATE degradation is not an outcome.
Neither short diagnostics nor historical measurements under different defaults
establish the restored-reference variant's quality or wall-time acceptance.
Negative or smaller gains remain visible. Engineering and numerical-correctness
changes are not silently folded into scientific pruning comparisons.

'''

DISCUSSION = r'''\section{Discussion and Conclusion}
\label{sec:conclusion}
The measured benefit depends on the fraction of SLAM work an intervention
changes. Morton execution affects query locality; DP removes rendering queries;
the separate Co engineering ablation reduces data/pose work. Component gains
do not establish proportional complete-call gains.

DP retains geometric supervision for a fixed update but approximates rendering.
Noisy depth, erroneous crossings and omitted support can affect ATE and meshes;
scene exceptions remain visible. Dense smoothness, map updates and growing
pose history limit the removable work. Earlier point redistribution and
depth-homogeneous BA batches are plausible degradation mechanisms, not
experimentally isolated causal conclusions.

Completed RTX and MAXN measurements use the labelled boundaries; EC scopes
differ across platforms. The 15\,W, disjoint profiling and joint-method evidence
remain pending. The submission abstract states the scientific scope without
quantitative claims.

'''


def replace_section(text, start, stop, replacement):
    a = text.index(start)
    b = text.index(stop, a)
    return text[:a] + replacement + text[b:]


def main():
    archive_dir = ROUND2 / 'archive'
    archive_dir.mkdir(exist_ok=True)
    archive = archive_dir / 'submitted_reference_20261017.zip'
    names = sorted(p for p in SOURCE.iterdir() if p.is_file() and (
        p.suffix in ('.tex', '.bib', '.cls', '.bst', '.eps') or
        p.name in ('source_final_dasip_rev2_clean.pdf', 'fig1-eps-converted-to.pdf',
                   'CUTS_FOR_VALIDATION.md', 'readme.txt')))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in names}
    metadata = dict(authoritative_submitted_pdf='source_final_dasip_rev2_clean.pdf',
                    note='Associated local sources preserved as received; rebuild equivalence to submitted PDF is not assumed.',
                    file_hashes=hashes)
    if archive.exists():
        with zipfile.ZipFile(archive) as saved:
            assert json.loads(saved.read('ARCHIVE_MANIFEST.json')) == metadata, 'Original inputs differ from preserved archive'
    else:
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as saved:
            for p in names:
                saved.write(p, p.name)
            saved.writestr('ARCHIVE_MANIFEST.json', json.dumps(metadata, indent=2) + '\n')
    with zipfile.ZipFile(archive) as saved:
        assert saved.testzip() is None
        for name, digest in hashes.items():
            assert hashlib.sha256(saved.read(name)).hexdigest() == digest
    (archive_dir / 'SUBMITTED_REFERENCE_MANIFEST.json').write_text(json.dumps(metadata, indent=2) + '\n')
    target = ROUND2 / 'paper_round2_placeholder'
    target.mkdir(exist_ok=True)
    for name in ('llncs.cls', 'main.bib', 'splncs04.bst', 'fig1.eps', 'fig1-eps-converted-to.pdf'):
        shutil.copy2(SOURCE / name, target / name)
    text = (SOURCE / 'source_final_dasip_rev2.tex').read_text()
    text = replace_section(text, r'\begin{abstract}', r'\section{Introduction}', ABSTRACT + '\n\n')
    text = replace_section(text, r'\section{Introduction}', r'\section{Related Work}', INTRODUCTION)
    text = replace_section(text, r'\section{Shared Workloads', r'\section{Proposed Optimisations}', CHARACTERIZATION)
    generated_profile = target/'generated_component_profile_table.tex'
    if generated_profile.exists():
        anchor = text.index(r'\label{tab:profiling_regimes}')
        start = text.rfind(r'\begin{table}',0,anchor)
        end = text.index(r'\end{table}',anchor)+len(r'\end{table}')
        assert start>=0
        text=text[:start]+generated_profile.read_text()+text[end:]
    text = replace_section(text, r'\subsection{Depth-Aware Sample Reduction}', r'\section{Results}', METHOD)
    results_template = Path(__file__).with_name('paper_results_template.tex')
    results_text = results_template.read_text() if results_template.exists() and (target/'generated_verified_results.tex').exists() else RESULTS
    text = replace_section(text, r'\section{Results}', r'\section{Discussion and Conclusion}', results_text)
    text = replace_section(text, r'\section{Discussion and Conclusion}', '% Keep deferred result tables', DISCUSSION)
    text = text.replace('The mechanisms are complementary and mutually reinforcing:',
                        'Their joint benefit is workload-dependent:')
    if (target / 'generated_verified_results.tex').exists():
        text = text.replace('Its speed and quality conclusions remain \\pending{} in this draft.',
                            'Completed measurements include negative cases; the final pruning choice remains pending.')
        text = text.replace('The working PDF deliberately contains no replacement numerical speed claim.',
                            'The working PDF includes verified partial results, including regressions; it makes no unsupported positive speed claim.')
        scored_summary = ROUND2 / 'results/reconstruction_v1_analysis/SUMMARY.json'
        if scored_summary.exists():
            scored = json.loads(scored_summary.read_text())
            if scored['total'] > 0 and scored['scored'] == scored['total'] and not scored['failures']:
                text = text.replace('The working PDF includes verified partial results, including regressions; it makes no unsupported positive speed claim.',
                                    'The 20 October abstract summarises the scientific scope; embedded timing and composition remain pending.')
    additions = r'''
% Round 2 author draft. TBD cells are not measurements.
\usepackage{lmodern}
\setlength{\emergencystretch}{2em}
\newcommand{\pending}{\textcolor{red!70!black}{\textsf{TBD}}}
\AddToHook{shipout/foreground}{\begin{tikzpicture}[remember picture,overlay]
\node[anchor=north,font=\sffamily\scriptsize,text=red!70!black]
 at ([yshift=-7mm]current page.north)
 {ROUND 2 WORKING DRAFT --- PARTIAL RESULTS --- NOT FOR SUBMISSION};
\end{tikzpicture}}
'''
    if (ROUND2/'results/orin_maxn_analysis/ANALYSIS_STATUS.json').exists():
        text=text.replace('embedded timing and composition remain pending.',
                          'MAXN timings are included; 15 W and composition remain pending.')
    if (ROUND2/'results/orin_maxn_complete_analysis_v1/ANALYSIS_STATUS.json').exists():
        text=text.replace('MAXN timings are included; 15 W and composition remain pending.',
                          'MAXN benchmark and component-profiling results are included; 15 W remains pending.')
    if (ROUND2/'results/orin_15w_analysis_v1/ANALYSIS_STATUS.json').exists():
        text=text.replace('MAXN benchmark and component-profiling results are included; 15 W remains pending.',
                          'MAXN and 15 W benchmarks and component profiles are included; 15 W uses seed 0 only.')
    text = text.replace(r'\begin{document}', additions + '\n' + r'\begin{document}', 1)
    text = '\\def\\DASIPCleanVersion{1}\n' + text
    (target / 'paper_round2_placeholder.tex').write_text(text)
    notes = '''# Provisional Round 2 paper

Build: `latexmk -pdf -interaction=nonstopmode -halt-on-error paper_round2_placeholder.tex`

This is a separate, visibly marked author draft. It is not the final
submission, actual highlighted-difference PDF or completed rebuttal. TBD means
pending, not zero. No old pruning speed/quality figures are presented as
results of the conservative variant. No partial single-seed results are
promoted into the tables.

The submitted clean PDF and associated local source/bibliography/assets were
preserved in `../archive/submitted_reference_20261017.zip` with hashes. The
PDF is the authoritative comparison reference; rebuilding the current
associated sources is not assumed to reproduce it. Original files were not
edited or cleaned.

Changes: conceptual 20 October submission abstract without numerical results;
provisional introduction; conservative native-
lattice method replaces the old sampling equations/bucket policy; explicit
reference/precision/capacity/seed/timing protocol; empty profiling, A/B/C/D
kernel, complete-call, per-scene ATE, reconstruction and separate engineering
tables; qualified discussion. Existing related work and Morton mechanism/
illustrative figures are retained for the later author-directed writing pass.

Verified completed results are generated into `generated_verified_results.tex`
from audited CSVs and checked completed EC pose outputs. `RESULTS_PROGRESS.json`
and `RESULTS_PROGRESS.md` record coverage and input hashes. Missing conditions
remain TBD. The detached paper updater adds newly audited complete comparisons;
it does not invent Orin, reconstruction, engineering or composition results.
The working draft has no page limit and may exceed 12 pages. Prioritize complete
results and clear explanations during updates. Reconcile final submission length
after results and coauthor feedback; do not trim this transitional draft solely
to meet the final page budget.

The 20 October abstract is maintained in abstracts/ABSTRACT_2026-10-20.txt and inserted
verbatim into the PDF (with LaTeX percent escaping). It follows the original
scientific narrative, updates the depth-pruning method and validation datasets,
and omits numerical results and experiment logistics. Orin MAXN/15 W remains
the declared hardware study scope. This abstract can be revised before the
full-paper deadline.
Before 27 October: freeze tables/claims, complete the
rebuttal and generate an actual checked diff against the archived submission.
'''
    (target / 'README.md').write_text(notes)
    print(json.dumps(dict(archive=str(archive), archived_files=len(names),
                          submitted_pdf_sha256=hashes['source_final_dasip_rev2_clean.pdf'],
                          provisional_source=str(target / 'paper_round2_placeholder.tex')), indent=2))


if __name__ == '__main__':
    main()
