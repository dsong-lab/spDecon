Model and core implementation
=============================

Molecular reconstruction
------------------------

Let :math:`Y` be a spots-by-features matrix of raw nonnegative measurements,
:math:`s_i=\sum_gY_{ig}`, :math:`P` the spots-by-components weights, and :math:`B`
the components-by-features profiles. spDecon models

.. math::

   Y_{ig}\sim\operatorname{Poisson}\{s_i(PB)_{ig}\},\qquad
   P_{ik}\geq0,\quad B_{kg}\geq0,\quad
   \sum_kP_{ik}=\sum_gB_{kg}=1.

The molecular loss is generalized KL divergence divided by total observed signal.
For noninteger assay measurements it is used as a nonnegative reconstruction
loss.
These weights measure contributions to the fitted molecular signal. Conversion
to cell fractions requires reference yields and is performed explicitly in the
reference-based RNA tutorial.

Spatial structure
-----------------

A graph connects nearby spots within each physical section. The regularizer is

.. math::

   \operatorname{TV}_{\epsilon}(P)
   =\frac{1}{Z}\sum_{(i,j)\in E}w_{ij}
      \left(\sqrt{\|P_i-P_j\|_2^2+\epsilon^2}-\epsilon\right),
   \qquad \epsilon=0.02.

The reference-free RNA graph uses symmetric 4NN edges, a distance cutoff at 1.5
median nearest-neighbour spacings and Gaussian coordinate weights. Its coefficient
is calibrated from seed-1 consensus gradients, without truth. Histology adds
exposure weighting and an image-similarity gate while retaining the coordinate
normalizer. The supplied joint-sample and paired-assay examples use their recorded
within-section edges and coefficient 0.05.

Reference-free initialization
-----------------------------

Three count-only branches use all features, variable features and expression
clusters. Branches are fitted on the complete selected feature axis, aligned by
profile cosine similarity and averaged with equal weights. The unrefined consensus
initializes the joint molecular/spatial fit. Labels or evaluation truth do not
enter the initialization. The solver uses majorization-minimization, safeguarded
extrapolation and stationarity checks.

Fixed references
----------------

Fixed-reference RNA learns only weights. Arithmetic mean reference counts are
normalized into profiles; reference row sums provide fitted-panel yields for
cell-fraction conversion. The strict main RNA protocol is
``reference_spatial.fit(..., tolerance=1e-10, max_steps=100000)`` with spatial
weight zero. The separate independent STARmap analysis uses different settings.
The ATAC reference adapter retains its own three-initializer protocol.

Histology-guided fitting
------------------------

HIPT and UNI features form two equally weighted 16-bin frequency blocks. The
model reconstructs their 32-column concatenation using :math:`PH`, sharing
:math:`P` with RNA. Each encoder block of a morphology profile sums to 0.5.
Exposure-weighted morphology KL and image-gated TV are added to the molecular
objective. PCA and visual-word fitting use only image features; pathology is held
out until evaluation. See the G2 notebook for the full extraction procedure.

Joint samples with design effects
---------------------------------

Each profile group has an effective profile. A flat design uses a shared profile
plus one group deviation. For a two-level nested design,

.. math::

   B^{(r)}=B_0+\Delta^{(1)}_{a(r)}+\Delta^{(2)}_r.

Here :math:`a(r)` is the parent of group :math:`r`. The ``effects`` specification
names the levels and selects their metadata columns. For example, the root can
be donor and the child visit, or the root can be batch and the child sample.
The CRC example uses patient and within-patient condition.

The parent profile averages its observed child-group profiles equally, and
:math:`B_0` averages parent profiles equally. Root deviations sum to zero across
parents; child deviations sum to zero within each parent. A parent with only one
observed child has zero child deviation. With one grouping level, :math:`B_0`
is the mean of group profiles and there is only one deviation term.

Effective profiles and spot weights are learned jointly. A separate quadratic
penalty shrinks each level's deviations, with the within-parent penalty averaged
over that parent's child groups. Deviations can be signed; nonnegativity and the
unit simplex constraint apply to each effective profile. Sections sharing a
profile group keep their own spatial graphs. The implementation supports one
flat effect or two nested effects. See :doc:`api` for configuration and outputs.

The named interface uses the same objective and five-cycle auxiliary profile
update as the CRC patient/condition fit. The CRC notebook provides its complete
saved fit and an optional fitting call.

Paired multi-omics
------------------

Paired assays are balanced to equal total signal and concatenated. They share one
:math:`P`; all features contribute to reconstruction. For linked features,
relative expression is computed from fitted profile values divided by observed
feature abundance. A quadratic penalty shrinks these comparable profiles toward
their across-assay mean, allowing assay-specific deviations. The Tonsil example
uses 20 links and coefficient 10. With fixed reference profiles the deviation
penalty is disabled. Independent RCC analysis fits two separate models and uses
neither shared weights nor this penalty.
