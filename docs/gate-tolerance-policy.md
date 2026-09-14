# Gate tolerance policy proposal

The layer 0 comparison shows a useful distinction. Native and CPU corrected
scores differ by at most `1.9073486328125e-6`, while all 60 top-6 selections
agree. The smallest sixth/seventh margin is `3.147125244140625e-5`, about 16.5
times the observed maximum score difference.

This supports a two-tier diagnostic policy, without changing runtime routing:

1. Keep selected expert IDs exact. Any ID difference is a discrete routing
   mismatch and remains unresolved, regardless of score tolerance.
2. For continuous score comparison, a provisional diagnostic bound of
   `max_abs <= 2e-6` may classify this layer/input as arithmetic-compatible.
   This is not a universal or M2 acceptance threshold.
3. Report the top-6/7 margin for every token. A score comparison is *stable*
   only when `margin > 4 * max_abs_observed` (or the pre-fixed score bound).
   Rows inside that band are `indeterminate` and require exact ID agreement on
   the tested input plus additional independent inputs; they are never silently
   accepted by the continuous bound.
4. Keep route weights separate. Their normalization and BF16 cast need their
   own comparison; matching IDs do not prove matching expert contribution.

For this 60-token run, every ID agrees and every margin exceeds `4 * 2e-6`, so
the gate is stable under this *diagnostic* policy. The result is evidence for a
reasonable MLX reference landing point, not proof of official CUDA behavior or
full-model correctness. Before promoting the bound, test more real prompts,
different layers, image-mask routing, ties, and score scales. If any discrete
decision changes, stop tolerance-based classification and investigate the
upstream boundary.

This policy deliberately does not relax index top-k tie rejection, local
reference→optimized bit equality, or production qualification gates. It only
describes how to label a cross-backend gate comparison while its oracle remains
CPU/oMLX rather than official CUDA.
