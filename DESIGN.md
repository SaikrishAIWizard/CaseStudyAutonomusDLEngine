# Design notes

## Connection importance: first-order Taylor score

Let the current loss be \(L(w)\) for one connection with weight \(w\). Removing
the connection changes its value to zero, so the weight change is
\(\Delta w = -w\). A first-order Taylor expansion around the current weight is

\[
L(w + \Delta w) - L(w) \approx \frac{\partial L}{\partial w}\Delta w
= -w\frac{\partial L}{\partial w}.
\]

The pruning score is the absolute value
\(\left|w\frac{\partial L}{\partial w}\right|\). A small score estimates that
removing the connection will change the loss relatively little. The estimate
ignores second- and higher-order terms and interactions among simultaneously
removed weights, so it is a local ranking heuristic, not a guarantee of the
resulting loss.

`prune.criteria.taylor_saliency` computes these per-weight scores.
`TaylorPruner` compares active connections across all linear layers and removes
the lowest scores until the global sparsity target is reached.

## Gradient of a masked weight

The forward path uses the effective weight \(w_{\mathrm{eff}} = m w\), where
the binary mask \(m\) is either zero or one. By the chain rule,

\[
\frac{\partial L}{\partial w}
= \frac{\partial L}{\partial w_{\mathrm{eff}}}
  \frac{\partial (m w)}{\partial w}
= m\frac{\partial L}{\partial w_{\mathrm{eff}}}.
\]

Therefore, a masked connection (\(m=0\)) receives a zero gradient in the
autodiff engine; an active connection receives its ordinary effective-weight
gradient. This is the correct derivative of the actual forward computation:
changing a stored weight that is multiplied by zero cannot change the output or
loss. The pruner ranks active weights before it updates the mask, so their
gradients still provide the Taylor estimate needed to decide what to remove.

## Autodiff bottleneck and optimization

The engine is designed for clarity, not large-scale throughput. Each operation
creates Python `Tensor` objects, parent references, and backward callbacks;
backpropagation traverses that Python object graph and many operations create
temporary NumPy arrays. Python dispatch, graph allocation, and small
unfused NumPy kernels will dominate for larger models, while NumPy's dense
matrix multiplication itself uses optimized native libraries.

An incremental optimization would profile representative workloads first,
then fuse common operations (for example, affine layer plus activation) to
reduce temporary arrays and graph nodes. For substantially larger models,
preallocated buffers, a compact operation tape with vectorized backward
kernels, or an established compiled array backend could reduce Python overhead.
Any optimization should retain the finite-difference gradient tests as a
correctness check.

## Serving pruned models in a multi-tenant inference service

Prune and validate models offline. Export an immutable, versioned artifact
containing architecture metadata, weights, masks or compact sparse matrices,
and validation metrics. Convert masks to the serving representation during
model loading rather than for every request.

At request time, authenticate the tenant and resolve its permitted model
version; do not allow a tenant identifier to select another tenant's artifact.
Use per-tenant quotas and request limits, and batch only requests that share a
compatible model/version and execution configuration. Keep model state
immutable during inference, isolate mutable buffers by request or worker, and
bound memory and queue usage so one tenant cannot starve others.

Cache loaded artifacts by immutable version with explicit memory limits.
Measure end-to-end latency and throughput, not just arithmetic counts, because
sparse kernels can be slower than dense kernels for small or lightly pruned
models. Add health checks, per-tenant usage metrics, safe rollout/rollback,
and a dense or previous-version fallback when sparse execution is unsupported
or fails validation. The repository's CSR path is a correctness and
microbenchmark example; it is not a production serving stack.
