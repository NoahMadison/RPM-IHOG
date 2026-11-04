# RPM-IHOG

Research code and benchmark data for language-model experts whose weight matrices are represented as sums of Kronecker products. The experiments compare training experts directly in this form with ordinary low-rank and dense experts. Separate experiments fit compressed representations to pretrained weights.

## What the experiments showed

On a small WikiText-2 mixture-of-experts model, native Kronecker experts had **5.2% lower held-out perplexity than ordinary low-rank experts**. Both used 4,352 learned scalars per 256 × 256 expert matrix. Across three paired seeds at 6,000 updates, the mean test loss difference was −0.0531 nats/token (paired 95% interval −0.0869 to −0.0192). The dense AdamW baseline achieved better quality than either factorization. The native model's loss was 0.0928 nats/token higher than dense.

The seed-11 FP16 native archive occupied 29.1 MB. Running its factors directly used 27.74 MiB of steady GPU allocation versus 42.68 MiB for a dense reconstruction of the same weights. Direct execution was slower in all 16 tested timing conditions across the native and pretrained workloads. The study ran on one V100. Multi-GPU training speed, full large-model deployment, and convergence beyond this training budget remain open.

## Exploring the repository

- [Measured findings](verification/v100_20261030T135845Z_ff01902f/reports/006_20261030T191420Z_SUMMARY.md) summarizes the results, controls, and limitations.
- [Technical report](verification/v100_20261030T135845Z_ff01902f/reports/005_20261030T190704Z.md) gives the per-seed results and links to the underlying measurements. The [claim ledger](verification/v100_20261030T135845Z_ff01902f/reports/005_20261030T190704Z_CLAIMS.json) records the verdict for each hypothesis.
- [Verification notes](verification/v100_20261030T135845Z_ff01902f/README_VERIFICATION.md) describe the experiment layout, pinned environment, and execution assumptions. The runners were written for the original workspace and GPU configuration. The published results can be inspected without downloading weights or running a GPU job.

The top-level scripts, [whitepaper](technical_whitepaper.md), and [earlier benchmark output](EVIDENCE.md) document the exploratory work that motivated verification. For conclusions, use the measured findings above.

## License

Original project code and documentation are available under the [MIT License](LICENSE). External models, datasets, and dependencies retain their respective licenses.
