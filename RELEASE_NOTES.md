# Release Notes

## v0.2.0 Research Validation Readiness

This release focuses on turning the platform into a validation-ready research engineering system:

- CPU CI can run independently from CUDA/Triton/FlashAttention.
- GPU validation is isolated behind self-hosted CUDA runners.
- Benchmarks, model checkpoints, and validation reports have first-class locations.
- The README and docs distinguish locally verified artifacts from GPU-required checks.
