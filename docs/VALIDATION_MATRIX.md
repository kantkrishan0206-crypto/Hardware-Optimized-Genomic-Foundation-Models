# Validation Matrix

| Area | Local Status | Requires GPU/Data |
| --- | --- | --- |
| Tokenizer serialization | Unit tests | No |
| FASTA/FASTQ/VCF/BED/GTF readers | Unit tests | No |
| Model forward/backward | Unit test when torch is installed | No |
| Performer numerical stability | Unit test target | No |
| FlashAttention integration | Optional import contract | CUDA |
| Triton kernels | Optional import contract | CUDA |
| Promoter task | Synthetic reproducible task | No |
| ClinVar/ENCODE/GTEx | Pipeline hooks | External data |
| Distributed NCCL | Config and helper | Multi-GPU |
| Cloud deployment | Templates | Provider credentials |
