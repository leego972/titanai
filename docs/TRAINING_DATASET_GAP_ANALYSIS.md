# TitanAI Training and Dataset Gap Analysis

## Scope

This analysis covers the repository's current base pretraining, staged 1B upgrades, 1B-to-3B expansion, instruction fine-tuning, DPO, corpus generation, cyber-builder datasets, evaluation and runtime requirements. It does not introduce new behavioural policies; existing project controls remain unchanged.

## Current capability inventory

### Implemented

- Custom decoder-only Transformer in PyTorch.
- RMSNorm, SwiGLU, RoPE, grouped-query attention, FlashAttention/SDPA fallback and gradient checkpointing.
- 32K tokenizer support.
- Resumable pretraining checkpoints.
- 1B pretraining configuration.
- Canonical depth-scaled 3B configuration.
- 1B-to-3B checkpoint expansion.
- Prompt-masked SFT dataset and trainer.
- DPO dataset and trainer with frozen reference model.
- Sequential domain-upgrade curriculum.
- Large general, reasoning, technical, cyber, cinema and instruction corpora.
- Cyber-builder synthetic examples and specialist upgrade sets.
- Local-corpus packed-sequence pretraining.

## Fixed integration defects

- 1B SFT and DPO configs now contain complete model, tokenizer, data, training and evaluation sections.
- Dedicated 3B SFT and DPO configs now match the canonical 3B architecture.
- SFT and DPO launchers now accept output-directory overrides.
- SFT and DPO checkpoint loading is architecture-strict; width/depth mismatches fail before training.
- Deterministic train/validation splitting was added to SFT and DPO entry points.
- Duplicate 3B configs were unified around the 2048-wide, 66-layer depth-scaled architecture.
- The domain-upgrade runner now accepts arbitrary Titan model configs and can process 1B or 3B checkpoints.
- The 3B launcher now uses the canonical upscaler, canonical 3B post-training configs and the generalized upgrade runner.
- A preflight validator now checks config completeness, model construction and architecture consistency without allocating full model weights.

## Training gaps

### P0 — required before committing to a full 3B run

1. **End-to-end smoke execution**
   - Run preflight validation.
   - Expand a small 1B-shaped test checkpoint into a small 3B-shaped test model.
   - Run 2–5 pretraining steps.
   - Run one upgrade stage.
   - Run 2–5 SFT steps.
   - Run 2–5 DPO steps.
   - Confirm checkpoint handoff at every phase.

2. **Measured throughput and cost model**
   - Record tokens/second for the actual GPU type and GPU count.
   - Include evaluation, checkpointing and restart overhead.
   - Replace static time and cost estimates with measured projections.

3. **Runtime token accounting**
   - Derive steps from target tokens, sequence length, microbatch, accumulation and world size.
   - Write the resolved runtime values into each checkpoint manifest.

4. **Distributed-training verification**
   - Confirm that the pretraining process initializes DDP correctly under `torchrun`.
   - Confirm distributed samplers or independent packed streams do not duplicate the same token sequence across ranks.
   - Confirm checkpoint saving is rank-zero only.
   - Confirm resume restores every rank consistently.

5. **3B DPO memory strategy**
   - DPO holds policy and reference models simultaneously.
   - Validate whether the intended hardware can hold both 3B models plus optimizer states and activations.
   - Add reference-logprob precomputation or CPU/offloaded reference inference if required by measured memory use.

6. **Checkpoint manifest standardisation**
   Every checkpoint should include:
   - architecture version;
   - exact model config;
   - tokenizer checksum;
   - corpus manifest checksum;
   - data-mixture weights;
   - tokens consumed;
   - optimizer/scheduler state;
   - source checkpoint;
   - git commit;
   - world size and GPU type.

### P1 — capability training gaps

1. **Long-context training**
   - Current production configs use 2,048 tokens.
   - Repository analysis and system construction need at least 8K practical context, preferably 16K–32K with staged extension.
   - Add long-document packing, RoPE scaling experiments and long-context validation.

2. **Tool-call supervised training**
   Add a canonical schema for:
   - file search;
   - file read/write;
   - shell command execution;
   - compiler/build invocation;
   - test execution;
   - static analysis;
   - dependency analysis;
   - infrastructure validation;
   - browser and API testing.

   Training records must include tool request, actual tool result, interpretation, next action and final answer.

3. **Multi-step trajectory training**
   Existing prompt/answer examples are useful but insufficient for autonomous construction. Add trajectories containing:
   - requirement extraction;
   - architecture plan;
   - file plan;
   - implementation;
   - build output;
   - failed test;
   - diagnosis;
   - repair;
   - successful validation;
   - packaging and report.

4. **Repository-scale training**
   Add complete multi-file projects rather than isolated scripts. Include source, tests, dependencies, Docker/IaC, CI and expected outputs.

5. **Failure-recovery curriculum**
   Train on:
   - compiler failures;
   - dependency conflicts;
   - failing tests;
   - broken Docker images;
   - invalid Terraform/Kubernetes;
   - incorrect scanner conclusions;
   - partial patches;
   - regressions introduced by repairs.

6. **Training mixture control**
   - Store target token proportions per domain.
   - Report actual token proportions after filtering and packing.
   - Prevent very large generic corpora from suppressing cyber-builder and code-repair examples.

7. **Continual-training regression control**
   After each upgrade stage, evaluate retained capability in:
   - general language;
   - coding;
   - reasoning;
   - cyber knowledge;
   - system construction;
   - instruction following.

## Dataset gaps

### P0 — cyber systems builder datasets

1. **Complete security-system repositories**
   Required project families:
   - SIEM ingestion and correlation;
   - endpoint telemetry agents;
   - secrets scanning;
   - vulnerability management;
   - asset inventory;
   - secure API gateways;
   - certificate and PKI lifecycle;
   - identity and privileged-access management;
   - incident evidence collection;
   - network monitoring;
   - malware-analysis pipelines;
   - secure sandbox orchestration;
   - cloud security posture management;
   - software-supply-chain analysis.

2. **Build/test/repair trajectories**
   Each record should include executable artefacts and machine-verifiable outcomes, not only prose explanations.

3. **Repository audit and remediation pairs**
   Include:
   - complete repository snapshot;
   - confirmed findings;
   - evidence locations;
   - false-positive labels;
   - remediation patch;
   - regression tests;
   - post-fix scan results.

4. **Tool output interpretation**
   Add real or faithfully generated outputs from:
   - Semgrep;
   - CodeQL;
   - Trivy;
   - Syft and Grype;
   - Gitleaks and detect-secrets;
   - Nmap;
   - Zeek;
   - Suricata;
   - YARA;
   - Sigma/SIEM queries;
   - osquery;
   - Terraform validators;
   - Kubernetes validators;
   - language compilers and test frameworks.

5. **Architecture decision records**
   Structure each example as requirements, constraints, threat model, trust boundaries, selected components, rejected alternatives, data flows, deployment plan and validation plan.

### P1 — specialist dataset gaps

1. **Cloud and identity engineering**
   - AWS IAM, VPC, KMS, CloudTrail, Organizations and workload identity.
   - Azure Entra ID, Key Vault, Defender and network controls.
   - Google Cloud IAM, service accounts, audit logs and organisation policy.
   - Hybrid identity, federation, SSO, MFA, PAM and secrets rotation.

2. **Infrastructure as code**
   - Terraform, CloudFormation, Bicep, Kubernetes, Helm, Docker and CI/CD.
   - Misconfiguration-to-remediation pairs with validation output.

3. **Detection engineering**
   Link telemetry to behaviour, ATT&CK technique, detection logic, expected alert, false positives, tuning and response playbook.

4. **Vulnerability lifecycle graph**
   Link CVE, CWE, CAPEC, affected product/version, vulnerable commit, patch commit, prerequisite, indicator, mitigation, detection and regression test.

5. **Secure code transformation**
   Expand paired vulnerable/fixed examples across Python, TypeScript, JavaScript, Go, Rust, Java, C/C++, C#, PHP, Bash, PowerShell, SQL, Terraform, Docker and Kubernetes YAML.

6. **Protocol and network engineering**
   Include packet captures, protocol-state interpretation, proxy/load-balancer behaviour, TLS/PKI, DNS, routing, VPN, segmentation and service-mesh examples.

7. **Windows, Linux and mobile internals**
   Add system-level implementation and diagnostic records, event/log interpretation, service configuration, kernel/user boundaries, application sandboxing and endpoint instrumentation.

8. **Binary and malware analysis**
   Include static and dynamic artefacts, disassembly/decompilation context, behavioural traces, unpacking concepts, YARA creation and analyst reports.

9. **Performance and reliability engineering**
   Security systems must also survive load. Add profiling, concurrency, queueing, storage, backpressure, retry, high availability and failure-mode examples.

### P2 — quality and governance gaps

1. **Near-duplicate removal**
   Replace first-300-character MD5 deduplication with full-content hashes plus MinHash/SimHash and code-aware duplicate detection.

2. **Benchmark contamination detection**
   Hash and n-gram scan training data against every evaluation set. Hold out by source, time and repository.

3. **Source manifests**
   Record source revision, retrieval date, licence, checksum, transformation chain, language, document type, quality score and split assignment.

4. **Automated source health checks**
   Fail corpus builds when a required source returns zero records, falls below expected volume, produces malformed content or changes schema.

5. **Synthetic-data verification**
   Compile and test generated code where possible. Reject examples that fail syntax, dependency or unit-test checks.

6. **Negative examples**
   Add incorrect findings, scanner false positives, invalid patches, incomplete architecture and plausible but nonfunctional code.

## Evaluation gaps

### Required benchmark families

- Compilation and syntax success.
- Unit and integration test pass rate.
- Multi-file consistency.
- Dependency correctness.
- Container build success.
- Terraform/Kubernetes validation success.
- Vulnerability precision, recall and false-positive rate.
- Patch correctness and regression rate.
- Evidence-location accuracy.
- Architecture requirement coverage.
- Tool selection and argument validity.
- Recovery after failed execution.
- End-to-end project completion.
- Long-context retrieval and cross-file reasoning.
- Capability retention after each upgrade stage.

### Private evaluation sets

Maintain private, unseen:

- complete repository build tasks;
- repository audits;
- incident-analysis exercises;
- architecture briefs;
- tool-use trajectories;
- failure-and-repair tasks.

Public benchmark scores alone should not determine promotion between checkpoints.

## Recommended implementation sequence

1. Run the preflight validator and miniature end-to-end pipeline.
2. Benchmark throughput and memory on the intended hardware.
3. Finalise runtime token accounting and checkpoint manifests.
4. Build repository-scale cyber project datasets.
5. Add tool-call/result trajectories.
6. Add build/test/failure/repair trajectories.
7. Add machine-verifiable evaluation harnesses.
8. Extend context length in a measured staged run.
9. Train 1B, run all capability gates, then expand to 3B.
10. Run 3B continued pretraining, domain upgrades, SFT and DPO with checkpoint-by-checkpoint regression evaluation.
