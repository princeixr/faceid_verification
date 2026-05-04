# FaceID Verification System Card

Final system version: Milestone 4 release candidate  
Primary config: `configs/default.yaml`  
Inference entrypoint: `scripts/infer_pair.py`  
Recommended final tag: `v1.0-final`

## System Overview

This project implements a pair-level face verification system. Given two face-image paths, the system preprocesses both images, generates face embeddings with `InceptionResnetV1` pretrained on VGGFace2 through `facenet-pytorch`, computes cosine similarity, applies an operating threshold, and returns a binary same-identity decision with a deterministic confidence score.

The runtime path used for the final release is:

1. Load and resize each RGB image to the configured FaceNet input size.
2. Normalize pixel values and run the embedding backend.
3. Compute cosine similarity between the two embeddings.
4. Apply the selected threshold.
5. Report the decision, confidence, total latency, and per-stage latency.

The CLI supports both single-pair inference and CSV batch inference.

## Intended Use

This system is intended for educational face-verification experiments where the user wants to compare two known face images and inspect a reproducible similarity-based decision. Appropriate uses include course project evaluation, controlled local experiments on LFW-style face crops, pipeline testing, and demonstrating thresholded embedding inference with reproducible artifacts.

The system is not intended for identity proofing, access control, surveillance, law-enforcement use, employment screening, credit decisions, medical decisions, or any other high-stakes automated decision. It should not be used as the only source of truth for a real person's identity.

## Data Summary

The project pipeline uses the Labeled Faces in the Wild style workflow. Pairs are generated deterministically into train, validation, and test splits, with same-identity positive pairs and different-identity negative pairs. The final operating threshold is selected on the validation split and then evaluated on the held-out test split.

Important data limitations:

* LFW is a celebrity/news-photo style dataset and does not represent all real-world capture conditions.
* The repo does not include reliable demographic metadata for audited subgroup claims.
* The pair-generation process controls labels and split reproducibility, but the source images still contain natural variation in pose, lighting, age, camera quality, and occlusion.
* Reported metrics should be interpreted as project-dataset results, not as deployment guarantees.

## Operating Threshold And Metrics

Final threshold:

* Threshold: `0.40`
* Selection split: validation
* Selection rule: `max_f1`
* Source run: `run_20260419T004952Z_a7f2c22b`
* Persisted runtime artifact: `outputs/inference/selected_threshold.json`

Validation metrics at threshold `0.40`:

| Metric | Value |
|---|---:|
| Accuracy | 0.9780 |
| Balanced accuracy | 0.9780 |
| Precision | 0.9825 |
| Recall | 0.9733 |
| F1 score | 0.9779 |
| Confusion matrix | TP=2920, FP=52, TN=2948, FN=80 |

Held-out test metrics at threshold `0.40`:

| Metric | Value |
|---|---:|
| Accuracy | 0.9770 |
| Balanced accuracy | 0.9770 |
| Precision | 0.9773 |
| Recall | 0.9767 |
| F1 score | 0.9770 |
| Confusion matrix | TP=2930, FP=68, TN=2932, FN=70 |

The confidence value is a deterministic logistic transform of the signed margin between the similarity score and threshold. With the default setting where higher cosine similarity means a same-identity match, values above `0.5` indicate support for a same-identity decision and values below `0.5` indicate support for a different-identity decision. It is a threshold-margin score, not a calibrated probability that the two images contain the same person.

## Failure Modes And Limitations

Known reliability risks include:

* Images with no clear frontal face, extreme pose, strong blur, poor lighting, heavy compression, or severe occlusion.
* Very small faces or images that require face detection/alignment before verification. This project assumes paths point to usable face images and does not provide a production face-detection gate.
* Visually similar different-identity pairs, twins, lookalikes, or same-identity pairs with large age, styling, expression, pose, or capture-condition changes.
* Threshold sensitivity. Scores near `0.40` should be treated as uncertain even if the binary decision is returned.
* Dataset shift. Performance can degrade when inputs differ from the project data distribution.
* CPU latency variability from hardware, thermal state, dependency versions, and model-cache warm-up.

## Fairness And Misuse Risks

The project does not include reliable demographic labels, so it does not make measured claims about demographic parity or subgroup-specific accuracy. This is a major limitation for any real deployment assessment.

Fairness-related risks still matter. Face verification systems can perform unevenly when image quality, pose, lighting, age, occlusion, camera type, or representation in the training/evaluation data differ across populations. Misuse is also possible if a verification score is treated as a definitive identity judgment or used in high-stakes contexts without consent, appeal, human review, and subgroup validation.

Responsible use requires additional evaluation on the intended population and capture process, explicit consent and privacy controls, monitoring for false accepts and false rejects, and a clear path for human review.

## Operational Constraints

* Runtime target: local CLI or Dockerized CLI, not a hardened production service.
* Default device: CPU, as configured in `configs/default.yaml`.
* Input format: paths to RGB-readable image files, typically JPEG images under the project data directory.
* Model dependency: `torch`, `torchvision`, and `facenet-pytorch`.
* The first model-backed run may be slower if pretrained weights need to be loaded or downloaded.
* The Docker image excludes `data/` and `outputs/`, so the repo directory must be mounted when running inference against local artifacts.
* Generated datasets and outputs are ignored by git; reproducibility depends on rerunning the documented commands.

## Reproducibility Pointer

Use [README.md](../README.md) as the release entry point and [Reproducibility_Checklist_Milestone4.md](Reproducibility_Checklist_Milestone4.md) for exact commands. The final tag should be created after the README, System Card, profiling report, and checklist are committed and verified:

```bash
git tag v1.0-final
git push origin v1.0-final
```
