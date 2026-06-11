Brain Tumor Contrast Enhancement Generation Framework
Overview

This study proposes a tumor-aware contrast enhancement generation framework for brain MRI. Existing contrast enhancement synthesis models, including transformer-based approaches such as DCE-FORMER, generate contrast-enhanced MRI directly from non-contrast MRI using an end-to-end architecture. While these methods achieve visually plausible results, they often fail to explicitly focus on tumor regions where contrast enhancement is clinically most important.

To address this limitation, we introduce a two-stage framework that first localizes tumor regions through segmentation and then performs region-guided contrast enhancement generation. By incorporating explicit tumor information into the generation process, the proposed framework aims to improve the realism and clinical relevance of synthesized contrast-enhanced MRI.

Motivation

Contrast-enhanced MRI plays a crucial role in brain tumor diagnosis and treatment planning. However, gadolinium-based contrast agents increase examination costs and may pose risks to certain patients.

Recent studies have attempted to synthesize contrast-enhanced MRI from non-contrast MRI using deep learning models. Among them, DCE-FORMER utilizes transformer architectures and mutual information learning to model global image relationships.

Despite their success, existing end-to-end approaches have several limitations:

Limitations of Existing Methods
1. Lack of Explicit Tumor Awareness

Most methods process the entire MRI volume without explicit knowledge of tumor location.

As a result:

Tumor and normal tissues are treated equally.
Important enhancement patterns may be diluted.
Small lesions can be overlooked.
2. Inefficient Use of Model Capacity

The model attempts to reconstruct all anatomical structures simultaneously:

Skull
Cerebrospinal fluid
Healthy brain tissue
Tumor regions

This may reduce the capacity available for learning tumor-specific enhancement characteristics.

3. Limited Interpretability

End-to-end generation models provide little insight into:

Whether tumor localization was successful.
Which regions contributed most to enhancement synthesis.
4. Reduced Clinical Explainability

Clinicians often require evidence showing where enhancement originates.

Existing methods typically provide only attention maps, making clinical interpretation difficult.

Proposed Framework

The proposed framework consists of two sequential stages.

Stage 1: Tumor Segmentation

A segmentation network is trained to identify tumor regions from non-contrast MRI.

Output:

Tumor probability map
Binary tumor mask

This stage provides explicit localization information that is unavailable in conventional synthesis models.

Stage 2: Tumor-Guided Contrast Enhancement Generation

The predicted tumor mask is incorporated into a conditional image generation model.

Input:

Non-contrast MRI
Predicted tumor mask

Output:

Synthesized contrast-enhanced MRI

The generator focuses on clinically important tumor regions while preserving surrounding anatomical structures.
