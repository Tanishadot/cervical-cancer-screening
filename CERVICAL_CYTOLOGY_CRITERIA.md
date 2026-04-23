# Cervical Cytology Classification Criteria (Bethesda-Aligned)

## 1. Core Cellular Features

### A. Nuclear Features (Primary but NOT sufficient alone)

* **Nuclear enlargement**
* **Irregular nuclear contours**
* **Hyperchromasia** (dense chromatin)
* **Increased N:C ratio**

⚠️ **Note:**
Nuclear features alone should NOT determine classification.
They must be interpreted along with cytoplasmic and background context.

---

### B. Cytoplasmic Features (CRITICAL for classification)

#### 1. Perinuclear Halo (Koilocytosis)

* Clear zone around nucleus
* Associated with HPV infection
* Key indicator of LSIL

#### 2. Cytoplasmic Texture Changes

* **Keratinization** (dyskeratosis)
* Dense or orangeophilic cytoplasm
* Cytoplasmic shrinkage or distortion

#### 3. Cytoplasmic Maturity

* **Superficial/intermediate** → normal (NILM)
* **Immature/metaplastic** → borderline (ASC-US)

---

### C. Background Features (Contextual Diagnosis)

#### 1. Dirty Background

* Debris, inflammatory cells, necrotic material
* Suggests infection, inflammation, or malignancy

#### 2. Clean Background

* Seen in normal (NILM)

#### 3. Tumor Diathesis

* Necrotic background with blood/debris
* Strong indicator of high-grade lesions or carcinoma

---

## 2. Specific Cytological Patterns

### A. Dyskeratosis

* Premature keratinization of individual cells
* Not specific alone
* If isolated → NILM
* If with atypia → ASC-US or LSIL

---

### B. Metaplasia

* Transformation of glandular → squamous cells
* Benign adaptive process
* Without atypia → NILM
* With atypia → ASC-US

---

### C. Koilocytosis (HPV Effect)

* Perinuclear clearing (halo)
* Enlarged nucleus
* Irregular nuclear contours

→ DEFINING FEATURE of LSIL

---

## 3. Bethesda Classification Mapping Logic

### NILM (Negative for Intraepithelial Lesion or Malignancy)

* Normal cells
* Metaplasia without atypia
* Dyskeratosis (mild, isolated)
* Clean background

---

### ASC-US (Atypical Squamous Cells of Undetermined Significance)

* Mild nuclear atypia
* Ambiguous cytoplasmic features
* Borderline cases

---

### LSIL (Low-grade Squamous Intraepithelial Lesion)

* Koilocytosis present
* Perinuclear halo
* Mild nuclear enlargement
* HPV-related changes

---

### HSIL (High-grade Squamous Intraepithelial Lesion)

* High N:C ratio
* Marked nuclear irregularity
* Hyperchromasia
* Reduced cytoplasm

---

### SCC (Squamous Cell Carcinoma)

* Severe nuclear atypia
* Tumor diathesis (dirty necrotic background)
* Highly abnormal cell clusters

---

## 4. Key Diagnostic Principles

1. **Classification is MULTI-FACTOR:**
   * Nuclear + Cytoplasmic + Background

2. **Context matters:**
   * Same feature → different class depending on combination

3. **Cytoplasm is NOT secondary:**
   * It is essential for LSIL detection

4. **Background is a diagnostic signal:**
   * Especially for HSIL and SCC

---

## 5. Model Design Implications

To align with clinical reasoning, model must:

* **Learn nuclear features** (chromatin, shape)
* **Learn cytoplasmic features** (halo, keratinization)
* **Learn background features** (debris, inflammation)

Model should NOT rely only on nucleus.

---

## 6. Explainability Requirement

Grad-CAM or attention maps should highlight:

* **Cytoplasmic halo** (LSIL)
* **Background debris** (HSIL/SCC)
* **Not only nucleus**

If attention is only on nucleus → model is incomplete.
