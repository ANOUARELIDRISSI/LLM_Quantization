# 📊 Rapport sur les Techniques de Quantization des LLMs

## 🎯 Introduction

Ce rapport présente les différentes techniques de quantization utilisées pour optimiser les Large Language Models (LLMs). La quantization est une méthode de compression qui réduit la précision des poids du modèle pour diminuer l'utilisation de la mémoire et accélérer l'inférence.

---

## 📐 Qu'est-ce que la Quantization?

La quantization convertit les poids d'un modèle d'une représentation à haute précision (comme FP32 ou FP16) vers une représentation à plus faible précision (comme INT8 ou INT4).

### Formule de Quantization de Base

$$Q(x) = \text{round}\left(\frac{x - x_{min}}{x_{max} - x_{min}} \times (2^n - 1)\right)$$

Où:
- $x$ = valeur originale
- $n$ = nombre de bits cible
- $Q(x)$ = valeur quantifiée

---

## 🔢 Types de Précision

| Type | Bits | Plage | Taille par paramètre |
|------|------|-------|---------------------|
| **FP32** | 32 | ±3.4 × 10³⁸ | 4 bytes |
| **FP16** | 16 | ±65,504 | 2 bytes |
| **BF16** | 16 | ±3.4 × 10³⁸ | 2 bytes |
| **INT8** | 8 | -128 à 127 | 1 byte |
| **INT4** | 4 | -8 à 7 | 0.5 byte |
| **INT2** | 2 | -2 à 1 | 0.25 byte |

---

## ⚡ Technique 1: INT8 Quantization (8-bit)

### 📖 Explication

La quantization INT8 convertit les poids de 32 bits (FP32) ou 16 bits (FP16) vers 8 bits entiers. Cela réduit la taille du modèle de **4x** (FP32→INT8) ou **2x** (FP16→INT8).

### 🔬 Comment ça fonctionne

1. **Analyse des poids**: Déterminer les valeurs min/max des poids
2. **Calcul du scale factor**: 
   $$\text{scale} = \frac{x_{max} - x_{min}}{255}$$
3. **Quantification**: Convertir chaque poids:
   $$x_{int8} = \text{round}\left(\frac{x - x_{min}}{\text{scale}}\right)$$
4. **Zero-point**: Calculer le point zéro pour la dé-quantification

### 💻 Types de Quantization INT8

#### A) Quantization Dynamique (Dynamic Quantization)
```python
# PyTorch Native
model_int8 = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Couches à quantifier
    dtype=torch.qint8
)
```
- Les poids sont quantifiés **à l'avance**
- Les activations sont quantifiées **dynamiquement** pendant l'inférence
- ✅ Simple à implémenter
- ✅ Pas besoin de données de calibration
- ⚠️ Overhead de calcul pour les activations

#### B) Quantization Statique (Static Quantization)
```python
# Nécessite une calibration
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
# Passer des données de calibration
calibrate(model, calibration_data)
torch.quantization.convert(model, inplace=True)
```
- Poids ET activations quantifiés **à l'avance**
- Nécessite un dataset de calibration
- ✅ Plus rapide à l'inférence
- ⚠️ Plus complexe à configurer

### 📊 Résultats Typiques INT8

| Métrique | FP32 | INT8 | Amélioration |
|----------|------|------|--------------|
| Taille modèle | 6 GB | 1.5 GB | **4x** plus petit |
| RAM requise | 8 GB | 3 GB | **2.6x** moins |
| Tokens/sec | 5 | 8-12 | **1.5-2x** plus rapide |
| Qualité | 100% | ~99% | Perte minimale |

---

## 🚀 Technique 2: INT4 Quantization (4-bit)

### 📖 Explication

La quantization INT4 pousse la compression encore plus loin en utilisant seulement 4 bits par poids. Cela permet une réduction de **8x** par rapport à FP32.

### 🔬 Comment ça fonctionne

1. **Groupage des poids**: Les poids sont divisés en groupes (typiquement 32-128 poids)
2. **Scale par groupe**: Chaque groupe a son propre facteur d'échelle
3. **Quantification**: 
   $$x_{int4} = \text{round}\left(\frac{x}{\text{scale}}\right) + 8$$
   
   Où les valeurs sont mappées à la plage [0, 15] ou [-8, 7]

### 💻 Implémentation avec BitsAndBytes

```python
from transformers import BitsAndBytesConfig

# Configuration INT4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat4
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True  # Double quantization
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
```

### 🎯 Types de Quantization 4-bit

#### A) NF4 (NormalFloat4)
- Optimisé pour les poids qui suivent une distribution normale
- Meilleure préservation de la qualité
- Utilisé par QLoRA

#### B) FP4 (Float4)
- Représentation flottante en 4 bits
- Plus flexible mais moins précis

### 📊 Résultats Typiques INT4

| Métrique | FP32 | INT4 | Amélioration |
|----------|------|------|--------------|
| Taille modèle | 6 GB | 0.75 GB | **8x** plus petit |
| RAM requise | 8 GB | 2 GB | **4x** moins |
| Tokens/sec | 5 | 6-10 | Variable |
| Qualité | 100% | ~95-98% | Légère perte |

---

## 🧮 Technique 3: GPTQ (Gradient-based Post-Training Quantization)

### 📖 Explication

GPTQ est une méthode de quantization post-entraînement qui minimise l'erreur de reconstruction en utilisant des informations de gradient.

### 🔬 Algorithme

1. **Calcul de la matrice Hessienne**: Approximer les courbures des poids
2. **Quantification séquentielle**: Quantifier les colonnes une par une
3. **Compensation d'erreur**: Ajuster les poids non-quantifiés pour compenser

### 💻 Code

```python
from transformers import GPTQConfig

gptq_config = GPTQConfig(
    bits=4,
    dataset="c4",
    tokenizer=tokenizer,
    group_size=128
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=gptq_config
)
```

### ✅ Avantages
- Haute qualité de quantization
- Supporte 2, 3, 4, 8 bits
- Fonctionne bien sur GPU

### ⚠️ Inconvénients
- Nécessite GPU pour la quantization
- Processus plus lent

---

## 🎯 Technique 4: AWQ (Activation-aware Weight Quantization)

### 📖 Explication

AWQ identifie les poids "saillants" qui ont le plus d'impact sur les activations et les préserve avec plus de précision.

### 🔬 Principe

1. **Analyse des activations**: Identifier quels poids affectent le plus les sorties
2. **Mise à l'échelle**: Appliquer des scales différents selon l'importance
3. **Quantification adaptative**: Plus de précision pour les poids critiques

### 💻 Code

```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_quantized(
    "model_name",
    fuse_layers=True,
    trust_remote_code=False
)
```

---

## 📈 Technique 5: QLoRA (Quantized Low-Rank Adaptation)

### 📖 Explication

QLoRA combine la quantization 4-bit avec le fine-tuning efficace LoRA.

### 🔬 Comment ça fonctionne

1. **Modèle gelé en 4-bit**: Le modèle de base est quantifié en NF4
2. **Adaptateurs LoRA**: Petites matrices entraînables en FP16
3. **Double quantization**: Les constantes de quantization sont aussi quantifiées

### 💻 Code

```python
from peft import LoraConfig, get_peft_model

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)

model = get_peft_model(model, lora_config)
```

---

## 🔄 Technique 6: GGUF/GGML Quantization

### 📖 Explication

Format de quantization optimisé pour l'inférence CPU, utilisé par llama.cpp.

### 🎯 Types de Quantization GGUF

| Type | Description | Taille | Qualité |
|------|-------------|--------|---------|
| Q2_K | 2-bit, super compact | ~0.3 GB/B | ⭐⭐ |
| Q3_K_S | 3-bit, small | ~0.4 GB/B | ⭐⭐⭐ |
| Q4_K_M | 4-bit, medium | ~0.5 GB/B | ⭐⭐⭐⭐ |
| Q5_K_M | 5-bit, medium | ~0.6 GB/B | ⭐⭐⭐⭐⭐ |
| Q6_K | 6-bit | ~0.7 GB/B | ⭐⭐⭐⭐⭐ |
| Q8_0 | 8-bit | ~1 GB/B | ⭐⭐⭐⭐⭐ |

### 💻 Utilisation

```bash
# Conversion avec llama.cpp
./quantize model.gguf model-q4_k_m.gguf Q4_K_M
```

---

## 📊 Comparaison des Techniques

| Technique | Bits | Compression | Qualité | GPU Requis | Complexité |
|-----------|------|-------------|---------|------------|------------|
| **INT8 Dynamic** | 8 | 4x | 99% | Non | ⭐ Facile |
| **INT8 Static** | 8 | 4x | 99% | Non | ⭐⭐ Moyen |
| **INT4 BnB** | 4 | 8x | 95-98% | Oui | ⭐⭐ Moyen |
| **GPTQ** | 2-8 | 4-16x | 96-99% | Oui | ⭐⭐⭐ Difficile |
| **AWQ** | 4 | 8x | 97-99% | Oui | ⭐⭐⭐ Difficile |
| **QLoRA** | 4 | 8x | 98%+ | Oui | ⭐⭐⭐ Difficile |
| **GGUF** | 2-8 | 4-16x | 90-99% | Non | ⭐⭐ Moyen |

---

## 🍓 Compatibilité Raspberry Pi

### Modèles Recommandés par RAM

| RAM | Quantization | Taille Max Modèle | Exemple |
|-----|--------------|-------------------|---------|
| 4 GB | INT4/INT8 | ~1.5B params | Qwen2-1.5B-INT4 |
| 8 GB | INT4/INT8 | ~3B params | Llama-3.2-3B-INT4 |
| 16 GB | INT8 | ~7B params | Mistral-7B-INT8 |

### Performance Attendue

| Device | INT8 Tokens/sec | INT4 Tokens/sec |
|--------|----------------|-----------------|
| Raspberry Pi 5 (8GB) | 1-3 | 2-5 |
| Raspberry Pi 4 (8GB) | 0.5-1.5 | 1-3 |
| Desktop (16GB RAM) | 5-15 | 8-20 |

---

## 🛠️ Notre Implémentation

### Projet: Quantization de Qwen2-1.5B-Instruct

Dans ce notebook, nous avons implémenté:

1. **Baseline FP16**: Modèle original en half-precision
2. **INT8 Dynamic**: Quantization PyTorch native
3. **INT4**: Quantization 4-bit pour compression maximale

### Métriques Mesurées

- **Taille du modèle** (MB)
- **Ratio de compression**
- **Latence d'inférence** (secondes)
- **Tokens par seconde**
- **Similarité Bigram** (qualité des réponses)

---

## 📝 Conclusion

La quantization est essentielle pour déployer des LLMs sur des appareils à ressources limitées:

- **INT8** offre un bon équilibre entre compression (4x) et qualité
- **INT4** maximise la compression (8x) avec une légère perte de qualité
- **GPTQ/AWQ** offrent la meilleure qualité pour INT4
- **GGUF** est idéal pour l'inférence CPU sur edge devices

### Recommandations

| Cas d'usage | Technique Recommandée |
|-------------|----------------------|
| Production rapide | INT8 Dynamic |
| Edge device (Pi) | INT4 GGUF |
| Fine-tuning efficace | QLoRA |
| Qualité maximale INT4 | GPTQ ou AWQ |

---

---

## 🔧 Technique 7: Static INT8 Quantization

### 📖 Explication

La quantization statique pré-calcule les scales pour les activations en utilisant des données de calibration.

### 🔬 Comment ça fonctionne

1. **Préparation**: Insérer des observateurs (FakeQuantize) dans le modèle
2. **Calibration**: Passer des données représentatives
3. **Collecte**: Les observateurs enregistrent min/max des activations
4. **Conversion**: Convertir avec les statistiques collectées

### 💻 Code

```python
# Préparer le modèle
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# Calibration avec données
for data in calibration_data:
    model(data)

# Convertir
torch.quantization.convert(model, inplace=True)
```

### ✅ Avantages vs Dynamic
- Plus rapide à l'inférence (pas de calcul de scales)
- Mieux optimisé pour le matériel

---

## 🔄 Technique 8: Per-Channel Quantization

### 📖 Explication

Utilise un facteur d'échelle différent pour chaque canal de sortie.

### 🔬 Formule

$$W_{quant}[c] = \text{round}\left(\frac{W[c]}{\text{scale}[c]}\right)$$

Où $c$ est l'indice du canal.

### ✅ Avantages
- Meilleure précision que per-tensor
- Gère mieux les variations de magnitude entre canaux

---

## ✂️ Technique 9: Pruning + Quantization

### 📖 Explication

Combine deux techniques de compression :
1. **Pruning**: Mettre à zéro les petits poids (sparsité)
2. **Quantization**: Quantifier les poids restants

### 💻 Code

```python
import torch.nn.utils.prune as prune

# Appliquer pruning (50% des poids)
prune.l1_unstructured(module, name='weight', amount=0.5)

# Puis quantization
model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

### 📊 Résultats
- Compression: **5-8x**
- Sparsité: 50-90%
- Qualité: ~95%

---

## 🎓 Technique 10: Quantization-Aware Training (QAT)

### 📖 Explication

Simule la quantization pendant l'entraînement pour que le modèle apprenne des poids robustes.

### 🔬 Comment ça fonctionne

```
Forward: x → FakeQuantize → Linear → FakeQuantize → ...
Backward: Gradients passent comme si pas de quantization (STE)
```

**Straight-Through Estimator (STE)**: Pendant la backprop, on ignore l'opération de round().

### 💻 Code

```python
class FakeQuantize(nn.Module):
    def forward(self, x):
        if self.training:
            x_q = torch.round(x / scale) * scale  # Fake quantize
            # STE: pretend no rounding for gradients
            return x + (x_q - x).detach()
        return x
```

---

## 📦 Technique 11: ONNX Quantization

### 📖 Explication

Exporte le modèle au format ONNX puis applique la quantization avec ONNX Runtime.

### ✅ Avantages
- Portabilité (CPU, GPU, mobile, edge)
- Optimisations spécifiques au hardware
- Support INT8, INT4, mixte

### 💻 Code

```python
# Export vers ONNX
torch.onnx.export(model, dummy_input, "model.onnx")

# Quantization ONNX
from onnxruntime.quantization import quantize_dynamic
quantize_dynamic("model.onnx", "model_int8.onnx", weight_type=QuantType.QInt8)
```

---

## 📊 Tableau Récapitulatif de Toutes les Techniques

| # | Technique | Compression | Qualité | GPU Requis | Complexité |
|---|-----------|-------------|---------|------------|------------|
| 1 | FP16 | 2x | 99.9% | Non | ⭐ |
| 2 | BF16 | 2x | 99.5% | Non | ⭐ |
| 3 | Dynamic INT8 | 4x | 99% | Non | ⭐ |
| 4 | Static INT8 | 4x | 99% | Non | ⭐⭐ |
| 5 | Symmetric INT8 | 4x | 99% | Non | ⭐ |
| 6 | Asymmetric INT8 | 4x | 99.5% | Non | ⭐⭐ |
| 7 | AbsMax INT8 | 4x | 99% | Non | ⭐ |
| 8 | MinMax INT8 | 4x | 99% | Non | ⭐ |
| 9 | Block-wise INT8 | 4x | 99.5% | Non | ⭐⭐ |
| 10 | Per-Channel INT8 | 4x | 99.5% | Non | ⭐⭐ |
| 11 | Histogram INT8 | 4x | 99% | Non | ⭐⭐ |
| 12 | K-Means 4-bit | 8x | 95-98% | Non | ⭐⭐⭐ |
| 13 | K-Means 8-bit | 4x | 99% | Non | ⭐⭐⭐ |
| 14 | Mixed Precision | 2-4x | 99% | Non | ⭐⭐ |
| 15 | QAT | 4x | 99%+ | Non | ⭐⭐⭐ |
| 16 | Pruning + INT8 | 5-8x | 95% | Non | ⭐⭐ |
| 17 | ONNX INT8 | 4x | 99% | Non | ⭐⭐ |
| 18 | INT4 BnB | 8x | 95-98% | Oui | ⭐⭐ |
| 19 | GPTQ | 4-16x | 96-99% | Oui | ⭐⭐⭐ |
| 20 | AWQ | 8x | 97-99% | Oui | ⭐⭐⭐ |
| 21 | QLoRA | 8x | 98%+ | Oui | ⭐⭐⭐ |
| 22 | GGUF | 4-16x | 90-99% | Non | ⭐⭐ |

---

## 🆕 Nouvelles Techniques Ajoutées

### BF16 (Brain Float16)
```
FP32:  1 sign | 8 exponent  | 23 mantissa = 32 bits
FP16:  1 sign | 5 exponent  | 10 mantissa = 16 bits  
BF16:  1 sign | 8 exponent  | 7 mantissa  = 16 bits
```
BF16 garde la même plage que FP32 mais avec moins de précision.

### Symmetric vs Asymmetric INT8
- **Symétrique**: `scale = max(|x|) / 127`, zero_point = 0
- **Asymétrique**: `scale = (max-min) / 255`, zero_point calculé

### Block-wise Quantization
Divise les poids en blocs avec des scales séparés:
```python
[w0...w63] → scale_0
[w64...w127] → scale_1
```

### K-Means Weight Clustering
Remplace les poids par des indices de clusters:
```python
weights → K-Means(K=16) → indices + 16 centroids
```

### Histogram/Percentile Clipping
Clip les outliers avant quantization:
```python
low = percentile(weights, 0.1%)
high = percentile(weights, 99.9%)
clipped = clip(weights, low, high)
```

---

## 🛠️ Modèles Sauvegardés dans ce Projet

| # | Modèle | Technique | Chemin |
|---|--------|-----------|--------|
| 1 | Qwen2-1.5B-Instruct | Dynamic INT8 | `Qwen2-1.5B-Instruct-INT8/` |
| 2 | Qwen2-1.5B-Instruct | FP16 | `Qwen2-1.5B-Instruct-FP16/` |
| 3 | Qwen2-1.5B-Instruct | BF16 | `Qwen2-1.5B-Instruct-BF16/` |
| 4 | Qwen2-1.5B-Instruct | Symmetric INT8 | `Qwen2-1.5B-Instruct-INT8-Symmetric/` |
| 5 | Qwen2-1.5B-Instruct | Asymmetric INT8 | `Qwen2-1.5B-Instruct-INT8-Asymmetric/` |
| 6 | Qwen2-1.5B-Instruct | Block-wise INT8 | `Qwen2-1.5B-Instruct-INT8-Blockwise-64/` |
| 7 | Qwen2-1.5B-Instruct | AbsMax INT8 | `Qwen2-1.5B-Instruct-INT8-AbsMax/` |
| 8 | Qwen2-1.5B-Instruct | MinMax INT8 | `Qwen2-1.5B-Instruct-INT8-MinMax/` |
| 9 | Qwen2-1.5B-Instruct | Histogram INT8 | `Qwen2-1.5B-Instruct-INT8-Histogram/` |
| 10 | Qwen2-1.5B-Instruct | Mixed Precision | `Qwen2-1.5B-Instruct-MixedPrecision/` |
| 11 | Qwen2-1.5B-Instruct | K-Means 4-bit | `Qwen2-1.5B-Instruct-KMeans-4bit/` |
| 12 | Qwen2-1.5B-Instruct | K-Means 8-bit | `Qwen2-1.5B-Instruct-KMeans-8bit/` |
| 13 | SimpleTransformer | Static INT8 | `static_int8_model/` |
| 14 | SimpleModel | ONNX INT8 | `onnx_quantized_model/` |
| 15 | PrunableModel | Pruning + INT8 | `pruned_quantized_model/` |
| 16 | SimpleModel | Per-Channel INT8 | `per_channel_quantized_model/` |
| 17 | QATModel | QAT + INT8 | `qat_quantized_model/` |

---

## 📚 Références

1. Dettmers et al. "LLM.int8(): 8-bit Matrix Multiplication" (2022)
2. Frantar et al. "GPTQ: Accurate Post-Training Quantization" (2023)
3. Lin et al. "AWQ: Activation-aware Weight Quantization" (2023)
4. Dettmers et al. "QLoRA: Efficient Finetuning" (2023)
5. Han et al. "Deep Compression: Pruning, Quantization, Huffman Coding" (2016)
6. Jacob et al. "Quantization and Training of Neural Networks" (2018)

---

*Rapport généré pour le projet de quantization LLM - Janvier 2026*
