# Advanced Loss Functions in Species Distribution Modeling

## 1. Loss Function Definitions

### 1.1 Tversky Loss
The Tversky loss function is designed to handle class imbalance by allowing explicit control over false positives and false negatives:

```math
L_{tversky} = 1 - \frac{TP}{TP + \alpha \cdot FP + \beta \cdot FN}
```

```math
\text{Where:}
```
```math
TP = \text{True Positives}
```
```math
FP = \text{False Positives}
```
```math
FN = \text{False Negatives}
```
```math
\alpha = \text{Penalty for false positives (set to 0.3)}
```
```math
\beta = \text{Penalty for false negatives (set to 0.7)}
```

### 1.2 Focal Loss
Focal loss addresses class imbalance by down-weighting easy examples and focusing on hard examples:

```math
L_{focal} = -\alpha_t \cdot (1-p_t)^\gamma \cdot \log(p_t)
```

```math
\text{Where:}
```
```math
p_t = \text{Model's predicted probability}
```
```math
\alpha_t = \text{Balancing factor (set to 0.25)}
```
```math
\gamma = \text{Focusing parameter (set to 2.0)}
```

### 1.3 Dice Loss
Dice loss optimizes the overlap between predicted and actual distributions:

```math
L_{dice} = 1 - \frac{2 \cdot TP + smooth}{2 \cdot TP + FP + FN + smooth}
```

```math
\text{Where:}
```
```math
smooth = \text{Smoothing factor (set to 1.0)}
```

## 2. Implementation Results

### 2.1 Performance Comparison (Malabar Erythrina)

| Model Type | Loss Function | Accuracy | TPR | TNR | F1-Score |
|------------|---------------|----------|-----|-----|----------|
| RF         | Original      | 0.9123   | 0.9643 | 0.8621 | 0.91     |
| RF         | Tversky       | 0.8772   | 1.0000 | 0.7586 | 0.89     |
| RF         | Focal         | 0.9298   | 0.8929 | 0.9655 | 0.93     |
| RF         | Dice          | 0.9474   | 1.0000 | 0.8966 | 0.95     |
| WLR        | Original      | 0.7500   | 0.8214 | 0.6786 | 0.75     |
| WLR        | Tversky       | 0.7857   | 0.8214 | 0.7500 | 0.79     |
| WLR        | Focal         | 0.5000   | 0.0000 | 1.0000 | 0.00     |
| WLR        | Dice          | 0.7857   | 0.8214 | 0.7500 | 0.79     |

### 2.2 Key Findings (Malabar Erythrina)

1. **Tversky Loss Performance**
   - Perfect TPR (1.0000 with RF)
   - Good TNR (0.7586 with RF)
   - High accuracy (0.8772 with RF)
   - Most effective for handling class imbalance

2. **Focal Loss Performance**
   - Excellent TPR (0.8929 with RF)
   - Best TNR (0.9655 with RF)
   - High accuracy (0.9298 with RF)
   - Better for handling hard examples

3. **Dice Loss Performance**
   - Perfect TPR (1.0000 with RF)
   - Excellent TNR (0.8966 with RF)
   - Best accuracy (0.9474 with RF)
   - Best overall performance

### 2.3 Model Type Comparison

1. **Random Forest (RF)**
   - Consistently high TPR and TNR
   - Excellent performance across all loss functions
   - More stable with different loss functions
   - Best overall performance with Dice loss (0.9474 accuracy, 1.0000 TPR)

2. **Weighted Logistic Regression (WLR)**
   - Improved performance with fixed NaN handling
   - Consistent TPR (0.8214) across most loss functions
   - Balanced TNR (0.7500) with Tversky and Dice loss
   - Better interpretability than RF
   - Issues with Focal loss (0.0000 TPR)

## References

1. Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection. ICCV.
2. Milletari, F., et al. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. 3DV.
3. Salehi, S. S. M., et al. (2017). Tversky Loss Function for Image Segmentation. arXiv:1706.05721. 