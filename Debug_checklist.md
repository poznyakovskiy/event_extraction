# AI Model Debug Checklist

A practical checklist for debugging supervised ML / Transformer models.
Run this before long training runs or when metrics behave unexpectedly.

Goal: detect **data issues, tokenization errors, training bugs, and initialization mistakes** in under 15 minutes.

---

# 1. Data Sanity

### 1.1 Label Mapping

* Define `label2id` explicitly in code.
* Print it at startup.

Example:

```python
label2id = {
    "VAGUE": 0,
    "BEFORE": 1,
    "AFTER": 2,
    "EQUAL": 3
}
```

Checklist:

* [ ] Mapping is deterministic.
* [ ] Dataset labels match mapping.
* [ ] No implicit ordering from dataframe.

---

### 1.2 Label Distribution

Print counts for train/val/test.

```python
print(df["label"].value_counts())
```

Compute a **majority baseline**:

* accuracy
* macro F1

Checklist:

* [ ] Class imbalance understood
* [ ] Baseline metrics recorded

---

### 1.3 Sample Inspection

Print a few examples.

For 5–10 samples display:

* decoded input text
* special markers
* gold label

Checklist:

* [ ] Markers wrap the correct spans
* [ ] Text/label pair makes sense to a human

---

### 1.4 Duplicate Inputs

Ensure identical inputs don't have different labels.

Example check:

```python
duplicates = df.groupby("input_text")["label"].nunique()
print((duplicates > 1).sum())
```

Checklist:

* [ ] No duplicate inputs with conflicting labels

---

# 2. Tokenization and Markers

### 2.1 Special Tokens Are Registered

After adding special tokens:

```python
tokenizer.add_special_tokens(...)
model.resize_token_embeddings(len(tokenizer))
```

Verify:

```python
tokenizer.convert_tokens_to_ids("[T1]")
```

Checklist:

* [ ] Token ID ≠ UNK
* [ ] Tokenizer does not split markers

---

### 2.2 Marker Survival

Ensure truncation does not remove task-defining tokens.

Example check:

```python
assert "[T1]" in decoded_text
assert "[T2]" in decoded_text
```

Checklist:

* [ ] ≥ 99% of examples retain markers after tokenization

---

### 2.3 Attention Mask Correctness

Check mask consistency:

```python
mask.sum()
```

Checklist:

* [ ] Non-padding tokens have mask=1
* [ ] Padding tokens have mask=0

---

# 3. Model Initialization

### 3.1 Pretrained Weights

Ensure model is loaded with pretrained weights.

Correct:

```python
RobertaModel.from_pretrained(...)
```

Incorrect:

```python
RobertaModel(config=config)
```

Checklist:

* [ ] `.from_pretrained()` used
* [ ] Correct model name logged

---

### 3.2 Parameter Counts

Print parameter statistics.

```python
sum(p.numel() for p in model.parameters())
sum(p.numel() for p in model.parameters() if p.requires_grad)
```

Checklist:

* [ ] Trainable parameters match expectation

---

# 4. Training Step Sanity

### 4.1 Single Batch Test

Run a single batch training step and inspect:

* loss
* gradient norm
* parameter update magnitude

Example:

```python
loss.backward()
print(model.classifier.weight.grad.norm())
```

Checklist:

* [ ] Gradients non-zero
* [ ] Parameters change after optimizer step

---

### 4.2 Learning Rate

Print optimizer LR.

```python
optimizer.param_groups[0]["lr"]
```

Checklist:

* [ ] LR within expected range
* [ ] Scheduler not collapsing LR prematurely

---

### 4.3 Micro Overfit Test

Train on **32–128 samples** only.

Expected result:

* training accuracy > 95%
* loss approaches 0

Checklist:

* [ ] Model can memorize tiny dataset

If not, suspect:

* labels
* tokenization
* training loop
* initialization

---

# 5. Metrics Diagnostics

### 5.1 Train vs Validation Metrics

Log both:

* train loss
* train accuracy
* validation loss
* validation accuracy
* validation macro F1

Interpretation:

| Pattern                        | Likely Issue                 |
| ------------------------------ | ---------------------------- |
| Train not improving            | bug in pipeline              |
| Train improves but val doesn't | weak signal / generalization |

---

### 5.2 Confusion Matrix

Always inspect.

```python
from sklearn.metrics import confusion_matrix
```

Checklist:

* [ ] No single-class collapse
* [ ] Misclassification pattern understood

---

### 5.3 Prediction Distribution

Track predicted class frequencies.

Example:

```python
np.bincount(preds)
```

Checklist:

* [ ] Distribution not dominated by one class

---

# 6. Fast Failure Map

## Cannot overfit tiny dataset

Likely causes:

* wrong labels
* tokenization bug
* pretrained weights missing
* optimizer misconfigured
* markers truncated

---

## Model predicts single class

Check:

* label imbalance
* class weighting
* sampler
* marker visibility

---

## Training loss flat

Check:

* LR too small
* scheduler misconfigured
* gradients zero
* encoder frozen accidentally

---

# 7. Two Essential Habits

## Habit 1: Always Run Micro-Overfit Test

Do this **before any full training run**.

It isolates:

* pipeline bugs
* label issues
* optimizer issues

---

## Habit 2: Design Experiments That Fail Fast

Use quick ablations:

* markers vs no markers
* small context vs full context
* class weights vs none

Small experiments produce fast insights.

---

# Personal Rule

When training fails, do not guess.

Run the checklist.

Debugging ML systems is a process of **eliminating uncertainty step by step**.
