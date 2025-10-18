"""Self-contained Kaggle pipeline for Ancient Texts Provenance Challenge."""

from __future__ import annotations

import argparse
import json
import math
import random
import string
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

try:
    from transformers import (
        AutoConfig,
        AutoModelForMaskedLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
except ImportError as exc:  # pragma: no cover - installation handled externally
    raise SystemExit(
        "transformers is required. Install via `pip install transformers accelerate datasets peft sentencepiece scikit-learn`"
    ) from exc

try:
    from datasets import Dataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "datasets library is required. Install via `pip install datasets`."
    ) from exc


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_model_name(model_name: str) -> str:
    """Sanitize model name for filesystem usage."""
    return model_name.replace("/", "-")


@dataclass
class Config:
    """Configuration container for the training pipeline."""

    train_csv: str
    test_csv: str
    sample_csv: str
    text_col: str = "text"
    label_col: str = "label"
    id_col: str = "id"
    models: str = "xlm-roberta-large"
    model_dir_override: str = ""
    cache_dir: str = "/kaggle/working/hf_cache"
    output_dir: str = "/kaggle/working"
    n_folds: int = 5
    seed: int = 42
    epochs: int = 4
    lr: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    train_batch_size: int = 16
    eval_batch_size: int = 32
    grad_accum: int = 2
    fp16: bool = True
    max_len: int = 512
    chunk_len: int = 384
    chunk_stride: int = 256
    chunk_agg: str = "mean"
    use_long_model: bool = False
    use_focal: bool = False
    focal_gamma: float = 1.5
    label_smoothing: float = 0.05
    alpha_class_weight: float = 0.5
    mlm_enable: bool = True
    mlm_steps: int = 20000
    mlm_seq_len: int = 256
    mlm_lr: float = 2e-4
    mlm_warmup: int = 1000
    mlm_weight_decay: float = 0.01
    mlm_batch_tokens: int = 2048
    mlm_mask_prob: float = 0.15
    calibrate: bool = True
    stacking: bool = True
    prior_nudge_json: str = ""
    augment_prob: float = 0.0
    strip_digits: bool = False
    strip_punct: bool = False
    mlm_epochs: Optional[int] = None

    def model_list(self) -> List[str]:
        return [m.strip() for m in self.models.split(",") if m.strip()]

    @classmethod
    def from_args(cls, args: Optional[Sequence[str]] = None) -> "Config":
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--train_csv", required=True)
        parser.add_argument("--test_csv", required=True)
        parser.add_argument("--sample_csv", required=True)
        parser.add_argument("--text_col", default="text")
        parser.add_argument("--label_col", default="label")
        parser.add_argument("--id_col", default="id")
        parser.add_argument("--models", default="xlm-roberta-large")
        parser.add_argument("--model_dir_override", default="")
        parser.add_argument("--cache_dir", default="/kaggle/working/hf_cache")
        parser.add_argument("--output_dir", default="/kaggle/working")
        parser.add_argument("--n_folds", type=int, default=5)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--epochs", type=int, default=4)
        parser.add_argument("--lr", type=float, default=2e-5)
        parser.add_argument("--warmup_ratio", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--train_batch_size", type=int, default=16)
        parser.add_argument("--eval_batch_size", type=int, default=32)
        parser.add_argument("--grad_accum", type=int, default=2)
        parser.add_argument("--fp16", type=lambda x: x.lower() == "true", default=True)
        parser.add_argument("--max_len", type=int, default=512)
        parser.add_argument("--chunk_len", type=int, default=384)
        parser.add_argument("--chunk_stride", type=int, default=256)
        parser.add_argument("--chunk_agg", default="mean")
        parser.add_argument("--use_long_model", type=lambda x: x.lower() == "true", default=False)
        parser.add_argument("--use_focal", type=lambda x: x.lower() == "true", default=False)
        parser.add_argument("--focal_gamma", type=float, default=1.5)
        parser.add_argument("--label_smoothing", type=float, default=0.05)
        parser.add_argument("--alpha_class_weight", type=float, default=0.5)
        parser.add_argument("--mlm_enable", type=lambda x: x.lower() == "true", default=True)
        parser.add_argument("--mlm_steps", type=int, default=20000)
        parser.add_argument("--mlm_seq_len", type=int, default=256)
        parser.add_argument("--mlm_lr", type=float, default=2e-4)
        parser.add_argument("--mlm_warmup", type=int, default=1000)
        parser.add_argument("--mlm_weight_decay", type=float, default=0.01)
        parser.add_argument("--mlm_batch_tokens", type=int, default=2048)
        parser.add_argument("--mlm_mask_prob", type=float, default=0.15)
        parser.add_argument("--calibrate", type=lambda x: x.lower() == "true", default=True)
        parser.add_argument("--stacking", type=lambda x: x.lower() == "true", default=True)
        parser.add_argument("--prior_nudge_json", default="")
        parser.add_argument("--augment_prob", type=float, default=0.0)
        parser.add_argument("--strip_digits", type=lambda x: x.lower() == "true", default=False)
        parser.add_argument("--strip_punct", type=lambda x: x.lower() == "true", default=False)
        parser.add_argument("--mlm_epochs", type=int, default=None)
        parsed = parser.parse_args(args=args)
        return cls(**vars(parsed))


def load_data(train_csv: str, test_csv: str, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and test data from CSV files."""
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    if cfg.text_col not in train_df.columns:
        raise KeyError(f"text column {cfg.text_col} missing in train")
    if cfg.text_col not in test_df.columns:
        raise KeyError(f"text column {cfg.text_col} missing in test")
    if cfg.label_col not in train_df.columns:
        raise KeyError(f"label column {cfg.label_col} missing in train")
    return train_df, test_df


def clean_text(text: str, cfg: Config) -> str:
    """Normalize and optionally augment text."""
    import unicodedata
    import re

    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    text = unicodedata.normalize("NFKC", text)
    if cfg.strip_digits:
        text = re.sub(r"\d+", " ", text)
    if cfg.strip_punct:
        translator = str.maketrans({ch: " " for ch in string.punctuation})
        text = text.translate(translator)
    text = re.sub(r"\s+", " ", text).strip()
    if cfg.augment_prob > 0 and random.random() < cfg.augment_prob:
        tokens = text.split()
        if tokens:
            span_len = max(1, int(len(tokens) * random.uniform(0.01, 0.03)))
            start = random.randint(0, max(0, len(tokens) - span_len))
            for idx in range(start, start + span_len):
                if random.random() < 0.5:
                    tokens[idx] = ""
            tokens = [tok for tok in tokens if tok]
            text = " ".join(tokens)
    return text


class LengthAwareDataset(torch.utils.data.Dataset):
    """Dataset storing tokenized inputs and labels, sampling random chunks."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        cfg: Config,
        label_encoder: Optional[LabelEncoder] = None,
        is_train: bool = True,
    ) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.texts = [clean_text(t, cfg) for t in df[cfg.text_col].tolist()]
        self.is_train = is_train
        if label_encoder is not None and cfg.label_col in df.columns:
            self.labels = label_encoder.transform(df[cfg.label_col].astype(str))
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.texts)

    def _select_chunk(self, text: str) -> str:
        cfg = self.cfg
        if len(text) <= cfg.max_len or self.cfg.use_long_model:
            return text
        tokens = text.split()
        max_tokens = cfg.chunk_len
        if len(tokens) <= max_tokens:
            return text
        if self.is_train:
            start = random.randint(0, max(0, len(tokens) - max_tokens))
        else:
            start = 0
        return " ".join(tokens[start : start + max_tokens])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self._select_chunk(self.texts[idx])
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.cfg.max_len,
            padding=False,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def prepare_corpus_for_mlm(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: Config) -> Dataset:
    """Prepare corpus dataset for MLM training."""
    texts = pd.Series(list(train_df[cfg.text_col]) + list(test_df[cfg.text_col]))
    texts = texts.dropna().drop_duplicates().sample(frac=1.0, random_state=cfg.seed)
    cleaned = [clean_text(t, cfg) for t in texts.tolist()]
    return Dataset.from_dict({"text": cleaned})


def run_mlm(model_name_or_path: str, tokenizer, dataset: Dataset, cfg: Config) -> str:
    """Run domain-adaptive MLM training and return adapted model path."""
    if not cfg.mlm_enable:
        return model_name_or_path

    output_dir = Path(cfg.output_dir) / f"{Path(model_name_or_path).name}-mlm"
    output_dir.mkdir(parents=True, exist_ok=True)

    def tokenize_mlm(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.mlm_seq_len,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize_mlm, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=cfg.mlm_mask_prob)

    total_examples = len(tokenized)
    train_steps = cfg.mlm_steps
    if cfg.mlm_epochs:
        steps_per_epoch = math.ceil(total_examples / cfg.train_batch_size)
        train_steps = steps_per_epoch * cfg.mlm_epochs

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=max(1, cfg.mlm_batch_tokens // max(1, cfg.mlm_seq_len)),
        learning_rate=cfg.mlm_lr,
        num_train_epochs=1,
        max_steps=train_steps,
        warmup_steps=cfg.mlm_warmup,
        weight_decay=cfg.mlm_weight_decay,
        fp16=cfg.fp16,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        dataloader_num_workers=2,
        seed=cfg.seed,
        report_to=["none"],
    )

    model = AutoModelForMaskedLM.from_pretrained(
        model_name_or_path,
        cache_dir=cfg.cache_dir,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized, data_collator=data_collator)
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))
    return str(output_dir)


def compute_class_weights(labels: np.ndarray, alpha: float) -> torch.Tensor:
    """Compute class weights based on label frequencies."""
    class_counts = np.bincount(labels)
    class_counts[class_counts == 0] = 1
    weights = 1.0 / (class_counts.astype(np.float32) ** alpha)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def focal_loss_fn(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Compute focal loss with optional class weights and label smoothing."""
    num_classes = logits.size(-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
    if label_smoothing > 0:
        targets_one_hot = targets_one_hot * (1 - label_smoothing) + label_smoothing / num_classes
    pt = (targets_one_hot * probs).sum(dim=-1)
    focal_factor = (1 - pt) ** gamma
    losses = -(focal_factor.unsqueeze(-1) * targets_one_hot * log_probs).sum(dim=-1)
    if weight is not None:
        losses = losses * weight[targets]
    return losses.mean()


class CustomTrainer(Trainer):
    """Trainer subclass to support focal loss and label smoothing."""

    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False):  # type: ignore[override]
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        if self.loss_fn is not None and labels is not None:
            loss = self.loss_fn(logits, labels)
        else:
            loss = outputs.loss if hasattr(outputs, "loss") else torch.nn.functional.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train_fold(
    model_path: str,
    tokenizer,
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_encoder: LabelEncoder,
    cfg: Config,
    class_weights: Optional[torch.Tensor],
) -> Tuple[Dict[str, Any], np.ndarray, "CustomTrainer"]:
    """Train a model on a single fold and return metrics and trainer."""

    num_labels = len(label_encoder.classes_)
    config = AutoConfig.from_pretrained(model_path, num_labels=num_labels, cache_dir=cfg.cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config, cache_dir=cfg.cache_dir)

    train_dataset = LengthAwareDataset(train_df, tokenizer, cfg, label_encoder=label_encoder, is_train=True)
    val_dataset = LengthAwareDataset(val_df, tokenizer, cfg, label_encoder=label_encoder, is_train=False)

    steps_per_epoch = math.ceil(len(train_dataset) / cfg.train_batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    args = TrainingArguments(
        output_dir=str(Path(cfg.output_dir) / f"{Path(model_name).name}-fold{fold}"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        warmup_steps=warmup_steps,
        weight_decay=cfg.weight_decay,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=cfg.fp16,
        seed=cfg.seed + fold,
        report_to=["none"],
    )

    def loss_wrapper(logits, labels):
        if cfg.use_focal:
            weight = class_weights.to(logits.device) if class_weights is not None else None
            return focal_loss_fn(logits, labels, gamma=cfg.focal_gamma, weight=weight, label_smoothing=cfg.label_smoothing)
        return torch.nn.functional.cross_entropy(
            logits,
            labels,
            weight=class_weights.to(logits.device) if class_weights is not None else None,
            label_smoothing=cfg.label_smoothing,
        )

    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, label_encoder),
        loss_fn=loss_wrapper,
    )

    trainer.train()
    trainer.save_model()
    metrics = trainer.evaluate()
    val_logits = trainer.predict(val_dataset).predictions
    return metrics, val_logits, trainer


def compute_metrics(eval_pred, label_encoder: LabelEncoder) -> Dict[str, float]:
    """Compute macro and micro F1 metrics."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = metrics.f1_score(labels, preds, average="macro")
    micro_f1 = metrics.f1_score(labels, preds, average="micro")
    return {"macro_f1": macro_f1, "micro_f1": micro_f1}


def predict_with_sliding_window(model, tokenizer, texts: Sequence[str], cfg: Config) -> np.ndarray:
    """Perform sliding-window inference for long texts."""
    model.eval()
    device = next(model.parameters()).device
    all_logits: List[np.ndarray] = []
    with torch.no_grad():
        for text in texts:
            cleaned = clean_text(text, cfg)
            tokens = cleaned.split()
            if cfg.use_long_model or len(tokens) <= cfg.chunk_len:
                encoded = tokenizer(
                    cleaned,
                    truncation=True,
                    max_length=cfg.max_len,
                    return_tensors="pt",
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                logits = model(**encoded).logits
                all_logits.append(logits.cpu().numpy()[0])
            else:
                chunk_logits = []
                for start in range(0, len(tokens), cfg.chunk_stride):
                    chunk = tokens[start : start + cfg.chunk_len]
                    if not chunk:
                        continue
                    encoded = tokenizer(
                        " ".join(chunk),
                        truncation=True,
                        max_length=cfg.max_len,
                        return_tensors="pt",
                    )
                    encoded = {k: v.to(device) for k, v in encoded.items()}
                    logits = model(**encoded).logits
                    chunk_logits.append(logits.cpu().numpy()[0])
                    if start + cfg.chunk_len >= len(tokens):
                        break
                if cfg.chunk_agg == "max":
                    agg = np.max(torch.nn.functional.softmax(torch.tensor(chunk_logits), dim=-1).numpy(), axis=0)
                    all_logits.append(np.log(np.clip(agg, 1e-8, 1.0)))
                else:
                    all_logits.append(np.mean(np.stack(chunk_logits), axis=0))
    return np.stack(all_logits)


def temperature_scale_fit(logits: np.ndarray, targets: np.ndarray, max_iter: int = 50) -> float:
    """Fit a temperature parameter via optimization on log-temperature."""
    logits_t = torch.tensor(logits, dtype=torch.float32)
    targets_t = torch.tensor(targets, dtype=torch.long)
    log_temp = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=max_iter)

    def closure():  # type: ignore[override]
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits_t / torch.exp(log_temp), targets_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = torch.exp(log_temp).detach().item()
    return float(np.clip(temperature, 0.05, 5.0))


def temperature_scale_apply(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to logits."""
    return logits / temperature


def stack_fit(oof_probs: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Fit stacking meta-model."""
    meta = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
    )
    meta.fit(oof_probs, y)
    return meta


def stack_predict(meta: LogisticRegression, test_probs: np.ndarray) -> np.ndarray:
    """Predict using stacking meta-model."""
    return meta.predict_proba(test_probs)


def evaluate_oof(y_true: np.ndarray, oof_probs: np.ndarray, class_names: Sequence[str], output_dir: Path) -> Dict[str, Any]:
    """Evaluate OOF predictions and save artifacts."""
    preds = oof_probs.argmax(axis=1)
    macro_f1 = metrics.f1_score(y_true, preds, average="macro")
    micro_f1 = metrics.f1_score(y_true, preds, average="micro")
    per_class = metrics.classification_report(y_true, preds, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(per_class).transpose()
    df_report.to_csv(output_dir / "per_class_report.csv")
    cm = metrics.confusion_matrix(y_true, preds)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)
    return {"macro_f1": macro_f1, "micro_f1": micro_f1, "per_class": per_class, "confusion_matrix": cm.tolist()}


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def log_versions(cfg: Config) -> Dict[str, Any]:
    """Collect environment and library version information."""
    info = {
        "python": sys.version,
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "transformers": __import__("transformers").__version__,
        "datasets": __import__("datasets").__version__,
    }
    save_json(Path(cfg.output_dir) / "env_info.json", info)
    return info


def main(args: Optional[Dict[str, Any]] = None) -> None:
    cfg = Config(**args) if isinstance(args, dict) else Config.from_args()
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_info = log_versions(cfg)
    train_df, test_df = load_data(cfg.train_csv, cfg.test_csv, cfg)

    train_df[cfg.text_col] = train_df[cfg.text_col].apply(lambda x: clean_text(x, cfg))
    test_df[cfg.text_col] = test_df[cfg.text_col].apply(lambda x: clean_text(x, cfg))
    corpus_dataset = prepare_corpus_for_mlm(train_df, test_df, cfg)

    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_df[cfg.label_col].astype(str))
    save_json(output_dir / "label_encoder.json", {cls: int(idx) for idx, cls in enumerate(label_encoder.classes_)})

    class_weights = compute_class_weights(train_labels, cfg.alpha_class_weight)
    save_json(output_dir / "class_weights.json", class_weights.tolist())

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)

    oof_logits_models: Dict[str, np.ndarray] = {}
    test_logits_models: Dict[str, np.ndarray] = {}
    metrics_summary = []

    for model_name in cfg.model_list():
        print(f"=== Training model: {model_name} ===")
        base_path = cfg.model_dir_override or model_name
        tokenizer = AutoTokenizer.from_pretrained(base_path, cache_dir=cfg.cache_dir, use_fast=True)
        adapted_path = run_mlm(base_path, tokenizer, corpus_dataset, cfg) if cfg.mlm_enable else base_path
        tokenizer = AutoTokenizer.from_pretrained(adapted_path, cache_dir=cfg.cache_dir, use_fast=True)
        model_safe_name = safe_model_name(model_name)
        num_labels = len(label_encoder.classes_)
        oof_logits = np.zeros((len(train_df), num_labels), dtype=np.float32)
        test_logits_accum = np.zeros((len(test_df), num_labels), dtype=np.float32)
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_labels)):
            print(f"Fold {fold+1}/{cfg.n_folds}")
            train_split = train_df.iloc[train_idx].reset_index(drop=True)
            val_split = train_df.iloc[val_idx].reset_index(drop=True)

            metrics_fold, val_logits, trainer = train_fold(
                adapted_path,
                tokenizer,
                fold,
                train_split,
                val_split,
                label_encoder,
                cfg,
                class_weights,
            )

            oof_logits[val_idx] = val_logits
            fold_metrics.append({"fold": fold, **metrics_fold})

            model = trainer.model
            model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            logits_test = predict_with_sliding_window(model, tokenizer, test_df[cfg.text_col].tolist(), cfg)
            test_logits_accum += logits_test / cfg.n_folds
            del model
            del trainer
            torch.cuda.empty_cache()

        oof_logits_models[model_name] = oof_logits
        test_logits_models[model_name] = test_logits_accum

        preds = oof_logits.argmax(axis=1)
        np.save(output_dir / f"oof_logits_model={model_safe_name}.npy", oof_logits)
        np.save(output_dir / f"oof_preds_model={model_safe_name}.npy", preds)
        np.save(output_dir / f"test_logits_model={model_safe_name}.npy", test_logits_accum)
        fold_metrics_df = pd.DataFrame(fold_metrics)
        fold_metrics_df.to_csv(output_dir / f"fold_metrics_{model_safe_name}.csv", index=False)

        macro_f1 = metrics.f1_score(train_labels, preds, average="macro")
        metrics_summary.append({"model": model_name, "macro_f1": macro_f1})

        # Plot learning curve
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, len(fold_metrics_df) + 1), fold_metrics_df["eval_macro_f1"], marker="o", label="Macro F1")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Macro F1")
        ax.set_title(f"OOF Macro F1 per fold - {model_name}")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"oof_curve_{model_safe_name}.png", dpi=200)
        plt.close(fig)

        save_json(output_dir / f"val_metrics_{model_safe_name}.json", {"folds": fold_metrics_df.to_dict(orient="records")})

    np.save(output_dir / "oof_targets.npy", train_labels)

    calibrated_models = {}
    for model_name, logits in oof_logits_models.items():
        model_safe_name = safe_model_name(model_name)
        if cfg.calibrate:
            temperature = temperature_scale_fit(logits, train_labels)
            save_json(output_dir / f"temperature_{model_safe_name}.json", {"temperature": temperature})
            logits_scaled = temperature_scale_apply(logits, temperature)
            test_logits_scaled = temperature_scale_apply(test_logits_models[model_name], temperature)
        else:
            temperature = 1.0
            logits_scaled = logits
            test_logits_scaled = test_logits_models[model_name]
        probs = torch.softmax(torch.tensor(logits_scaled), dim=-1).numpy()
        test_probs = torch.softmax(torch.tensor(test_logits_scaled), dim=-1).numpy()
        np.save(output_dir / f"oof_probs_model={model_safe_name}.npy", probs)
        np.save(output_dir / f"test_probs_model={model_safe_name}.npy", test_probs)
        calibrated_models[model_name] = (probs, test_probs)

    all_oof_probs = np.concatenate([probs for probs, _ in calibrated_models.values()], axis=1)
    all_test_probs = np.concatenate([probs for _, probs in calibrated_models.values()], axis=1)

    if cfg.stacking and len(calibrated_models) > 1:
        meta = stack_fit(all_oof_probs, train_labels)
        save_path = output_dir / "stacking_meta.pkl"
        import joblib

        joblib.dump(meta, save_path)
        final_probs = stack_predict(meta, all_oof_probs)
        test_final_probs = stack_predict(meta, all_test_probs)
    else:
        final_probs = np.mean([probs for probs, _ in calibrated_models.values()], axis=0)
        test_final_probs = np.mean([probs for _, probs in calibrated_models.values()], axis=0)

    if cfg.prior_nudge_json:
        prior_path = Path(cfg.prior_nudge_json)
        if prior_path.exists():
            with prior_path.open("r", encoding="utf-8") as f:
                prior_data = json.load(f)
            multipliers = np.ones(len(label_encoder.classes_), dtype=np.float32)
            for key, value in prior_data.items():
                try:
                    if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
                        idx = int(key)
                    else:
                        idx = label_encoder.transform([str(key)])[0]
                    multipliers[idx] = float(value)
                except Exception:
                    continue
            def apply_prior(probs: np.ndarray) -> np.ndarray:
                adjusted = probs * multipliers
                adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
                return adjusted

            final_probs = apply_prior(final_probs)
            test_final_probs = apply_prior(test_final_probs)

    final_metrics = evaluate_oof(train_labels, final_probs, label_encoder.classes_, output_dir)

    summary_df = pd.DataFrame(metrics_summary)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("macro_f1", ascending=False)
        print(summary_df)
        summary_df.to_csv(output_dir / "model_summary.csv", index=False)

    print("Ensemble macro F1:", final_metrics["macro_f1"])

    sample_df = pd.read_csv(cfg.sample_csv)
    id_col = cfg.id_col if cfg.id_col in sample_df.columns else sample_df.columns[0]
    label_col = cfg.label_col if cfg.label_col in sample_df.columns else sample_df.columns[-1]
    pred_indices = test_final_probs.argmax(axis=1)
    sample_df[label_col] = label_encoder.inverse_transform(pred_indices)
    sample_df.to_csv(output_dir / "submission.csv", index=False)
    print(sample_df.head())
    print(sample_df[label_col].value_counts())

    run_config = {
        "config": asdict(cfg),
        "env": env_info,
        "metrics": {
            "per_model": metrics_summary,
            "ensemble": final_metrics,
        },
    }
    save_json(output_dir / "run_config.json", run_config)


if __name__ == "__main__":
    main()
