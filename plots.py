"""breast_cancer_model.py icin grafikler (PNG olarak kaydedilir).

Renkler doğrulanmış bir paletten gelir: mavi ve turuncu renk körlüğünde
de ayırt edilebilir. Her grafikte seriler ayrıca doğrudan etiketlenir,
böylece anlam yalnızca renge bağlı kalmaz.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

# Palet ve mürekkep tonları
BLUE = "#2a78d6"
ORANGE = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a8201f"]
)
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#f0efec", "#86b6ef", "#2a78d6", "#104281"])


def _tr(x, _pos=None, digits=2):
    """Türkçe ondalık ayırıcı (virgül)."""
    return f"{x:.{digits}f}".replace(".", ",")


def tr_formatter(digits: int = 2) -> FuncFormatter:
    return FuncFormatter(lambda x, pos: _tr(x, pos, digits))


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 26,
        "legend.frameon": False,
    })


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _subtitle(ax, text: str) -> None:
    ax.text(0, 1.015, text, transform=ax.transAxes, color=INK_2, fontsize=10, va="bottom")


def plot_correlation(X, feature_names, path: Path) -> None:
    """Yalnızca 'mean' ile başlayan 10 özellik; 30x30 harita okunmaz."""
    idx = [i for i, n in enumerate(feature_names) if n.startswith("mean ")]
    names = [feature_names[i].replace("mean ", "") for i in idx]
    corr = np.corrcoef(X[:, idx], rowvar=False)
    # Alt üçgen; köşegen ve boş kalan ilk satır/son sütun çıkarıldı
    sub = corr[1:, :-1]
    mask = np.triu(np.ones_like(sub, dtype=bool), k=1)
    shown = np.ma.array(sub, mask=mask)

    fig, ax = plt.subplots(figsize=(8, 6.8))
    ax.grid(False)
    im = ax.imshow(shown, cmap=DIVERGING, vmin=-1, vmax=1)
    for i in range(sub.shape[0]):
        for j in range(i + 1):
            val = sub[i, j]
            ax.text(j, i, _tr(val), ha="center", va="center", fontsize=8,
                    color="white" if abs(val) > 0.6 else INK)
    ax.set_xticks(range(len(names) - 1), names[:-1], rotation=45, ha="right")
    ax.set_yticks(range(len(names) - 1), names[1:])
    for s in ax.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, format=tr_formatter(1))
    cbar.outline.set_visible(False)
    cbar.set_label("Pearson korelasyonu")
    ax.set_title("Yarıçap, çevre ve alan neredeyse aynı bilgiyi taşıyor")
    _subtitle(ax, "10 'ortalama' özellik arasındaki korelasyon, 569 örnek")
    _save(fig, path)


def plot_loss_curve(history: dict, path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, history["train_loss"], color=BLUE, lw=2)
    ax.plot(epochs, history["test_loss"], color=ORANGE, lw=2)
    ax.axhline(np.log(2), color=MUTED, lw=1, ls="--")
    ax.text(epochs[-1], np.log(2) + 0.01, "ln 2 ≈ 0,693 (her örneğe 0,5 olasılık)",
            ha="right", va="bottom", color=INK_2, fontsize=9)
    # Doğrudan etiketler (lejant yerine)
    # Uç değerler birbirine yakın: eğitim etiketi üstte, test altta
    ax.text(epochs[-1] + 1.5, history["train_loss"][-1] + 0.012, f"Eğitim  {_tr(history['train_loss'][-1], digits=3)}",
            color=INK, va="bottom", fontsize=10)
    ax.text(epochs[-1] + 1.5, history["test_loss"][-1] - 0.012, f"Test  {_tr(history['test_loss'][-1], digits=3)}",
            color=INK, va="top", fontsize=10)
    ax.set_xlim(0, epochs[-1] + 22)
    ax.set_ylim(0, 0.75)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("İkili çapraz entropi")
    ax.yaxis.set_major_formatter(tr_formatter(1))
    ax.set_title("Kayıp ilk 10 epoch'ta hızla düşüyor")
    _subtitle(ax, "Öğrenme oranı 0,1, ağırlıklar sıfırla başlatıldı")
    _save(fig, path)


def plot_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    labels = ["Kötü huylu", "İyi huylu"]
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.grid(False)
    ax.imshow(cm, cmap=SEQUENTIAL, vmin=0, vmax=cm.max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=18, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() * 0.5 else INK)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("114 test örneğinde 2 hata")
    _subtitle(ax, "Eşik 0,5 · sağ üst: iyi huylu sanılan kötü huylu tümör")
    _save(fig, path)


def plot_split_variability(results: dict, reference: dict, path: Path) -> None:
    """30 bölmenin dağılımı: her nokta bir bölme; kısa çizgi ortalama."""
    rng = np.random.default_rng(0)
    metrics = [("accuracy", "Doğruluk"), ("malignant_recall", "Kötü huylu\nduyarlılık")]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for k, (key, label) in enumerate(metrics):
        vals = results[key]
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), k) + jitter, vals, s=36, color=BLUE, alpha=0.75,
                   edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.hlines(vals.mean(), k - 0.25, k + 0.25, color=INK, lw=2, zorder=4)
        ax.text(k + 0.3, vals.mean(), f"ort. {_tr(vals.mean(), digits=3)}", va="center", fontsize=10, color=INK)
        ax.scatter([k], [reference[key]], marker="D", s=60, color=ORANGE, edgecolor=SURFACE,
                   linewidth=1.2, zorder=5)
    ax.annotate("random_state=42", xy=(0.04, reference["accuracy"]), xytext=(0.3, reference["accuracy"] + 0.012),
                color=INK_2, fontsize=9, va="bottom",
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax.set_xticks([0, 1], [m[1] for m in metrics])
    ax.set_xlim(-0.6, 1.9)
    ax.set_ylim(0.84, 1.01)
    ax.yaxis.set_major_formatter(tr_formatter(2))
    ax.grid(axis="x", visible=False)
    ax.set_title("Aynı model, 30 farklı bölme")
    _subtitle(ax, "Her nokta bir bölme (stratify=y) · turuncu: yazıdaki tek bölme")
    _save(fig, path)


def plot_threshold(sweep: dict, path: Path) -> None:
    t = sweep["threshold"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.step(t, sweep["false_negative"], where="mid", color=ORANGE, lw=2)
    ax.step(t, sweep["false_positive"], where="mid", color=BLUE, lw=2)
    ax.axvline(0.5, color=MUTED, lw=1, ls="--")
    ax.text(0.505, ax.get_ylim()[1] * 0.95, "varsayılan eşik 0,5", color=INK_2, fontsize=9, va="top")
    ax.text(t[3], sweep["false_negative"][3] + 0.3, "Kaçan kötü huylu (yanlış negatif)",
            color=INK, fontsize=10, va="bottom")
    ax.text(t[-4], sweep["false_positive"][-4] + 0.3, "Yanlış alarm (yanlış pozitif)",
            color=INK, fontsize=10, va="bottom", ha="right")
    ax.set_xlabel("Eşik: P(iyi huylu) bu değerin üstündeyse 'iyi huylu' de")
    ax.set_ylabel("Hata sayısı (114 örnekte)")
    ax.xaxis.set_major_formatter(tr_formatter(1))
    ax.set_title("Eşik yükseldikçe kaçan kötü huylu azalıyor, yanlış alarm artıyor")
    _subtitle(ax, "Test seti · yalnızca açıklama amaçlı, eşik burada seçilmedi")
    _save(fig, path)


def plot_gradient_check(analytic, numeric, rel_diff: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    lo = min(analytic.min(), numeric.min())
    hi = max(analytic.max(), numeric.max())
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", zorder=1)
    ax.scatter(analytic, numeric, s=40, color=BLUE, edgecolor=SURFACE, linewidth=1, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Analitik gradyan  Xᵀ(ŷ − y) / n")
    ax.set_ylabel("Sayısal gradyan (merkezi fark)")
    ax.xaxis.set_major_formatter(tr_formatter(2))
    ax.yaxis.set_major_formatter(tr_formatter(2))
    exp = int(np.floor(np.log10(rel_diff)))
    mant = rel_diff / 10**exp
    ax.text(0.04, 0.96, f"30 ağırlığın tamamı y = x çizgisinde\nen büyük göreli fark ≈ {_tr(mant, digits=1)} × $10^{{{exp}}}$",
            transform=ax.transAxes, va="top", fontsize=9.5, color=INK_2)
    ax.set_title("Analitik ve sayısal gradyan örtüşüyor")
    _save(fig, path)
