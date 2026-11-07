"""MIUUL - Breast Cancer Classification | Logistic Regression from Scratch.

PyCharm: python breast_cancer_model.py
Paketler: pip install numpy pandas matplotlib seaborn scikit-learn
Grafikler icin ayni klasorde plots.py dosyasi olmali.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import plots

# True yaparsan EDA grafiklerini de olusturur.
RUN_EDA = False

# Grafiklerin kaydedilecegi klasor
FIGURE_DIR = "figures"


def load_data():
    """30 ozellik, hedef sinif ve ozellik adlarini dondurur."""
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    return X, y, feature_names


def explore_data(X, y, feature_names):
    """Veriyi tablo, istatistik ve grafiklerle incele."""
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    print("\n=== VERIYI TANIMA (EDA) ===")
    print("Boyut:", df.shape)
    print("\nIlk 5 satir:\n", df.head())
    print("\nOzet istatistikler:\n", df.describe().round(2))
    print("\nSinif dagilimi:\n", df["target"].value_counts())
    print("\nEksik deger sayisi:", df.isnull().sum().sum())
    print("\nSiniflara gore mean radius:\n", df.groupby("target")["mean radius"].agg(["count", "mean", "median", "std", "min", "max"]).round(2))

    selected = ["mean radius", "mean perimeter", "mean texture", "mean area", "target"]
    corr_matrix = df[selected].corr()
    print("\nKorelasyon matrisi:\n", corr_matrix.round(2))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes[0, 0].hist(df["mean radius"], bins=20)
    axes[0, 0].set(title="Mean Radius Histogram", xlabel="Mean Radius", ylabel="Frekans")

    sns.countplot(data=df, x="target", ax=axes[0, 1])
    axes[0, 1].set(title="Sinif Dagilimi", xlabel="0: Malignant, 1: Benign")

    sns.boxplot(data=df, x="target", y="mean radius", ax=axes[1, 0])
    axes[1, 0].set(title="Siniflara Gore Mean Radius", xlabel="Target")

    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, center=0, ax=axes[1, 1])
    axes[1, 1].set_title("Korelasyon Matrisi")
    fig.tight_layout()
    return df


def prepare_data(X, y, seed=42, stratify=False):
    """Once bol, sonra sadece train verisine fit yaparak olcekle."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed,
        stratify=y if stratify else None,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def sigmoid(z):
    """Ham ciktiyi 0-1 arasinda tahmin olasiligina donusturur."""
    return 1 / (1 + np.exp(-z))


def compute_loss(y, y_pred):
    """Dersteki epsilon eklemeli Binary Cross-Entropy (beklenen ciktiyla ayni)."""
    epsilon = 1e-5
    return -np.mean(
        y * np.log(y_pred + epsilon)
        + (1 - y) * np.log(1 - y_pred + epsilon)
    )


def compute_gradients(X, y, y_pred):
    """Her ozellik icin ayri gradient hesabi (vektor: 30 eleman)."""
    return np.dot(X.T, (y_pred - y)) / len(y)


def gradient_descent(X, y, lr=0.1, epochs=100, X_test=None, y_test=None, verbose=True):
    """Tum train verisiyle her epoch bir kez parametre guncelle.

    X_test ve y_test verilirse test kaybi da kaydedilir. Bu veri sadece
    izlenir, agirlik guncellemesine hic katilmaz.
    """
    weights = np.zeros(X.shape[1])
    bias = 0.0
    history = {"train_loss": [], "test_loss": []}

    for epoch in range(epochs):
        z = np.dot(X, weights) + bias
        y_pred = sigmoid(z)
        loss = compute_loss(y, y_pred)
        history["train_loss"].append(loss)

        if X_test is not None:
            test_pred = sigmoid(np.dot(X_test, weights) + bias)
            history["test_loss"].append(compute_loss(y_test, test_pred))

        dw = compute_gradients(X, y, y_pred)
        db = np.mean(y_pred - y)

        weights -= lr * dw
        bias -= lr * db

        if verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
            print(f"Epoch {epoch + 1:3d}/{epochs} | Loss: {loss:.6f}")

    return weights, bias, history


def predict_proba(X, weights, bias):
    """Sinif 1 (Benign) icin olasilik doner."""
    return sigmoid(np.dot(X, weights) + bias)


def predict(X, weights, bias, threshold=0.5):
    """Olasiligi 0 veya 1 etiketine donusturur."""
    probabilities = predict_proba(X, weights, bias)
    return (probabilities >= threshold).astype(int)


def evaluate_model(y_true, y_pred):
    """Test performansini raporla; pozitif sinif Malignant = 0."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("\n=== MODEL PERFORMANSI ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.6f}")
    print("\nConfusion Matrix [0, 1] sirasi:\n", cm)
    print("\nClassification Report:\n", classification_report(
        y_true, y_pred, labels=[0, 1],
        target_names=["Malignant", "Benign"], digits=4
    ))
    print(f"Malignant Precision: {precision_score(y_true, y_pred, pos_label=0):.6f}")
    print(f"Malignant Recall:    {recall_score(y_true, y_pred, pos_label=0):.6f}")
    print(f"Malignant F1:        {f1_score(y_true, y_pred, pos_label=0):.6f}")
    return cm


def analyze_errors(X_test, y_test, y_pred, probabilities, feature_names):
    """Yanlis siniflandirilan test orneklerinin olasilik ve ozelliklerini yazdir."""
    wrong_indices = np.where(y_pred != y_test)[0]
    wrong_predictions = pd.DataFrame({
        "Test Index": wrong_indices,
        "Gercek Sinif": y_test[wrong_indices],
        "Tahmin": y_pred[wrong_indices],
        "P(Benign)": probabilities[wrong_indices],
        "P(Malignant)": 1 - probabilities[wrong_indices],
    })
    print("\n=== HATALI TAHMINLER ===")
    print(wrong_predictions.round(4).to_string(index=False))

    selected_columns = ["mean radius", "mean texture", "mean perimeter", "mean area"]
    wrong_features = pd.DataFrame(X_test[wrong_indices], columns=feature_names)
    print("\nHatali gozlemlerin orijinal ozellikleri:\n",
          wrong_features[selected_columns].round(2).to_string(index=False))
    return wrong_predictions


def compare_with_sklearn(X_train, y_train, X_test, y_test, probabilities):
    """Kendi modelimizi sklearn LogisticRegression ile karsilastir.

    Not: sklearn varsayilan olarak L2 duzenlilestirme kullanir, bizim
    modelimizde duzenlilestirme yok. Birebir ayni sonuc beklenmez.
    """
    sk_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    sk_probabilities = sk_model.predict_proba(X_test)[:, 1]

    ours = summarize(y_test, probabilities)
    theirs = summarize(y_test, sk_probabilities)

    print("\n=== SKLEARN ILE KARSILASTIRMA ===")
    print(f"{'':20}{'Bizim model':>12}{'sklearn':>10}")
    for metric in ours:
        print(f"{metric:20}{ours[metric]:12.3f}{theirs[metric]:10.3f}")


def summarize(y_true, probabilities, threshold=0.5):
    """Accuracy, malignant recall ve ROC AUC degerlerini dondur."""
    preds = (probabilities >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, preds),
        "Malignant Recall": recall_score(y_true, preds, pos_label=0),
        "ROC AUC": roc_auc_score(y_true, probabilities),
    }


def repeated_splits(X, y, n_splits=30, lr=0.1, epochs=100):
    """Ayni modeli farkli rastgele bolmelerde egit.

    Tek bir random_state ile alinan sonucun ne kadar degisebilecegini gosterir.
    Sinif oranini korumak icin stratify kullanildi.
    """
    accuracies = []
    recalls = []
    for seed in range(n_splits):
        _, _, X_tr, X_te, y_tr, y_te, _ = prepare_data(X, y, seed=seed, stratify=True)
        weights, bias, _ = gradient_descent(X_tr, y_tr, lr=lr, epochs=epochs, verbose=False)
        preds = predict(X_te, weights, bias)
        accuracies.append(accuracy_score(y_te, preds))
        recalls.append(recall_score(y_te, preds, pos_label=0))

    accuracies = np.array(accuracies)
    recalls = np.array(recalls)
    print(f"\n=== {n_splits} FARKLI BOLME ===")
    print(f"Accuracy:         ort. {accuracies.mean():.3f} | std {accuracies.std():.3f} | "
          f"min {accuracies.min():.3f} | max {accuracies.max():.3f}")
    print(f"Malignant Recall: ort. {recalls.mean():.3f} | min {recalls.min():.3f}")
    return {"accuracy": accuracies, "malignant_recall": recalls}


def numerical_gradient(loss_fn, w, h=1e-6):
    """Her agirligi biraz arttirip azaltarak gradyani sayisal olarak hesapla."""
    grad = np.zeros_like(w)
    for i in range(len(w)):
        step = np.zeros_like(w)
        step[i] = h
        grad[i] = (loss_fn(w + step) - loss_fn(w - step)) / (2 * h)
    return grad


def gradient_check(X, y):
    """Elle turettigimiz gradyan ile sayisal gradyani karsilastir.

    Burada epsilon'suz kayip kullaniliyor, cunku analitik formul
    epsilon'suz kaybin turevi.
    """
    rng = np.random.default_rng(0)
    w = rng.normal(scale=0.1, size=X.shape[1])

    def loss_without_epsilon(weights):
        p = sigmoid(np.dot(X, weights))
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    analytic = compute_gradients(X, y, sigmoid(np.dot(X, w)))
    numeric = numerical_gradient(loss_without_epsilon, w)
    max_diff = np.max(np.abs(numeric - analytic)) / np.max(np.abs(analytic))

    print("\n=== GRADYAN KONTROLU ===")
    print(f"En buyuk goreli fark: {max_diff:.1e}")
    return analytic, numeric, max_diff


def threshold_sweep(y_true, probabilities):
    """Esik degistikce kacan malignant ve yanlis alarm sayilarini hesapla.

    Model P(Benign) verir. Esik yukseldikce 'benign' demek zorlasir.
    Sadece grafik icin; burada esik secilmiyor.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    false_negatives = []
    false_positives = []
    for t in thresholds:
        preds = (probabilities >= t).astype(int)
        false_negatives.append(np.sum((y_true == 0) & (preds == 1)))
        false_positives.append(np.sum((y_true == 1) & (preds == 0)))
    return {
        "threshold": thresholds,
        "false_negative": np.array(false_negatives),
        "false_positive": np.array(false_positives),
    }


def save_figures(X, feature_names, history, cm, split_results, reference, sweep, grad_results):
    """Tum grafikleri figures/ klasorune kaydet."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plots.set_style()
    analytic, numeric, max_diff = grad_results

    plots.plot_correlation(X, list(feature_names), os.path.join(FIGURE_DIR, "fig1_korelasyon.png"))
    plots.plot_loss_curve(history, os.path.join(FIGURE_DIR, "fig2_kayip_egrisi.png"))
    plots.plot_confusion_matrix(cm, os.path.join(FIGURE_DIR, "fig3_karisiklik_matrisi.png"))
    plots.plot_split_variability(split_results, reference, os.path.join(FIGURE_DIR, "fig4_bolme_degiskenligi.png"))
    plots.plot_threshold(sweep, os.path.join(FIGURE_DIR, "fig5_esik_analizi.png"))
    plots.plot_gradient_check(analytic, numeric, max_diff, os.path.join(FIGURE_DIR, "fig6_gradyan_kontrolu.png"))
    print(f"\nGrafikler {FIGURE_DIR}/ klasorune kaydedildi.")


def main():
    # 1. Veriyi yukle
    X, y, feature_names = load_data()
    print("X:", X.shape, "| y:", y.shape)

    # 2. Istersen veri analizi grafiklerini de ac
    if RUN_EDA:
        explore_data(X, y, feature_names)

    # 3. Train / Test ve olcekleme
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(X, y)
    print("Train:", X_train_scaled.shape, "| Test:", X_test_scaled.shape)

    # 4. Modeli sifirdan egit (test kaybi sadece izleniyor)
    weights, bias, history = gradient_descent(
        X_train_scaled, y_train, lr=0.1, epochs=100,
        X_test=X_test_scaled, y_test=y_test,
    )

    # 5. Test verisi icin olasilik ve etiket tahmin et
    probabilities = predict_proba(X_test_scaled, weights, bias)
    predictions = predict(X_test_scaled, weights, bias)
    print("\nIlk 5 P(Benign):", probabilities[:5].round(4))
    print("Ilk 5 tahmin:  ", predictions[:5])
    print("Ilk 5 gercek:  ", y_test[:5])

    # 6. Metrikler ve hatali gozlemler
    cm = evaluate_model(y_test, predictions)
    analyze_errors(X_test, y_test, predictions, probabilities, feature_names)

    # 7. sklearn ile karsilastir
    compare_with_sklearn(X_train_scaled, y_train, X_test_scaled, y_test, probabilities)

    # 8. Tek bolmeye ne kadar guvenebiliriz?
    split_results = repeated_splits(X, y, n_splits=30, lr=0.1, epochs=100)

    # 9. Gradyan kontrolu ve esik analizi
    grad_results = gradient_check(X_train_scaled, y_train)
    sweep = threshold_sweep(y_test, probabilities)

    # 10. Grafikleri kaydet
    reference = {
        "accuracy": accuracy_score(y_test, predictions),
        "malignant_recall": recall_score(y_test, predictions, pos_label=0),
    }
    save_figures(X, feature_names, history, cm, split_results, reference, sweep, grad_results)

    if RUN_EDA:
        plt.show()


if __name__ == "__main__":
    main()
