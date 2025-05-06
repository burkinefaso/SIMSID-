# ======================= eval.py =======================
import argparse
import importlib
from pathlib import Path

import torch
import numpy as np
from scipy.special import expit
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix
)

# 1) Config ve Dataset loader’lar
from configs.karma4_dev import Config          # CONFIG içinde train/test loader var
from dataloader.dataloader_karma4 import KarmaDataset

# 2) GanAlert – eğitim skorlarını toplamak için
from alert import GanAlert                      # .collect() metodunu kullanacağız


def plot_roc(labels, scores, path):
    fpr, tpr, thresh = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],"--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Eğrisi")
    plt.legend(loc="lower right")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return fpr, tpr, thresh, roc_auc

def plot_prc(labels, scores, path):
    prec, rec, _ = precision_recall_curve(labels, scores)
    prc_auc = auc(rec, prec)
    plt.figure()
    plt.step(rec, prec, where="post", lw=2, label=f"AUC={prc_auc:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Eğrisi")
    plt.legend(loc="lower left")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return rec, prec, prc_auc

def plot_confusion(cm, path):
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix"); plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ["Normal","Anomali"]); plt.yticks(ticks, ["Normal","Anomali"])
    thresh = cm.max()/2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j], ha="center", va="center",
                     color="white" if cm[i,j]>thresh else "black")
    plt.ylabel("Gerçek"); plt.xlabel("Tahmin")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="SimSID Eval Script")
    parser.add_argument("--exp", required=True, help="checkpoints/<exp> klasörü")
    args = parser.parse_args()

    # A) CONFIG ve device
    cfg = Config()
    device = torch.device(cfg.device)

    # B) Model ve Discriminator yükle (checkpoints/<exp>/squid.py içinden)
    spec = importlib.import_module(f"checkpoints.{args.exp}.squid")
    # features_root sabit 32, checkpoint’inize uyacak şekilde
    encoder = spec.AE(cfg, features_root=32, level=cfg.level).to(device)
    try:
        discriminator = spec.Discriminator(cfg).to(device)
    except AttributeError:
        from tools import build_disc
        discriminator = build_disc(cfg).to(device)
    ckpt = Path("checkpoints")/args.exp
    encoder.load_state_dict(torch.load(ckpt/"model.pth",         map_location=device))
    discriminator.load_state_dict(torch.load(ckpt/"discriminator.pth", map_location=device))
    encoder.eval(); discriminator.eval()

    # C) Loader’lar doğrudan CONFIG’ten
    train_loader = cfg.train_loader
    test_loader  = cfg.test_loader

    # D) GanAlert ile eğitim skorlarını topla → mean/std
    alert = GanAlert(discriminator=discriminator,
                     args=argparse.Namespace(),
                     CONFIG=cfg,
                     generator=encoder)
    train_scores, _ = alert.collect(train_loader)
    mean, std = train_scores.mean().item(), train_scores.std().item()

    # E) Test seti raw discriminator skorları & etiketler
    raw_scores, labels = [], []
    for imgs, labs in tqdm(test_loader, desc="Test skorları"):
        imgs = imgs.to(device)
        out  = encoder(imgs)
        recon = out["recon"] if isinstance(out, dict) else encoder.decode(out)
        logits = discriminator(recon).view(-1)
        raw_scores += logits.cpu().tolist()
        labels     += labs.cpu().tolist()

    raw   = np.array(raw_scores)

    # F) Eq.(2): normalize ve ters sigmoid
    norm   = (raw - mean) / (std + 1e-8)
    scores = 1.0 - expit(norm)

    # G) ROC ve PRC çiz, AUC hesapla
    fpr, tpr, thresh, roc_auc = plot_roc(labels, scores, ckpt/"test"/"roc.png")
    rec, prec, prc_auc         = plot_prc(labels, scores, ckpt/"test"/"prc.png")

    # H) Youden eşiği
    youden_idx = int(np.argmax(tpr - fpr))
    best_thr   = thresh[youden_idx]

    # I) Confusion Matrix
    preds = [1 if s>=best_thr else 0 for s in scores]
    cm = confusion_matrix(labels, preds)
    plot_confusion(cm, ckpt/"test"/"confmat.png")


    print("DEBUG → Normal örnek sayısı:", sum(1 for s,y in zip(scores,labels) if y==0))
    print("DEBUG → Anomali örnek sayısı:", sum(1 for s,y in zip(scores,labels) if y==1))
    print("DEBUG → Score min/max:", min(scores), max(scores))

    # 2) Inline histogram (adet ekseninde)
    normal_scores  = [s for s,y in zip(scores,labels) if y==0]
    anomaly_scores = [s for s,y in zip(scores,labels) if y==1]

    # J) Histogram (adet ekseni)
    plt.figure()
    plt.hist([s for s,y in zip(scores,labels) if y==0],
             bins=50, alpha=0.7, label="Normal", density=False)
    plt.hist([s for s,y in zip(scores,labels) if y==1],
             bins=50, alpha=0.7, label="Anomali", density=False)
    plt.xlabel("Anomali Skoru"); plt.ylabel("Örnek Sayısı")
    plt.title("Skor Histogramı")
    plt.legend()
    plt.savefig(ckpt/"test"/"hist.png", dpi=300, bbox_inches="tight")
    plt.close()

    # K) Log’u JSON’a yaz
    log_path = ckpt/"test"/"eval_log.txt"
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "w") as f:
        import json
        json.dump({
            "mean": mean,
            "std": std,
            "roc_auc": roc_auc,
            "prc_auc": prc_auc,
            "youden_threshold": float(best_thr),
            "confusion_matrix": cm.tolist()
        }, f, indent=2)

    print(f"[OK] ROC AUC={roc_auc:.4f}, PRC AUC={prc_auc:.4f}, Thr={best_thr:.4f}")
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    main()
