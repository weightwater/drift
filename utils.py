import torch
from tqdm import tqdm
import numpy as np

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

def predict(test_loader, model, device):
    model.eval()
    preds = torch.Tensor([])
    for x in tqdm(test_loader):
        x = x.float().to(device)
        with torch.no_grad():
            pred = model(x)
            preds = torch.concat((preds, pred.detach().to('cpu')))
    # prds = torch.cat(preds, dim=0).numpy()
    return preds

def getMSE(y, yHat):
    num = y.shape[1]
    return np.sum((y-yHat)**2, axis=1) / num


def draw_hitogram(df, column='MAPE'):
    sns.set_theme(style="ticks")

    f, ax = plt.subplots(figsize=(7, 5))
    sns.despine(f)

    sns.histplot(
        df,
        x=column,
        multiple="stack",
        palette="light:m_r",
        edgecolor=".3",
        linewidth=.5,
        log_scale=False,
    )
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    # ax.set_xticks([500, 1000, 2000, 5000, 10000])