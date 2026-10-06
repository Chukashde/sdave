import argparse, glob, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True,
                    help='нэр=glob, жишээ: seq="results/seq_w_s*.json"')
    ap.add_argument('--out', default='results/curriculum')
    a = ap.parse_args()

    data = {}
    for item in a.runs:
        name, pat = item.split('=', 1)
        data[name] = [json.load(open(f)) for f in sorted(glob.glob(pat))]
        print(f'{name}: {len(data[name])} run')

    first = next(iter(data.values()))[0]
    splits = list(first[0]['test'].keys())
    K = len(first)
    graphs = [s['graph'].split('/')[-1].replace('.txt', '') for s in first]

    def curve(runs, sp):
        arr = np.array([[st['test'][sp][1] for st in r] for r in runs], dtype=float)
        return arr.mean(0), arr.std(0)

    for sp in splits:
        print(f'\n== {sp} (cut/ref, mean±std) ==')
        print('stage | graph | ' + ' | '.join(data))
        cs = {n: curve(r, sp) for n, r in data.items()}
        for k in range(K):
            cells = [f'{cs[n][0][k]:.3f}±{cs[n][1][k]:.3f}' for n in data]
            print(f'{k+1} | {graphs[k]} | ' + ' | '.join(cells))

    n_plots = len(splits) + 1
    fig, axs = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 4))
    x = np.arange(1, K + 1)
    for ax, sp in zip(axs, splits):
        for n, runs in data.items():
            m, s = curve(runs, sp)
            ax.plot(x, m, marker='o', label=n)
            ax.fill_between(x, m - s, m + s, alpha=0.2)
        ax.set_title(sp); ax.set_xlabel('stage'); ax.set_ylabel('cut / ref')
        ax.set_xticks(x); ax.grid(alpha=0.3)
    ax = axs[-1]
    for n, runs in data.items():
        arr = np.array([[st['seen']['G1'][0] for st in r] for r in runs])
        ax.plot(x, arr.mean(0), marker='o', label=n)
    ax.set_title('forgetting: G1 cut/edge'); ax.set_xlabel('stage')
    ax.set_xticks(x); ax.grid(alpha=0.3)
    axs[0].legend()
    plt.tight_layout()
    plt.savefig(a.out + '.png', dpi=150)
    print('saved', a.out + '.png')


if __name__ == '__main__':
    main()
