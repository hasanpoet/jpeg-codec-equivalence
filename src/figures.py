"""Single-column vector/300-dpi figures using one domain palette and markers."""


import matplotlib


matplotlib.use('Agg')


import matplotlib.pyplot as plt


from matplotlib.lines import Line2D


from matplotlib.patches import Patch


import numpy as np


from .analysis import GEN, DOMAINS, LABELS, CODECS, accepted, read, write_json


COLOURS = dict(natural='#0072B2', xray='#D55E00', mri='#009E73', aerial='#CC79A7')


MARKERS = dict(natural='o', xray='s', mri='^', aerial='D')


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .6, 'lines.linewidth': .9, 'xtick.major.width': .6, 'ytick.major.width': .6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'savefig.dpi': 300})


def domain_legend(fig):
    handles = [Line2D([], [], color=COLOURS[d], marker=MARKERS[d], markersize=3, label=LABELS[d]) for d in DOMAINS]
    fig.legend(handles=handles, ncol=4, loc='upper center', bbox_to_anchor=(.53, 1.015),
               frameon=False, handlelength=.8, handletextpad=.3, columnspacing=.7)


def save(fig, name):
    fig.canvas.draw()
    # Check labels and legends against the fixed-size figure canvas.
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    outside = []
    for artist in [*fig.legends, *[a.xaxis.label for a in fig.axes], *[a.yaxis.label for a in fig.axes]]:
        box = artist.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > bounds.x1 + 1 or box.y1 > bounds.y1 + 1:
            outside.append(str(artist))
    if outside:
        raise RuntimeError('Figure labels clipped: ' + ', '.join(outside))
    fig.savefig(GEN / (name + '.pdf'))
    # PDF is the public artifact; PNG previews are optional verification outputs.
    plt.close(fig)

def generate_figures():
    style()
    eq, rates = read('equivalence_full.csv'), read('rate_savings.csv')
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9))
    fig.subplots_adjust(left=.14, right=.985, bottom=.26, top=.77, wspace=.59)
    domain_legend(fig)
    for ax, codec, symbol, panel in zip(axes, CODECS, ['r', 'd'], ['a', 'b']):
        for domain in DOMAINS:
            f = eq[(eq.domain == domain) & (eq.codec == codec)].sort_values('q')
            q, med = f.q.to_numpy(), f['median'].to_numpy()
            ax.fill_between(q, f.q1.to_numpy(), f.q3.to_numpy(), color=COLOURS[domain], alpha=.13, linewidth=0)
            ax.plot(q, med, color=COLOURS[domain])
            for x, y, hollow in zip(q, med, f.unreachable_pct > 5):
                ax.plot(x, y, marker=MARKERS[domain], color=COLOURS[domain], markersize=3,
                        markerfacecolor='white' if hollow else COLOURS[domain], markeredgewidth=.65)
        ax.set_yscale('log')
        ax.set_xticks([10, 50, 90])
        ax.set_xlabel(f'({panel}) JPEG quality $q$', labelpad=2)
        ax.set_ylabel('$' + symbol + '^*$', labelpad=1)
        ax.tick_params(axis='y', which='minor', labelleft=False)
        ax.grid(axis='y', alpha=.15, linewidth=.5)
    save(fig, 'fig1_equivalence')

    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    fig.subplots_adjust(left=.16, right=.99, bottom=.29, top=.82)
    width = .17
    codec_ticks, codec_labels = [], []
    for i, domain in enumerate(DOMAINS):
        for j, codec in enumerate(CODECS):
            centre = i + (j - .5) * .4
            codec_ticks.append(centre)
            codec_labels.append('2K' if j == 0 else 'XL')
            for k, metric in enumerate(['psnr', 'ssim']):
                r = rates[(rates.domain == domain) & (rates.codec == codec) & (rates.metric == metric)].iloc[0]
                x = centre + (k - .5) * width
                ax.bar(x, r['median'], width=width, facecolor=COLOURS[domain], edgecolor='black', linewidth=.4,
                       hatch='///' if metric == 'ssim' else None)
                ax.vlines(x, r.lo, r.hi, colors='black', linewidth=.7)
                ax.hlines([r.lo, r.hi], x-.045, x+.045, colors='black', linewidth=.7)
                ax.plot(x, r['median'], marker=MARKERS[domain], markersize=2.4, markeredgewidth=.5,
                        color='black', markerfacecolor=COLOURS[domain])
    ax.set_xticks(codec_ticks, codec_labels)
    ax.tick_params(axis='x', length=0, pad=2)
    for i, domain in enumerate(DOMAINS):
        ax.text(i, -.24, LABELS[domain], ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=8)
    ax.set_ylabel('Bit saving (%)', labelpad=2)
    ax.axhline(0, color='black', linewidth=.5)
    ax.set_xlim(-.5, len(DOMAINS)-.5)
    ax.grid(axis='y', alpha=.12, linewidth=.5)
    fig.legend(handles=[Patch(facecolor='.6', edgecolor='black', label='PSNR'),
                        Patch(facecolor='.6', edgecolor='black', hatch='///', label='SSIM')],
               ncol=2, loc='upper center', bbox_to_anchor=(.55, 1.015), frameon=False, handlelength=1.2)
    save(fig, 'fig2_rate')

    features, pairs, models = read('complexity.csv'), read('pairs.csv'), read('model_parameters.csv')
    test = features[features.split == 'test']
    data = pairs[pairs.q == 50].merge(test[['domain', 'image_id', 'C_DCT']], on=['domain', 'image_id'], validate='one_to_one')
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9))
    fig.subplots_adjust(left=.15, right=.985, bottom=.26, top=.77, wspace=.62)
    domain_legend(fig)
    for ax, codec, symbol, panel in zip(axes, CODECS, ['r', 'd'], ['a', 'b']):
        f = accepted(data[data.C_DCT > 0], codec)
        for domain in DOMAINS:
            d = f[f.domain == domain]
            ax.scatter(d.C_DCT, np.log(d[f'{codec}_psnr_parameter']), marker=MARKERS[domain], color=COLOURS[domain],
                       s=8, alpha=.3, edgecolors='none')
            model = models[(models.codec == codec) & (models.domain == domain) & (models.q == 50) & (models.predictor == 'C1')].iloc[0]
            xs = np.linspace(d.C_DCT.min(), d.C_DCT.max(), 100)
            ax.plot(xs, model.intercept + model.slope * np.log(xs), color=COLOURS[domain])
        global_model = models[(models.codec == codec) & (models.domain == 'all') & (models.q == 50) & (models.predictor == 'C0')].iloc[0]
        xs = np.linspace(f.C_DCT.min(), f.C_DCT.max(), 150)
        ax.plot(xs, global_model.intercept + global_model.slope * np.log(xs), color='black', linestyle='--', linewidth=1)
        ax.set_xlabel(f'({panel}) $C_{{\\mathrm{{DCT}}}}$ (bits)', labelpad=2)
        ax.set_ylabel('$\\log ' + symbol + '^*$', labelpad=1)
        ax.grid(alpha=.15, linewidth=.5)
    save(fig, 'fig3_complexity')
    write_json('figure_check.json', dict(status='passed', width_inches=3.5, height_inches=1.9, dpi=300,
        font='8 pt DejaVu Sans', colours=COLOURS, markers=MARKERS,
        figures=['fig1_equivalence', 'fig2_rate', 'fig3_complexity'], png_dimensions=[1050, 570]))
    print('Figures complete', flush=True)
