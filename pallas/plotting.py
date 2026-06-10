"""Pathway plotting: enthalpy profile along the discovered path."""


def profile_png(names, enthalpies, spgs, outfile, title=None):
    """Write an enthalpy-profile figure for a pathway.

    Parameters
    ----------
    names : list of str — node labels along the path (M*/S*).
    enthalpies : list of float — enthalpy of each node (eV, relative to A).
    spgs : list of str — space-group label per node ('' to skip annotation).
    outfile : str — output PNG path.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = list(range(len(names)))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, enthalpies, '-', color='0.55', lw=1.5, zorder=1)

    for i, (name, h) in enumerate(zip(names, enthalpies)):
        if name.startswith('M'):
            ax.plot(i, h, 'o', ms=9, color='#1f6fb4', zorder=3)
        else:
            ax.plot(i, h, '^', ms=8, color='#d1495b', zorder=3)
        label = spgs[i] if i < len(spgs) and spgs[i] else ''
        if label and name.startswith('M'):
            ax.annotate(label, (i, h), textcoords='offset points',
                        xytext=(0, -16), ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('Enthalpy (eV)')
    ax.set_xlabel('Pathway node')
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    return outfile
