"""
Created on Fri Oct 23 07:20:04 2026

@author: Angelo Antonio Manzatto

"""

###############################################################################
# libraries
###############################################################################

import matplotlib.pyplot as plt

###############################################################################
# Constants
###############################################################################

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e2de"
DISABLED = "#b4b3ad"
POSITIVE = "#2a78d6"
NEGATIVE = "#eb6834"
NODE_FILL = {"input": "#e3e2de", "hidden": "#9cc3f0", "output": "#f6b79f"}


def node_depths(genome):
    """Depth of every node: inputs 0, hidden at least 1 and one past its deepest enabled source, outputs one past the deepest hidden.

    Raises ValueError if enabled connections form a cycle.
    """
    nodes = genome["nodes"]
    depth = {nid: (1 if node["type"] == "hidden" else 0) for nid, node in nodes.items()}
    enabled = [c for c in genome["connections"].values() if c["enabled"]]
    for _ in range(len(nodes) + 1):
        changed = False
        for c in enabled:
            if nodes[c["dst"]]["type"] == "output":
                continue
            new = depth[c["src"]] + 1
            if new > depth[c["dst"]]:
                depth[c["dst"]] = new
                changed = True
        if not changed:
            break
    else:
        raise ValueError("enabled connections contain a cycle")
    last = max([d for nid, d in depth.items() if nodes[nid]["type"] == "hidden"], default=0) + 1
    for nid, node in nodes.items():
        if node["type"] == "output":
            depth[nid] = last
    return depth


def node_layout(genome):
    """{node id: (x, y)} with x in [0, 1] by depth (inputs left, outputs right), y in [0, 1] spread inside a column."""
    depth = node_depths(genome)
    last = max(depth.values())
    columns = {}
    for nid in sorted(genome["nodes"]):
        columns.setdefault(depth[nid], []).append(nid)
    layout = {}
    for d, ids in columns.items():
        for k, nid in enumerate(ids):
            layout[nid] = (d / last, (k + 1) / (len(ids) + 1))
    return layout


def genome_table(genome):
    """Two text tables: every node, then every connection (disabled ones included)."""
    lines = [f"{'node':>5} {'type':>7} {'bias':>8} activation", "-" * 34]
    for nid in sorted(genome["nodes"]):
        n = genome["nodes"][nid]
        if n["type"] == "input":
            lines.append(f"{nid:>5} {'input':>7}")
        else:
            lines.append(f"{nid:>5} {n['type']:>7} {n['bias']:>8.3f} {n['activation']}")
    lines += ["", f"{'innov':>5} {'src':>4} {'dst':>4} {'weight':>8} enabled", "-" * 34]
    for innov in sorted(genome["connections"]):
        c = genome["connections"][innov]
        lines.append(f"{innov:>5} {c['src']:>4} {c['dst']:>4} {c['weight']:>8.3f} {'yes' if c['enabled'] else 'no'}")
    return "\n".join(lines)


def draw_genome(genome, ax=None, title=None):
    """Draw the network. Blue = positive weight, orange = negative, thickness = |weight|, dashed grey = disabled."""
    layout = node_layout(genome)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    for innov in sorted(genome["connections"]):
        c = genome["connections"][innov]
        (x0, y0), (x1, y1) = layout[c["src"]], layout[c["dst"]]
        if c["enabled"]:
            color = POSITIVE if c["weight"] >= 0 else NEGATIVE
            style = dict(color=color, linewidth=0.6 + 1.6 * min(abs(c["weight"]), 3.0), linestyle="-")
        else:
            style = dict(color=DISABLED, linewidth=1.0, linestyle="--")
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                    arrowprops=dict(arrowstyle="-|>", shrinkA=11, shrinkB=11, mutation_scale=9, **style))
    for nid, (x, y) in layout.items():
        node = genome["nodes"][nid]
        ax.scatter([x], [y], s=420, color=NODE_FILL[node["type"]], edgecolor=INK_SOFT, linewidth=1.0, zorder=3)
        ax.text(x, y, str(nid), ha="center", va="center", fontsize=9, color=INK, zorder=4)
        if node["type"] != "input":
            ax.text(x, y - 0.045, node["activation"], ha="center", va="top", fontsize=7, color=INK_SOFT, zorder=4)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.05, 1.05)
    enabled = sum(1 for c in genome["connections"].values() if c["enabled"])
    ax.set_title(title or f"Genome: {len(genome['nodes'])} nodes, {enabled} of {len(genome['connections'])} connections enabled",
                 loc="left", fontsize=11, color=INK)
    return ax