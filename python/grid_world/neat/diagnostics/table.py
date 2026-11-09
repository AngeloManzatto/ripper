"""
Created on Thu Oct 22 21:36:41 2026

@author: Angelo Antonio Manzatto

"""

###############################################################################
# libraries
###############################################################################

def generation_table(history):
    """One line per generation: fitness, species, genome size, innovations."""
    head = (f"{'gen':>4} {'best':>8} {'mean':>8} {'best_ever':>9} {'spec':>4} "
            f"{'hid(avg/max)':>12} {'conn(avg/max)':>13} {'enab%':>6} {'innov':>6}")
    lines = [head, "-" * len(head)]
    for r in history:
        pct = 100.0 * r["mean_enabled"] / r["mean_connections"] if r["mean_connections"] else 0.0
        lines.append(
            f"{r['generation']:>4} {r['best_fitness']:>8.4f} {r['mean_fitness']:>8.4f} "
            f"{r['best_ever_fitness']:>9.4f} {r['n_species']:>4} "
            f"{r['mean_hidden']:>6.2f}/{r['max_hidden']:<5} "
            f"{r['mean_connections']:>7.2f}/{r['max_connections']:<5} "
            f"{pct:>6.1f} {r['next_innovation']:>6}")
    return "\n".join(lines)


def species_table(record):
    """One line per species of a single generation's record."""
    removed = record.get("removed_species", [])
    note = f" (removed for stagnation: {', '.join(str(i) for i in removed)})" if removed else ""
    head = f"generation {record['generation']}{note}\n{'id':>4} {'size':>5} {'best':>8} {'mean':>8} {'stuck':>6}"
    lines = [head, "-" * 36]
    for s in record["species"]:
        stuck = "-" if s["since_improved"] is None else str(s["since_improved"])
        lines.append(f"{s['id']:>4} {s['size']:>5} {s['best']:>8.4f} {s['mean']:>8.4f} {stuck:>6}")
    return "\n".join(lines)