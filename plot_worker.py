import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path

def safe_filename(s: str) -> str:
    return s.replace(':', '-').replace('T', '_').replace('/', '_')

def main():
    # args: pt, visit_start, visit_end, out_png, out_pdf, out_pickle
    if len(sys.argv) < 7:
        print('Usage: plot_worker.py <pt> <visit_start> <visit_end> <out_png> <out_pdf> <out_pickle>')
        sys.exit(2)

    pt = sys.argv[1]
    visit_start = sys.argv[2]
    visit_end = sys.argv[3]
    out_png = sys.argv[4]
    out_pdf = sys.argv[5]
    out_pickle = sys.argv[6]

    # Import here to avoid heavy imports in parent process
    try:
        import data_viz_plot as dv
    except Exception as e:
        print('Failed to import data_viz_plot:', e)
        sys.exit(3)

    try:
        fig, ax = plt.subplots(figsize=(15, 8), constrained_layout=True)
        ax = dv.main(pt, visit_start, visit_end, ax)
        # Save outputs: PNG for quick preview, PDF for upload, and pickle the Figure
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        fig.savefig(out_pdf, bbox_inches='tight')
        import pickle
        with open(out_pickle, 'wb') as f:
            pickle.dump(fig, f)
    except Exception as e:
        print('Plotting error:', e)
        sys.exit(4)

if __name__ == '__main__':
    main()
