"""Tool for building and exporting interactive HTML summary tables for training results."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import matplotlib.figure
    from openff.toolkit import ForceField


def _plot_to_base64_img(
    fig: "matplotlib.figure.Figure",
    width: int = 600,
    height: int = 350,
) -> str:
    """Render a matplotlib figure as a base64 ``<img>`` tag and close it."""
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" width="{width}" height="{height}">'


def _smiles_to_labeled_image(
    smiles: str,
    ff: "ForceField",
    handler_key: str = "vdW",
    img_size: tuple[int, int] = (400, 300),
) -> str:
    """Render a molecule as a PNG ``<img>`` tag with force-field atom-type labels."""
    from openff.toolkit import Molecule as OFFMolecule
    from rdkit.Chem import AllChem, Draw

    off_mol = OFFMolecule.from_smiles(smiles, allow_undefined_stereo=True)

    atom_labels: dict[int, str] = {}
    if ff is not None:
        try:
            labels = ff.label_molecules(off_mol.to_topology())[0]
            if handler_key in labels:
                for atom_indices, param in labels[handler_key].items():
                    pid = getattr(param, "id", None)
                    if pid:
                        for idx in atom_indices:
                            atom_labels[idx] = pid
        except Exception:
            pass  # Fall back to unlabeled molecule image

    rdmol = off_mol.to_rdkit()
    AllChem.Compute2DCoords(rdmol)
    for atom in rdmol.GetAtoms():
        label = atom_labels.get(atom.GetIdx(), "")
        if label:
            atom.SetProp("atomNote", label)

    drawer = Draw.MolDraw2DCairo(img_size[0], img_size[1])
    drawer.drawOptions().annotationFontScale = 0.6
    drawer.DrawMolecule(rdmol)
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return (
        f'<img src="data:image/png;base64,{b64}" '
        f'width="{img_size[0]}" height="{img_size[1]}">'
    )


_DEFAULT_COLORS = [
    "red",
    "blue",
    "green",
    "orange",
    "purple",
    "brown",
    "pink",
    "olive",
    "cyan",
]


class TrainingSummary:
    """
    Build and export interactive HTML summary tables for training results.

    Accepts an arbitrary number of evaluation parquet files (produced by
    ``save_prediction_parquet``) via a ``dict[str, path]``.  The
    ``energy_ref`` column from the **first** parquet is used as the
    reference curve; ``energy_pred`` from every parquet is plotted and
    compared.

    Parameters
    ----------
    parquets : dict[str, str | Path]
        Ordered mapping ``{label: path}`` of evaluation parquet files.
        The first entry is taken as the reference for ``energy_ref``.
        Each parquet must have columns ``id``, ``conformer_idx``,
        ``energy_ref``, ``energy_pred``, and optionally ``scale_factor``.
    df : pd.DataFrame | None
        Optional reference dataset with ``"Id"``, ``"Component 1"``,
        ``"Component 2"``, ``"Mole Fraction 1"``, and
        ``"Mole Fraction 2"`` columns.
    reference_label : str
        Display label for the reference (MLP) energy curve.

    Examples
    --------
    >>> ts = TrainingSummary({
    ...     "Trained": "output/final_evaluations.parquet",
    ...     "Initial (perturbed)": "output/initial_evaluations.parquet",
    ...     "OpenFF": "output/openff_evaluations.parquet",
    ... })
    >>> summary_df = ts.build_summary_df()
    >>> ts.save_summary_html(summary_df, "training_summary.html")
    """

    def __init__(
        self,
        parquets: dict[str, str | Path],
        df: pd.DataFrame | None = None,
        reference_label: str = "Reference (MLP)",
    ) -> None:
        if not parquets:
            raise ValueError("parquets must contain at least one entry.")
        self._labels = list(parquets.keys())
        self._dfs: dict[str, pd.DataFrame] = {
            label: pd.read_parquet(path) for label, path in parquets.items()
        }
        self._ref_label = reference_label
        if isinstance(df, (str, Path)):
            p = Path(df)
            if p.suffix == ".csv":
                df = pd.read_csv(p)
            else:
                df = pd.read_parquet(p)
        self._df = df
        # Entry ordering taken from the first parquet.
        first = self._dfs[self._labels[0]]
        self._entry_ids = list(first.groupby("id", sort=False).groups.keys())

    def _extract_sample(
        self, entry_id: str
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """
        Extract energy curves for a single entry.

        Returns
        -------
        x : np.ndarray
            Scale factors (or integer indices if unavailable).
        y_ref : np.ndarray
            Reference energies (from the first parquet).
        curves : dict[str, np.ndarray]
            ``{label: energy_pred}`` for every supplied parquet.
        """
        first_df = self._dfs[self._labels[0]]
        ref_entry = first_df[first_df["id"] == entry_id].sort_values("conformer_idx")

        if "scale_factor" in ref_entry.columns:
            x = ref_entry["scale_factor"].to_numpy()
        else:
            x = np.arange(len(ref_entry), dtype=float)

        y_ref = ref_entry["energy_ref"].to_numpy()

        curves: dict[str, np.ndarray] = {}
        for label, df in self._dfs.items():
            entry = df[df["id"] == entry_id].sort_values("conformer_idx")
            curves[label] = entry["energy_pred"].to_numpy()

        return x, y_ref, curves

    @staticmethod
    def _compute_sample_metrics(
        y_ref: np.ndarray,
        curves: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Compute RMSE, MAE, and offset for each curve."""
        metrics: dict[str, float] = {}
        for label, y in curves.items():
            diff = y - y_ref
            metrics[f"RMSE {label}"] = float(np.sqrt(np.mean(diff**2)))
            metrics[f"MAE {label}"] = float(np.mean(np.abs(diff)))
            metrics[f"Offset {label}"] = float(np.mean(diff))
        return metrics

    def _render_plot(
        self,
        x: np.ndarray,
        y_ref: np.ndarray,
        curves: dict[str, np.ndarray],
        plot_size: tuple[int, int],
        y_lim: tuple[float, float] | None,
    ) -> str:
        """Render an energy-vs-scale-factor plot as a base64 ``<img>`` tag."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(plot_size[0] / 100, plot_size[1] / 100))
        ax.plot(x, y_ref, label=self._ref_label, color="black", linewidth=1.5, zorder=2)
        for idx, (label, y) in enumerate(curves.items()):
            color = _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)]
            ax.plot(x, y, label=label, color=color, linewidth=1.5, zorder=3 + idx)

        ax.set_ylabel("Energy [kcal/mol]")
        ax.set_xlabel("Scale Factor")
        if y_lim is not None:
            ax.set_ylim(y_lim)
        ax.legend(fontsize="small")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        return _plot_to_base64_img(fig, width=plot_size[0], height=plot_size[1])

    def build_summary_df(
        self,
        forcefield_path: str | None = None,
        handler_key: str = "vdW",
        mol_img_size: tuple[int, int] = (400, 300),
        plot_size: tuple[int, int] = (600, 350),
        y_lim: tuple[float, float] | None = (-30, 30),
    ) -> pd.DataFrame:
        """
        Build a summary DataFrame with per-mixture plots and metrics.

        Parameters
        ----------
        forcefield_path : str | None
            If given, renders labeled molecule images using the specified
            OpenFF force field. Set to *None* to skip molecule rendering.
        handler_key : str
            The parameter handler key for atom labels (e.g. ``"vdW"``).
        mol_img_size : tuple[int, int]
            Width and height of molecule images in pixels.
        plot_size : tuple[int, int]
            Width and height of the inline energy plot in pixels.
        y_lim : tuple[float, float] | None
            Y-axis limits for energy plots.  *None* for auto-scaling.

        Returns
        -------
        pd.DataFrame
            Summary table with columns for mixture info, molecule image,
            energy plot, and error metrics.
        """
        ff = None
        if forcefield_path is not None:
            from openff.toolkit import ForceField

            ff = ForceField(forcefield_path, load_plugins=True)
            # Auto-detect handler key from available handlers.
            handler_names = [h._TAGNAME for h in ff._parameter_handlers.values()]
            if handler_key not in handler_names:
                for candidate in ("DoubleExponential", "vdW"):
                    if candidate in handler_names:
                        handler_key = candidate
                        break

        rows: list[dict] = []

        for i, entry_id in enumerate(self._entry_ids):
            x, y_ref, curves = self._extract_sample(entry_id)
            metrics = self._compute_sample_metrics(y_ref, curves)

            # Metadata from the reference DataFrame (if provided).
            comp1 = comp2 = ""
            x1 = x2 = None
            display_id = entry_id
            if self._df is not None and i < len(self._df):
                entry = self._df.iloc[i]
                display_id = entry.get("Id", entry_id)
                comp1 = entry.get("Component 1", "")
                comp2 = entry.get("Component 2", "")
                x1 = entry.get("Mole Fraction 1", None)
                x2 = entry.get("Mole Fraction 2", None)

            # Molecule images
            mol_html = ""
            if ff is not None:
                components = [(comp1, x1, "Comp 1"), (comp2, x2, "Comp 2")]
                images: list[str] = []
                for smi, xfrac, label in components:
                    if pd.notna(smi) and smi:
                        xfrac_str = f" (x={xfrac:.4f})" if pd.notna(xfrac) else ""
                        img_tag = _smiles_to_labeled_image(
                            str(smi),
                            ff,
                            handler_key=handler_key,
                            img_size=mol_img_size,
                        )
                        images.append(f"<b>{label}{xfrac_str}</b><br>{img_tag}")
                mol_html = "<br>".join(images)

            # Mole fractions string
            frac_parts: list[str] = []
            if pd.notna(x1):
                frac_parts.append(f"{x1:.4f}")
            if pd.notna(x2):
                frac_parts.append(f"{x2:.4f}")
            fractions = " / ".join(frac_parts)

            # Inline plot
            plot_html = self._render_plot(
                x, y_ref, curves, plot_size=plot_size, y_lim=y_lim
            )

            row: dict = {
                "Run": f"run_{i:04d}",
                "ID": display_id,
                "Mole Fractions": fractions,
                "Molecule": mol_html,
                "Energy Plot": plot_html,
            }
            row.update(metrics)
            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def save_summary_html(
        summary_df: pd.DataFrame,
        output_file: str | Path = "training_summary.html",
        title: str = "Training Summary",
    ) -> None:
        """
        Save a training summary DataFrame as a scrollable HTML table.

        Parameters
        ----------
        summary_df : pd.DataFrame
            DataFrame produced by :meth:`build_summary_df`.
        output_file : str | Path
            Path of the HTML file to write.
        title : str
            Title shown in the HTML page.
        """
        html_cols = {"Molecule", "Energy Plot"}
        string_cols = {"Run", "ID", "Mole Fractions"}

        # Build table rows with raw HTML for image columns
        header = "".join(f"<th>{col}</th>" for col in summary_df.columns)
        rows_html: list[str] = []
        for _, row in summary_df.iterrows():
            cells: list[str] = []
            for col in summary_df.columns:
                val = row[col]
                if col in html_cols:
                    cells.append(f"<td>{val}</td>")
                elif col in string_cols:
                    cells.append(f"<td>{val}</td>")
                elif pd.notna(val):
                    cells.append(f"<td>{val:.4f}</td>")
                else:
                    cells.append("<td></td>")
            rows_html.append("<tr>" + "".join(cells) + "</tr>")

        page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
  body {{ font-family: sans-serif; margin: 20px; }}
  h1 {{ margin-bottom: 10px; }}
  .table-wrapper {{
    overflow: auto;
    max-height: 90vh;
    border: 1px solid #ccc;
  }}
  table {{ border-collapse: collapse; white-space: nowrap; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; vertical-align: top; }}
  th {{ background: #f5f5f5; position: sticky; top: 0; z-index: 2;
        cursor: pointer; user-select: none; }}
  th:hover {{ background: #e8e8e8; }}
  th .sort-arrow {{ font-size: 0.7em; margin-left: 4px; }}
  tr:hover {{ background: #f9f9f9; }}
  img {{ display: block; }}
</style>
</head><body>
<h1>{title}</h1>
<div class="table-wrapper">
<table id="summary-table">
  <thead><tr>{header}</tr></thead>
  <tbody>{chr(10).join(rows_html)}</tbody>
</table>
</div>
<script>
(function() {{
  var table = document.getElementById('summary-table');
  var headers = table.querySelectorAll('th');
  var sortState = {{}};  // col index -> 'asc' | 'desc'
  headers.forEach(function(th, idx) {{
    th.addEventListener('click', function() {{
      var tbody = table.querySelector('tbody');
      var rows = Array.from(tbody.querySelectorAll('tr'));
      var dir = sortState[idx] === 'asc' ? 'desc' : 'asc';
      sortState = {{}};
      sortState[idx] = dir;
      rows.sort(function(a, b) {{
        var aText = a.children[idx].textContent.trim();
        var bText = b.children[idx].textContent.trim();
        var aNum = parseFloat(aText);
        var bNum = parseFloat(bText);
        if (!isNaN(aNum) && !isNaN(bNum)) {{
          return dir === 'asc' ? aNum - bNum : bNum - aNum;
        }}
        return dir === 'asc'
          ? aText.localeCompare(bText)
          : bText.localeCompare(aText);
      }});
      rows.forEach(function(r) {{ tbody.appendChild(r); }});
      headers.forEach(function(h) {{
        var arrow = h.querySelector('.sort-arrow');
        if (arrow) arrow.remove();
      }});
      var span = document.createElement('span');
      span.className = 'sort-arrow';
      span.textContent = dir === 'asc' ? ' \u25b2' : ' \u25bc';
      th.appendChild(span);
    }});
  }});
}})()
</script>
</body></html>"""
        Path(output_file).write_text(page, encoding="utf-8")
        print(f"Summary saved to {output_file}")
