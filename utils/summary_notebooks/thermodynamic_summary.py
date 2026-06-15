"""Interactive HTML summary tables for thermodynamic benchmark results."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING

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


def _load_df(path: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load a CSV or parquet file into a DataFrame, or pass through a DataFrame."""
    if isinstance(path, pd.DataFrame):
        return path
    p = Path(path)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    return pd.read_parquet(p)


class ThermodynamicSummary:
    """
    Build and export interactive HTML summary tables for thermodynamic benchmarks.

    Accepts an arbitrary number of benchmark result sets via a
    ``dict[str, dict]``.  Each entry maps a display label to a dict
    with ``"density"`` and/or ``"hmix"`` keys pointing to file paths.

    Parameters
    ----------
    benchmarks : dict[str, dict[str, str | Path]]
        Ordered mapping ``{label: {"density": path, "hmix": path}}``.
        Each density file must have columns ``"run"`` and ``"density"``
        (g/mL).  Each hmix file must have columns ``"run"`` and
        ``"hvap"`` (kcal/mol — converted internally to kJ/mol).
        Either key may be omitted if that property is unavailable.
    df : pd.DataFrame | str | Path
        Reference dataset with columns ``"Id"``, ``"Component 1"``,
        ``"Component 2"``, ``"Mole Fraction 1"``, ``"Mole Fraction 2"``,
        ``"Density Value (g / ml)"``, and
        ``"EnthalpyOfMixing Value (kJ / mol)"``.
        Accepts a file path (CSV or parquet) or a DataFrame.

    Examples
    --------
    >>> ts = ThermodynamicSummary(
    ...     {
    ...         "Trained": {"density": "trained/density.csv", "hmix": "trained/hmix.csv"},
    ...         "OpenFF":  {"density": "openff/density.csv",  "hmix": "openff/hmix.csv"},
    ...     },
    ...     df="sage-training-set.csv",
    ... )
    >>> summary_df = ts.build_summary_df()
    >>> ts.save_summary_html(summary_df, "thermo_summary.html")
    """

    def __init__(
        self,
        benchmarks: dict[str, dict[str, str | Path]],
        df: pd.DataFrame | str | Path,
    ) -> None:
        if not benchmarks:
            raise ValueError("benchmarks must contain at least one entry.")
        self._labels = list(benchmarks.keys())

        # Load density / hmix DataFrames for each label.
        self._density: dict[str, pd.DataFrame] = {}
        self._hmix: dict[str, pd.DataFrame] = {}
        for label, paths in benchmarks.items():
            if "density" in paths:
                self._density[label] = _load_df(paths["density"])
            if "hmix" in paths:
                self._hmix[label] = _load_df(paths["hmix"])

        # Reference dataset.
        if isinstance(df, (str, Path)):
            df = _load_df(df)
        self._df = df

    def _prepare_property(
        self,
        df_pred: pd.DataFrame,
        pred_col: str,
        ref_col: str,
        scale: float = 1.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Align and filter predictions against reference data.

        Returns ``(run_indices, y_true, y_pred)`` with NaN rows removed.
        """
        run_idx = df_pred["run"].astype(int)
        y_pred = df_pred[pred_col].reset_index(drop=True) * scale
        y_true = self._df.loc[run_idx, ref_col].reset_index(drop=True)

        mask = ~y_true.isna() & ~y_pred.isna()
        run_idx = run_idx.reset_index(drop=True)[mask].reset_index(drop=True)
        y_true = y_true[mask].reset_index(drop=True)
        y_pred = y_pred[mask].reset_index(drop=True)
        return run_idx, y_true, y_pred

    def build_summary_df(
        self,
        forcefield_path: str | None = None,
        handler_key: str = "vdW",
        mol_img_size: tuple[int, int] = (400, 300),
    ) -> pd.DataFrame:
        """
        Build a summary DataFrame with molecule images and property data.

        Parameters
        ----------
        forcefield_path : str | None
            If given, renders labeled molecule images using the specified
            OpenFF force field.  Set to *None* to skip molecule rendering.
        handler_key : str
            The parameter handler key for atom labels (e.g. ``"vdW"``).
        mol_img_size : tuple[int, int]
            Width and height of molecule images in pixels.

        Returns
        -------
        pd.DataFrame
            Summary table with columns for mixture info, molecule image,
            and per-label density/hmix predictions and errors.
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

        # Prepare density and hmix for each label.
        density_data: dict[str, tuple[pd.Series, pd.Series, pd.Series]] = {}
        hmix_data: dict[str, tuple[pd.Series, pd.Series, pd.Series]] = {}
        all_runs: set[int] = set()

        for label in self._labels:
            if label in self._density:
                run_idx, y_true, y_pred = self._prepare_property(
                    self._density[label], "density", "Density Value (g / ml)"
                )
                density_data[label] = (run_idx, y_true, y_pred)
                all_runs.update(run_idx.tolist())
            if label in self._hmix:
                run_idx, y_true, y_pred = self._prepare_property(
                    self._hmix[label],
                    "hvap",
                    "EnthalpyOfMixing Value (kJ / mol)",
                    scale=4.184,  # kcal -> kJ
                )
                hmix_data[label] = (run_idx, y_true, y_pred)
                all_runs.update(run_idx.tolist())

        rows: list[dict] = []
        for run in sorted(all_runs):
            entry = self._df.iloc[run]
            entry_id = entry.get("Id", run)
            comp1 = entry.get("Component 1", "")
            comp2 = entry.get("Component 2", "")
            x1 = entry.get("Mole Fraction 1", None)
            x2 = entry.get("Mole Fraction 2", None)

            # Molecule images
            mol_html = ""
            if ff is not None:
                components = [(comp1, x1, "Comp 1"), (comp2, x2, "Comp 2")]
                images: list[str] = []
                for smi, xfrac, lbl in components:
                    if pd.notna(smi) and smi:
                        xfrac_str = f" (x={xfrac:.4f})" if pd.notna(xfrac) else ""
                        img_tag = _smiles_to_labeled_image(
                            str(smi), ff, handler_key=handler_key, img_size=mol_img_size
                        )
                        images.append(f"<b>{lbl}{xfrac_str}</b><br>{img_tag}")
                mol_html = "<br>".join(images)

            # Mole fractions string
            frac_parts: list[str] = []
            if pd.notna(x1):
                frac_parts.append(f"{x1:.4f}")
            if pd.notna(x2):
                frac_parts.append(f"{x2:.4f}")

            row: dict = {
                "Run": f"run_{run:04d}",
                "ID": entry_id,
                "Mole Fractions": " / ".join(frac_parts),
                "Molecule": mol_html,
            }

            # Density columns -- experimental (once) + per-label pred & error
            dens_true_val = None
            for label in self._labels:
                if label in density_data:
                    run_idx, y_true, y_pred = density_data[label]
                    mask = run_idx == run
                    if mask.any():
                        pred = float(y_pred[mask].iloc[0])
                        true = float(y_true[mask].iloc[0])
                        if dens_true_val is None:
                            dens_true_val = true
                        row[f"Density {label} (g/mL)"] = pred
                        row[f"Density Err {label} (g/mL)"] = abs(pred - true)
                    else:
                        row[f"Density {label} (g/mL)"] = None
                        row[f"Density Err {label} (g/mL)"] = None
            if density_data:
                row["Density Exp (g/mL)"] = dens_true_val

            # Hmix columns -- experimental (once) + per-label pred & error
            hmix_true_val = None
            for label in self._labels:
                if label in hmix_data:
                    run_idx, y_true, y_pred = hmix_data[label]
                    mask = run_idx == run
                    if mask.any():
                        pred = float(y_pred[mask].iloc[0])
                        true = float(y_true[mask].iloc[0])
                        if hmix_true_val is None:
                            hmix_true_val = true
                        row[f"Hmix {label} (kJ/mol)"] = pred
                        row[f"Hmix Err {label} (kJ/mol)"] = abs(pred - true)
                    else:
                        row[f"Hmix {label} (kJ/mol)"] = None
                        row[f"Hmix Err {label} (kJ/mol)"] = None
            if hmix_data:
                row["Hmix Exp (kJ/mol)"] = hmix_true_val

            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def save_summary_html(
        summary_df: pd.DataFrame,
        output_file: str | Path = "thermo_summary.html",
        title: str = "Thermodynamic Summary",
    ) -> None:
        """
        Save a summary DataFrame as a scrollable HTML table.

        Parameters
        ----------
        summary_df : pd.DataFrame
            DataFrame produced by :meth:`build_summary_df`.
        output_file : str | Path
            Path of the HTML file to write.
        title : str
            Title shown in the HTML page.
        """
        html_cols = {"Molecule"}
        string_cols = {"Run", "ID", "Mole Fractions"}

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
