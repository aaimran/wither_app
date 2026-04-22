#!/usr/bin/env python3
"""Dash app to browse WaveQLab3D station time series."""

import argparse
import glob
import hashlib
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, NamedTuple

try:
    from typing import TypedDict
except ImportError:
    try:
        from typing_extensions import TypedDict
    except ImportError:
        TypedDict = dict

import hashlib
import plotly.graph_objects as go
from plotly.colors import qualitative as plotly_qual
from plotly.subplots import make_subplots
from dash import ALL, Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# --- 1. SETUP & HELPER CLASSES ---

class TimeSeries(TypedDict):
    t: List[float]
    vx: List[float]
    vy: List[float]
    vz: List[float]

# Try to import ctx
try:
    from dash import ctx
except ImportError:
    ctx = None

# Regex patterns
FNAME_RE = re.compile(
    r"^(?P<dataset>.+?)_(?P<q>[^_]+)_(?P<r>[^_]+)_(?P<s>[^_]+)_(?P<block>block[^.]+)\.dat$",
    flags=re.IGNORECASE,
)
STATION_NAME_RE = re.compile(
    r"^(?P<dataset>.+?)_station_(?P<station>[A-Za-z0-9]+)\.dat$",
    flags=re.IGNORECASE,
)
NUM_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?"
XYZ_RE = re.compile(
    rf"^(?P<dataset>.+?)_(?P<x>{NUM_RE})_(?P<y>{NUM_RE})_(?P<z>{NUM_RE})\.dat$",
    flags=re.IGNORECASE,
)

class DatasetInfo(NamedTuple):
    path: Path
    dataset: str
    station: str

VARIANT_ELASTIC = "elastic"
VARIANT_ANELASTIC = "anelastic"

PML_TOKEN_RE = re.compile(r"^pml-(?P<mode>[^_]+)$", flags=re.IGNORECASE)
RES_TOKEN_RE = re.compile(r"^res-(?P<value>[^_]+)$", flags=re.IGNORECASE)
TEST_TOKEN_RE = re.compile(r"(?:^|[_-])test[-_]?(?P<id>\d+[a-z0-9]*)", flags=re.IGNORECASE)
CG_VALUE_RE = re.compile(rf"(?:^|[_-])cg-(?P<val>{NUM_RE})(?:[_-]|$)", flags=re.IGNORECASE)
ANELASTIC_GAMMA_RE = re.compile(r"anelastic[-_]gamma-(?P<val>[+-]?(?:\d+(?:\.\d+)?|\.\d+))", flags=re.IGNORECASE)
ANELASTIC_RE = re.compile(r"(?:^|[_-])anelastic(?:[_-]|$)", flags=re.IGNORECASE)
ELASTIC_RE = re.compile(r"(?:^|[_-])elastic(?:[_-]|$)", flags=re.IGNORECASE)
STENCIL_SET = {"traditional", "upwind", "upwind-drp"}

# --- 2. HELPER FUNCTIONS ---

def dataset_base_and_variant(dataset_name: str) -> Tuple[str, Optional[str]]:
    name = dataset_name
    if name.endswith('.dat'):
        name = name[:-4]

    # If filename ends with three numeric components (x_y_z) treat those as station coords
    parts = name.split('_')
    if len(parts) > 3 and all(re.match(r'^-?\d+(\.\d+)?$', p) for p in parts[-3:]):
        name = '_'.join(parts[:-3])

    # Try to pull out explicit variant tokens (anelastic/elastic) so callers get a clean base
    # Prefer the more specific anelastic-gamma token, falling back to plain 'elastic'
    m = ANELASTIC_GAMMA_RE.search(name)
    if m:
        # remove the matched token from the base name but keep separators
        base = ANELASTIC_GAMMA_RE.sub('_', name)
        base = re.sub(r'_+', '_', base).strip('_')
        return base, m.group('val')
    if ANELASTIC_RE.search(name):
        base = ANELASTIC_RE.sub('_', name)
        base = re.sub(r'_+', '_', base).strip('_')
        return base, 'anelastic'
    if ELASTIC_RE.search(name):
        base = ELASTIC_RE.sub('_', name)
        base = re.sub(r'_+', '_', base).strip('_')
        return base, 'elastic'

    return name, None


def parse_stencil_order_pml_ver(base: str) -> Optional[Tuple[str, str, str, str, str]]:
    parts = [p for p in base.split('_') if p]
    if len(parts) < 2:
        return None

    stencil = ""
    order = ""
    idx = -1
    for i, p in enumerate(parts):
        low = p.lower()
        if low in STENCIL_SET:
            # stencil and order may be separate tokens (stencil, order)
            if i + 1 < len(parts):
                stencil = p
                order = parts[i + 1]
                idx = i + 2
                break
            else:
                return None
        # handle hyphenated stencil-order like 'traditional-6' or 'upwind-drp-6'
        m_st_h = re.match(r'(?P<st>traditional|upwind|upwind-drp)[-_](?P<ord>[0-9]+)$', p, flags=re.IGNORECASE)
        if m_st_h:
            stencil = m_st_h.group('st')
            order = m_st_h.group('ord')
            idx = i + 1
            break
    if not stencil or not order or idx < 0:
        return None

    res_value = ""
    pml_mode: Optional[str] = None
    rest: List[str] = []

    for p in parts[idx:]:
        if pml_mode is None:
            mres = RES_TOKEN_RE.match(p)
            if mres and not res_value:
                res_value = mres.group('value')
                continue
            m = PML_TOKEN_RE.match(p)
            if m:
                pml_mode = m.group('mode').lower()
                continue
        else:
            rest.append(p)

    if not pml_mode:
        return None

    if not rest:
        ver = ""
    elif rest[0].lower() == 'b' and len(rest) >= 2:
        ver = '_'.join(rest[1:])
    else:
        ver = '_'.join(rest)

    return stencil, order, res_value, pml_mode, ver

def parse_test_id(dataset: str) -> str:
    # If an explicit test token exists, keep existing behavior
    m = TEST_TOKEN_RE.search(dataset)
    if m:
        return m.group('id')

    # Otherwise synthesize a compact test id from dataset components.
    # Examples:
    #  withers_anelastic-Q4_ver-a_traditional-6_res-... -> q4t6p0a
    #  withers_elastic_ver-x_traditional-6_res-... -> e0t6p0x
    try:
        # Determine variant (anelastic/elastic) and possible Q/gamma
        variant, gamma = parse_variant_gamma(dataset)

        if variant == VARIANT_ANELASTIC:
            # Prefer the newer 'Qf' token (mapped to 'fN.val' id), then 'Qc'->'cN',
            # otherwise fall back to legacy 'QN' -> 'qN'.
            # find Qf tokens anywhere in the dataset string
            m_qf = re.search(r"Qf(?P<n>\d+)(?:[-_](?P<val>[0-9]+(?:\.[0-9]+)?))?", dataset, flags=re.IGNORECASE)
            if m_qf:
                qn = m_qf.group('n')
                qval = m_qf.group('val')
                try:
                    qvf = float(qval) if qval is not None else 0.0
                except Exception:
                    qvf = 0.0
                # Normalize F-style variants explicitly so F4.0 never drifts to f46.
                qval_norm = str(qval).strip() if qval is not None else ""
                if qval_norm in {"", "0", "0.0", "4.0"} or abs(qvf - 0.0) < 1e-6 or abs(qvf - 4.0) < 1e-6:
                    prefix = f"f{qn}0"
                elif qval_norm in {"0.6", "4.6"} or abs(qvf - 0.6) < 1e-3 or abs(qvf - 4.6) < 1e-3:
                    prefix = f"f{qn}6"
                else:
                    # fallback: keep one decimal place semantics without the dot
                    prefix = f"f{qn}{int(round(qvf * 10))}"
            else:
                m_qc = re.search(r"anelastic[-_]?Qc(?P<n>\d+)", dataset, flags=re.IGNORECASE)
                if m_qc:
                    qn = m_qc.group('n')
                    prefix = f"c{qn}"
                else:
                    m_q = re.search(r"anelastic[-_]?Q(?P<n>\d+)", dataset, flags=re.IGNORECASE)
                    qn = m_q.group('n') if m_q else '4'
                    prefix = f"q{qn}"
        elif variant == VARIANT_ELASTIC:
            prefix = "e0"
        else:
            prefix = "u0"

        # Use base name (with variant tokens removed) to extract stencil/order/pml/ver
        base, _ = dataset_base_and_variant(dataset)
        parsed = parse_stencil_order_pml_ver(base) or ("", "", "", "", "")
        stencil, order, res_value, pml_mode, ver = parsed

        # Stencil abbreviation
        abbr = {'traditional': 't', 'upwind': 'u', 'upwind-drp': 'd'}.get(stencil.lower() if stencil else '', 't')
        order_part = order if order else ''

        # Extract leading number from pml_mode (e.g., '0km' -> '0', '2km' -> '2')
        pnum = ''
        if pml_mode:
            m_p = re.search(r"([0-9]+)", pml_mode)
            pnum = m_p.group(1) if m_p else pml_mode
        else:
            pnum = '0'

        # Normalise ver: accept 'ver-a' or 'a' -> 'a'
        ver_char = ''
        if ver:
            mv = re.search(r"ver[-_]?([A-Za-z0-9]+)", ver, flags=re.IGNORECASE)
            ver_char = mv.group(1) if mv else ver.split('_')[-1]
        else:
            # also try to extract ver token directly from dataset
            mv2 = re.search(r"ver[-_]?([A-Za-z0-9]+)", dataset, flags=re.IGNORECASE)
            ver_char = mv2.group(1) if mv2 else ''

        # Build id pieces
        parts = [prefix]
        if abbr:
            parts.append(f"{abbr}{order_part}")
        parts.append(f"p{pnum}")
        if ver_char:
            parts.append(ver_char)

        return ''.join(parts)
    except Exception:
        return ""

def parse_cg_value(dataset: str) -> str:
    m = CG_VALUE_RE.search(dataset)
    return m.group('val') if m else ""

def selection_to_test_prefix(selection: str) -> str:
    sel = str(selection or "").strip()
    upper = sel.upper()
    if upper in {"C4", "Q4"}:
        return "c4"
    if upper in {"C8", "Q8"}:
        return "c8"
    if upper == "F4.0":
        return "f40"
    if upper == "F4.6":
        return "f46"
    return upper.lower()

def test_id_sort_key(test_id: str) -> Tuple[int, str]:
    tid = str(test_id or "").lower()
    if tid.startswith("e0"):
        return (0, tid)
    if tid.startswith("c4"):
        return (1, tid)
    if tid.startswith("c8"):
        return (2, tid)
    if tid.startswith("f40"):
        return (3, tid)
    if tid.startswith("f46"):
        return (4, tid)
    return (99, tid)

def parse_variant_gamma(dataset: str) -> Tuple[Optional[str], Optional[str]]:
    m = ANELASTIC_GAMMA_RE.search(dataset)
    if m:
        return VARIANT_ANELASTIC, m.group('val')
    if ANELASTIC_RE.search(dataset):
        return VARIANT_ANELASTIC, ""
    if ELASTIC_RE.search(dataset):
        return VARIANT_ELASTIC, "0.0"
    return None, None

def pml_label(pml_mode: str) -> str:
    mode = (pml_mode or "").lower()
    if mode == "off":
        return "0.0"
    if mode == "on":
        return "3.0"
    if mode == "60":
        return "6.0"
    # Try to extract leading numeric value (strip units like 'km' or 'm')
    m = re.match(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", mode)
    if m:
        return m.group(1)
    return pml_mode

def iter_dataset_files(data_dirs: Iterable[Path]) -> List[Path]:
    patterns = ["*.dat"]
    files: List[Path] = []
    for data_root in data_dirs:
        for pat in patterns:
            files.extend(Path(p).resolve() for p in glob.glob(str(Path(data_root) / pat)))
    seen_paths: Set[Path] = set()
    unique_files: List[Path] = []
    for path in files:
        if path.is_file() and path not in seen_paths:
            seen_paths.add(path)
            unique_files.append(path)
    unique_files.sort(key=lambda p: p.name)
    return unique_files

def parse_dataset_info(path: Path) -> Optional[DatasetInfo]:
    if path.suffix.lower() != ".dat":
        return None
    stem = path.stem
    parts = stem.split('_')
    if len(parts) >= 3 and all(re.match(r'^-?\d+(?:\.\d+)?$', p) for p in parts[-3:]):
        station = f"{parts[-3]}_{parts[-2]}_{parts[-1]}"
        dataset = '_'.join(parts[:-3])
        return DatasetInfo(path=path, dataset=dataset, station=station)

    m2 = STATION_NAME_RE.match(path.name)
    if m2:
        dataset = m2.group("dataset")
        station = f"station_{m2.group('station')}"
        return DatasetInfo(path=path, dataset=dataset, station=station)

    m3 = XYZ_RE.match(path.name)
    if m3:
        dataset = m3.group("dataset")
        x, y, z = m3.group("x"), m3.group("y"), m3.group("z")
        station = f"{x}_{y}_{z}"
        return DatasetInfo(path=path, dataset=dataset, station=station)

    if len(parts) >= 5:
        dataset = "_".join(parts[:-4])
        q, r, s, block = parts[-4], parts[-3], parts[-2], parts[-1]
        if dataset and block.lower().startswith("block"):
            station = f"{q}_{r}_{s}_{block}"
            return DatasetInfo(path=path, dataset=dataset, station=station)
    return None

def load_timeseries(path: Path) -> TimeSeries:
    t, vx, vy, vz = [], [], [], []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line: continue
            parts = line.split()
            if len(parts) < 4: continue
            try:
                tt, vxx, vyy, vzz = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
            except ValueError: continue
            t.append(tt); vx.append(vxx); vy.append(vyy); vz.append(vzz)
    return {"t": t, "vx": vx, "vy": vy, "vz": vz}

PALETTE = list(plotly_qual.Plotly) or ["#1f77b4"]

def dataset_color(ds_name: str) -> str:
    digest = hashlib.sha256(ds_name.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:4], "big") % len(PALETTE)
    return PALETTE[idx]


def make_figure(station: str, selected: List[DatasetInfo], plot: str, properties: Dict[str, Dict[str, float]], height: int = 400, show_title: bool = False) -> go.Figure:
    plot_meta = {
        "vx": (f"Radial velocity at station {station}", "vx"),
        "vy": (f"Transverse velocity at station {station}", "vy"),
        "vz": (f"Vertical velocity at station {station}", "vz"),
    }
    if plot not in plot_meta:
        fig = go.Figure()
        fig.add_annotation(text="Invalid plot", showarrow=False)
        return fig

    # Use descriptive y-axis labels (RTZ) and plot velocities in cm/s
    y_label_map = {
        "vx": "Radial Velocity (cm/s)",
        "vy": "Transverse Velocity (cm/s)",
        "vz": "Vertical Velocity (cm/s)",
    }

    title, y_col = plot_meta[plot]
    fig = go.Figure()
    for info in selected:
        df = load_timeseries(info.path)
        # Use test id (e.g. '1x', '1y', '1z') as the legend label
        test_id = parse_test_id(info.dataset)
        label = test_id if test_id else info.dataset
        props = properties.get(label, {})
        color = props.get('color') or dataset_color(info.dataset)
        width = props.get('width', 3)  # Thicker default line
        dash = props.get('dash', 'solid')
        # Convert velocities from m/s to cm/s for plotting
        y_vals_cm = [v * 100.0 for v in df[y_col]]
        # Attach full dataset name to trace `meta` and provide a hovertemplate so
        # hovering (line or legend, depending on renderer) shows the full dataset.
        fig.add_trace(go.Scatter(
            x=df["t"], y=y_vals_cm, mode="lines", name=label,
            line=dict(color=color, width=width, dash=dash),
            showlegend=True,
            meta=info.dataset,
            hovertemplate="%{meta}<br>Time: %{x:.3f}s<br>Value: %{y:.3f} cm/s",
        ))

    y_axis_title = y_label_map.get(plot, "particle velocity (cm/s)")
    layout_dict = dict(
        height=height,
        margin=dict(l=60, r=30, t=60, b=60),
        xaxis_title="Time (s)",
        yaxis_title=y_axis_title,
        font=dict(family="Arial, sans-serif", size=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            font=dict(size=18),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            borderwidth=0,
            bgcolor="rgba(0,0,0,0)",
            itemwidth=60
        ),
    )
    if show_title:
        layout_dict["title"] = dict(
            text=title,
            font=dict(size=22, family="Arial, sans-serif", color="#222"),
            x=0.5,
            xanchor="center"
        )
    fig.update_layout(**layout_dict)
    fig.update_xaxes(
        showline=True, linewidth=2, linecolor="black",
        mirror=True, ticks="outside", tickwidth=2, tickcolor="black", ticklen=8,
        gridcolor="#e5e5e5", gridwidth=1,
        zeroline=False,
        title_font=dict(size=22, family="Arial, sans-serif", color="#222"),
        tickfont=dict(size=18, family="Arial, sans-serif", color="#222")
    )
    fig.update_yaxes(
        showline=True, linewidth=2, linecolor="black",
        mirror=True, ticks="outside", tickwidth=2, tickcolor="black", ticklen=8,
        gridcolor="#e5e5e5", gridwidth=1,
        zeroline=False,
        title_font=dict(size=22, family="Arial, sans-serif", color="#222"),
        tickfont=dict(size=18, family="Arial, sans-serif", color="#222")
    )
    return fig

def build_index(data_dirs: Iterable[Path]) -> Tuple[List[DatasetInfo], List[str]]:
    infos: List[DatasetInfo] = []
    for path in iter_dataset_files(data_dirs):
        info = parse_dataset_info(path)
        if info is not None:
            infos.append(info)
    stations = sorted({i.station for i in infos})
    return infos, stations

def group_by_station(infos: Iterable[DatasetInfo]) -> Dict[str, List[DatasetInfo]]:
    out: Dict[str, List[DatasetInfo]] = {}
    for i in infos:
        out.setdefault(i.station, []).append(i)
    for st in out:
        out[st].sort(key=lambda x: x.dataset)
    return out

# --- 3. GLOBAL INITIALIZATION (Executed on Import) ---

# Argument Parsing (handled safely for Gunicorn)
# We use argparse primarily to find the Data Dir, but fallback to environment/CWD
parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default=None, help="Directory containing station .dat files")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8050)
parser.add_argument("--debug", action="store_true")

# In Gunicorn/Production, sys.argv might not be what we expect, so we catch errors
# or rely on defaults.
try:
    args, unknown = parser.parse_known_args()
except SystemExit:
    # If argparse fails (e.g. during build), just use defaults
    args = argparse.Namespace(data_dir=None, host="0.0.0.0", port=8050, debug=False)

# Locate Data Directory
script_dir = Path(__file__).resolve().parent
candidates = [
    Path.cwd() / "data_rtz",
    Path.cwd() / "data",
    Path.cwd() / "waveqlab3d/simulation/plots",
    script_dir / "waveqlab3d/simulation/plots",
    script_dir / "data",
    script_dir,
]
# If --data-dir was passed, prioritize it
if args.data_dir:
    candidates.insert(0, Path(args.data_dir).expanduser().resolve())

data_dir = next((p.resolve() for p in candidates if p.exists()), candidates[-1].resolve())

data_dirs: List[Path] = []
data_dirs.append(data_dir)
for extra_base in (Path.cwd(), script_dir):
    filtered = extra_base / "data_rtz_filtered"
    if filtered.exists():
        filtered_resolved = filtered.resolve()
        if filtered_resolved not in data_dirs:
            data_dirs.append(filtered_resolved)

# Initialize Data Index
all_infos, stations = build_index(data_dirs)
by_station = group_by_station(all_infos)

# --- Available filter values (computed from datasets) ---
available_domains: List[str] = []
available_stencils: List[str] = []
available_pmls: List[str] = []
_doms = set()
_sts = set()
_pmls = set()
for info in all_infos:
    base, _ = dataset_base_and_variant(info.dataset)
    parsed = parse_stencil_order_pml_ver(base) or (None, None, None, None, None)
    st, order, res_v, pml_mode_v, ver_v = parsed
    if st:
        _sts.add(st)
    # collect pml numeric if present
    if pml_mode_v:
        m_p = re.search(r"([0-9]+)", str(pml_mode_v))
        if m_p:
            _pmls.add(m_p.group(1))
        else:
            _pmls.add(str(pml_mode_v))
    # domain / ver token
    dom = ""
    if ver_v:
        mver = re.search(r"ver[-_]?([A-Za-z0-9]+)", str(ver_v), flags=re.IGNORECASE)
        if mver:
            dom = mver.group(1).lower()
        else:
            dom = str(ver_v).split('_')[-1].lower()
    if not dom:
        mver2 = re.search(r"ver[-_]?([A-Za-z0-9]+)", info.dataset, flags=re.IGNORECASE)
        if mver2:
            dom = mver2.group(1).lower()
    if dom:
        _doms.add(dom[0])
available_domains = sorted(_doms)
available_stencils = sorted(_sts)
available_pmls = sorted(_pmls, key=lambda x: int(x) if re.match(r"^\d+$", x) else 999)

# Initialize App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# CRITICAL: Expose server for Gunicorn
server = app.server

# Setup Defaults for Layout
preferred_station = "12.000_0.000_9.000"
if preferred_station in stations:
    initial_station = preferred_station
elif stations:
    initial_station = stations[0]
else:
    initial_station = ""
initial_infos = by_station.get(initial_station, [])
initial_selected_infos = [initial_infos[0]] if initial_infos else []
initial_plots = ["vx", "vy", "vz"]
initial_figs = [make_figure(initial_station, initial_selected_infos, p, {}, 400) for p in initial_plots]

# --- 4. LAYOUT DEFINITION ---

app.layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col(
                html.Div(
                    html.H2("Station viewer: Withers", style={"margin": "0", "padding": "12px 0", "textAlign": "center", "fontWeight": "bold"}),
                    style={
                        "background": "#f8f9fa",
                        "borderBottom": "2px solid #dee2e6",
                        "width": "100%",
                        "position": "fixed",
                        "top": 0,
                        "left": 0,
                        "zIndex": 1000,
                        "height": "60px"
                    }
                ),
                width=12
            )
        ], style={"marginBottom": "72px"}),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Select Depth (y)"),
                        dcc.Dropdown(
                            id="depth-dropdown",
                            options=[{"label": str(y), "value": str(y)} for y in sorted({s.split('_')[1] for s in stations})],
                            value=(stations[0].split('_')[1] if stations else ""),
                            multi=False, clearable=False, placeholder="Choose depth",
                        ),
                        html.Div(id="xz-table-container", style={"marginTop": "8px"}),
                        # keep station-dropdown as the internal selected station value
                        dcc.Store(id="station-dropdown", data=initial_station),
                        html.Hr(),
                        html.Label("Datasets"),
                        html.Button("Clear Dataset Selections", id="clear-dataset-button", style={"marginLeft": "10px"}),
                        # Column selector always visible
                        html.Div([
                            html.Label("Show columns:"),
                            dcc.Checklist(
                                id="dataset-table-column-selector",
                                options=[
                                    {"label": "Test", "value": "test"},
                                    {"label": "Res", "value": "res"},
                                    {"label": "Stencil", "value": "stencil"},
                                    {"label": "Order", "value": "order"},
                                    {"label": "Domain", "value": "domain"},
                                    {"label": "PML", "value": "pml"},
                                    {"label": "Response", "value": "response"},
                                    {"label": "CG", "value": "cg"}
                                ],
                                # Default: Test, Res, Order, CG are unselected
                                value=["stencil", "domain", "pml", "response"],
                                inline=True,
                                inputStyle={"marginRight": "4px"},
                                style={"marginLeft": "8px"}
                            )
                        ,
                        html.Div([
                            html.Label("Filters:"),
                            html.Div([
                                dcc.Dropdown(
                                    id="filter-domain-dropdown",
                                    options=[{"label": d, "value": d} for d in available_domains],
                                    value=[],
                                    multi=True,
                                    placeholder="Domain (ver)",
                                    style={"minWidth": "120px"}
                                ),
                                dcc.Dropdown(
                                    id="filter-stencil-dropdown",
                                    options=[{"label": s, "value": s} for s in available_stencils],
                                    value=[],
                                    multi=True,
                                    placeholder="Stencil",
                                    style={"minWidth": "160px"}
                                ),
                                dcc.Dropdown(
                                    id="filter-pml-dropdown",
                                    options=[{"label": p, "value": p} for p in available_pmls],
                                    value=[],
                                    multi=True,
                                    placeholder="PML",
                                    style={"minWidth": "100px"}
                                ),
                            ], style={"display": "flex", "flexDirection": "row", "gap": "8px", "marginTop": "6px"}),
                        ])
                        ], style={"marginBottom": "8px"}),
                        dcc.Store(id="dataset-path-map", data={}),
                        dcc.Store(id="dataset-base-order", data=[]),
                        dcc.Store(id="dataset-selection-store", data={}),
                        html.Div(id="dataset-table-container"),
                        html.Hr(),
                        html.Hr(),
                        html.Label("Adjust line properties:"),
                        dcc.Store(id="line-properties-store", data={}),
                        html.Div(id="line-controls-container"),
                        html.Hr(),
                        html.Label("Adjust plot height:"),
                        dcc.Slider(id="plot-height-slider", min=200, max=800, value=400, step=50, marks={200:'200', 400:'400', 600:'600', 800:'800'}),
                        html.Div([
                            html.Label("Select timeseries plot(s):"),
                            html.Br(),
                            dcc.Checklist(
                                id="plot-checklist",
                                options=[
                                    {"label": "Radial", "value": "vx"},
                                    {"label": "Transverse", "value": "vy"},
                                    {"label": "Vertical", "value": "vz"}
                                ],
                                value=initial_plots,
                                labelStyle={"display": "inline-block", "marginRight": "20px", "whiteSpace": "nowrap"},
                                style={"display": "flex", "flexDirection": "row", "gap": "10px", "marginTop": "4px"},
                            ),
                            dcc.RadioItems(
                                id="plot-grid-mode",
                                options=[{"label": "Default", "value": "default"}, {"label": "Stacked", "value": "stacked"}],
                                value="default",
                                inline=True,
                                style={"marginTop": "8px", "marginBottom": "4px"},
                            ),
                            dcc.Checklist(
                                id="plot-title-toggle",
                                options=[{"label": "Plot Title", "value": "show_title"}],
                                value=[],
                                style={"marginTop": "8px", "marginLeft": "2px"},
                                inputStyle={"marginRight": "6px"}
                            )
                        ], style={"marginTop": "12px", "marginBottom": "8px"}),
                        html.Hr(),
                    ],
                    style={
                        "flex": "0 0 30%",
                        "maxWidth": "30%",
                        "overflowY": "auto",
                        "maxHeight": "95vh",
                    },
                ),
                dbc.Col(
                    [
                        html.Div(id="plot-container", children=[dcc.Graph(figure=fig) for fig in initial_figs])
                    ],
                    style={
                        "flex": "0 0 70%",
                        "maxWidth": "70%",
                        "overflowY": "auto",
                        "maxHeight": "95vh",
                    },
                ),
            ],
            align="start",
        ),
    ],
    fluid=True,
    className="p-3",
)

# --- 5. CALLBACKS ---

@app.callback(Output("xz-table-container", "children"), [Input("depth-dropdown", "value"), Input("station-dropdown", "data")])
def render_xz_table(depth_value: str, current_station: str):
    # Build a table of X,Z positions available at this depth (y)
    if not depth_value or depth_value is None:
        if stations:
            depth_value = stations[0].split('_')[1]
        else:
            return html.Div("No stations available")
    
    # find stations matching the selected depth
    matches = [s for s in stations if s.split('_')[1] == str(depth_value)]
    if not matches:
        return html.Div(f"No stations at depth {depth_value}")
    # gather unique sorted X and Z values
    xs = sorted({s.split('_')[0] for s in matches}, key=lambda v: float(v))
    zs = sorted({s.split('_')[2] for s in matches}, key=lambda v: float(v))
    stations_set = set(matches)

    # Render all X columns and let the container scroll horizontally
    xs_vis = xs
    header_cells = [html.Th("Z / X", style={"textAlign": "center"})]
    for x in xs_vis:
        try:
            x_float = float(x)
            if x_float.is_integer():
                x_val = f"{int(x_float)}"
            else:
                x_val = f"{x_float:.2f}"
        except Exception:
            x_val = x
        header_cells.append(html.Th(x_val, style={"textAlign": "center"}))
    header = html.Thead(html.Tr(header_cells))

    body_rows = []
    for z in zs:
        try:
            z_float = float(z)
            if z_float.is_integer():
                z_val = f"{int(z_float)}"
            else:
                z_val = f"{z_float:.2f}"
        except Exception:
            z_val = z
        row_cells = [html.Th(z_val, style={"textAlign": "center"})]
        for x in xs_vis:
            coord = f"{x}_{depth_value}_{z}"
            if coord in stations_set:
                # Clickable circle: filled (●) if selected, empty (○) if not
                is_selected = (coord == current_station)
                symbol = "●" if is_selected else "○"
                cell = html.Div(
                    symbol,
                    id={"type": "xz-cell", "coord": coord},
                    n_clicks=0,
                    style={
                        "cursor": "pointer",
                        "fontSize": "14px",
                        "fontWeight": "bold",
                        "userSelect": "none",
                        "textAlign": "center",
                        "margin": "0",
                        "padding": "0",
                        "lineHeight": "1.1",
                        "height": "18px",
                        "width": "18px",
                        "display": "inline-flex",
                        "alignItems": "center",
                        "justifyContent": "center"
                    },
                    title=f"Click to select {coord}"
                )
            else:
                cell = html.Div()
            # Remove horizontal padding for radio button cells, match style to first column
            row_cells.append(html.Td(cell, style={"textAlign": "center"}))
        body_rows.append(html.Tr(row_cells))
    
    table = dbc.Table(
        [header, html.Tbody(body_rows)],
        bordered=True,
        size="sm",
        style={
            "whiteSpace": "nowrap",
            "borderCollapse": "collapse",
            "margin": "0",
            "padding": "0"
        },
        className="table-compact"
    )

    # Wrap table in a horizontally scrollable container so the user can pan across X
    # Add compact cell padding via inline style for all cells
    return html.Div(
        table,
        style={
            "overflowX": "auto",
            "marginTop": "8px",
            "fontSize": "15px"
        }
    )


@app.callback(Output("station-dropdown", "data"), Input({"type": "xz-cell", "coord": ALL}, "n_clicks"), State({"type": "xz-cell", "coord": ALL}, "id"))
def select_station_from_xz(n_clicks_list, ids):
    # Determine which circle was just clicked
    # Determine which circle was just clicked. Dash sometimes gives a prop_id
    # string in JSON or Python-literal form; try several robust parsing methods
    # and fall back to inspecting the n_clicks_list/ids arrays.
    if not n_clicks_list:
        raise PreventUpdate

    # Try to read the triggered prop from the callback context first.
    coord = None
    try:
        triggered = ctx.triggered if ctx else []
    except Exception:
        triggered = []

    if triggered:
        pid = triggered[0].get("prop_id", "")
        if pid and ".n_clicks" in pid:
            prop_id_part = pid.split('.', 1)[0]
            # Try JSON first, then Python literal fallback (ast.literal_eval)
            try:
                import json
                id_dict = json.loads(prop_id_part)
            except Exception:
                try:
                    import ast
                    id_dict = ast.literal_eval(prop_id_part)
                except Exception:
                    id_dict = None
            if isinstance(id_dict, dict):
                coord = id_dict.get("coord")

    # If we couldn't get coord from ctx, fall back to inspecting n_clicks_list
    if not coord:
        # compute max clicks (treat None as 0)
        try:
            max_clicks = max([v or 0 for v in n_clicks_list])
        except ValueError:
            raise PreventUpdate
        if not max_clicks:
            raise PreventUpdate
        # prefer the last index that has the max value (most recently clicked)
        indices = [i for i, v in enumerate(n_clicks_list) if (v or 0) == max_clicks]
        if not indices:
            raise PreventUpdate
        idx = indices[-1]
        if isinstance(ids, (list, tuple)) and idx < len(ids):
            id_obj = ids[idx]
            if isinstance(id_obj, dict):
                coord = id_obj.get("coord")

    if coord:
        return coord

    raise PreventUpdate


@app.callback(
    [Output("dataset-table-container", "children"),
     Output("dataset-path-map", "data"),
     Output("dataset-base-order", "data")],
    [Input("station-dropdown", "data"),
     Input("dataset-selection-store", "data"),
     Input("dataset-table-column-selector", "value"),
     Input("filter-domain-dropdown", "value"),
     Input("filter-stencil-dropdown", "value"),
     Input("filter-pml-dropdown", "value")]
)
def update_dataset_table(selected_station: str, selection_store: Dict, selected_columns=None, domain_filters=None, stencil_filters=None, pml_filters=None):
    infos = by_station.get(selected_station or "", [])
    stencil_to_variants: Dict[str, Dict[Tuple[str, str, str, str], Dict[str, List[str]]]] = {}
    base_to_cg: Dict[str, str] = {}
    base_to_gamma: Dict[str, str] = {}
    base_to_testids: Dict[str, Set[str]] = {}
    
    # Apply optional filters (domain, stencil, pml) to dataset infos
    if (domain_filters and len(domain_filters) > 0) or (stencil_filters and len(stencil_filters) > 0) or (pml_filters and len(pml_filters) > 0):
        filtered = []
        for info in infos:
            base_i, _ = dataset_base_and_variant(info.dataset)
            parsed_i = parse_stencil_order_pml_ver(base_i)
            st_i = pml_i = ver_i = None
            if parsed_i:
                st_i, _, _, pml_i, ver_i = parsed_i
            # domain char from ver or dataset
            dom_i = ''
            if ver_i:
                mv = re.search(r"ver[-_]?([A-Za-z0-9]+)", str(ver_i), flags=re.IGNORECASE)
                dom_i = mv.group(1).lower() if mv else str(ver_i).split('_')[-1].lower()
            else:
                mv2 = re.search(r"ver[-_]?([A-Za-z0-9]+)", info.dataset, flags=re.IGNORECASE)
                dom_i = mv2.group(1).lower() if mv2 else ''
            # pml numeric
            pnum_i = ''
            if pml_i:
                mpp = re.search(r"([0-9]+)", str(pml_i))
                pnum_i = mpp.group(1) if mpp else str(pml_i)

            # check filters
            if domain_filters and len(domain_filters) > 0:
                if not dom_i or dom_i[0] not in [d.lower() for d in domain_filters]:
                    continue
            if stencil_filters and len(stencil_filters) > 0:
                if not st_i or st_i not in stencil_filters:
                    continue
            if pml_filters and len(pml_filters) > 0:
                if not pnum_i or pnum_i not in [str(p) for p in pml_filters]:
                    continue
            filtered.append(info)
        infos = filtered

    for info in infos:
        base, _ = dataset_base_and_variant(info.dataset)
        test_id = parse_test_id(info.dataset)
        variant, gamma = parse_variant_gamma(info.dataset)
        if variant not in (VARIANT_ELASTIC, VARIANT_ANELASTIC):
            continue
        parsed = parse_stencil_order_pml_ver(base)
        if parsed is None:
            continue
        stencil, order, res, pml_mode, ver = parsed
        # Normalize domain (ver) token to a single char like 'a','r','x' when present
        domain_char = ''
        if ver:
            mver = re.search(r"ver[-_]?([A-Za-z0-9]+)", str(ver), flags=re.IGNORECASE)
            domain_char = mver.group(1).lower() if mver else str(ver).split('_')[-1].lower()
        else:
            mver2 = re.search(r"ver[-_]?([A-Za-z0-9]+)", info.dataset, flags=re.IGNORECASE)
            domain_char = mver2.group(1).lower() if mver2 else ''
        # Group datasets by order,res,pml,ver (ignore test id so E/Q variants for same
        # configuration appear on one row). Store a list of paths per variant.
        key = (order, res, pml_mode, domain_char)
        if stencil not in stencil_to_variants:
            stencil_to_variants[stencil] = {}
        if key not in stencil_to_variants[stencil]:
            stencil_to_variants[stencil][key] = {}
        stencil_to_variants[stencil][key].setdefault(variant, []).append(str(info.path))
        # Record which test ids belong to this base (for merged display)
        base_key = f"{stencil}_{order}"
        if res:
            base_key += f"_res-{res}"
        base_key += f"_pml-{pml_mode}"
        if domain_char:
            base_key += f"_ver-{domain_char}"
        if test_id:
            base_to_testids.setdefault(base_key, set()).add(test_id)
        if base_key not in base_to_cg:
            base_to_cg[base_key] = parse_cg_value(info.dataset)
        if variant == VARIANT_ANELASTIC and gamma and base_key not in base_to_gamma:
            base_to_gamma[base_key] = gamma

    sorted_stencils = sorted(stencil_to_variants.keys(), key=lambda s: ['traditional', 'upwind', 'upwind-drp'].index(s) if s in ['traditional', 'upwind', 'upwind-drp'] else 999)
    grouped = {}
    base_order = []

    def test_sort_key(base: str) -> Tuple[int, str]:
        # Extract test number from base string, default to large if not found
        m = re.search(r"test(\d+)([a-z0-9]*)", base)
        if m:
            return (int(m.group(1)), m.group(2) or "")
        return (10**9, "")

    # Collect all (base, key) pairs across all stencils
    all_base_keys = []
    for stencil in sorted_stencils:
        keys = list(stencil_to_variants[stencil].keys())
        for key in keys:
            # key is (order, res, pml_mode, ver)
            order_k, res_k, pml_mode_k, ver_k = key
            # build the same base_key used earlier when populating base_to_cg
            base_key = f"{stencil}_{order_k}"
            if res_k:
                base_key += f"_res-{res_k}"
            base_key += f"_pml-{pml_mode_k}"
            if ver_k:
                base_key += f"_{ver_k}"
            # append cg suffix if we have a recorded cg for this base_key
            cg_val = base_to_cg.get(base_key, "")
            if cg_val:
                base_with_cg = base_key + f"_cg-{cg_val}"
            else:
                base_with_cg = base_key
            all_base_keys.append((base_with_cg, key, stencil))

    # Sort all_base_keys by stencil (traditional, upwind, upwind-drp),
    # then by Domain (ver token ordering a, r, x), then by PML (prefer 0, then 2),
    # and finally by test number and base name for stable ordering.
    def base_sort_key(item):
        base, key, stencil = item
        # key is (order, res, pml_mode, domain_char)
        order_k, res_k, pml_mode_k, domain_k = key
        stencil_map = {"traditional": 0, "upwind": 1, "upwind-drp": 2}
        sidx = stencil_map.get(stencil, 99)

        # Domain ordering: look for ver token first in parsed ver_k, fallback to base
        domain_char = domain_k or ''
        domain_map = {'a': 0, 'r': 1, 'x': 2}
        didx = domain_map.get(domain_char[:1] if domain_char else '', 99)

        # PML ordering: extract leading numeric (0,2 preferred)
        pnum = None
        if pml_mode_k:
            m_pml = re.search(r"([0-9]+)", str(pml_mode_k))
            if m_pml:
                try:
                    pnum = int(m_pml.group(1))
                except Exception:
                    pnum = None
        pmap = {0: 0, 2: 1}
        pidx = pmap.get(pnum if pnum is not None else 0, 99)

        return (sidx, didx, pidx, test_sort_key(base), base)

    all_base_keys_sorted = sorted(all_base_keys, key=base_sort_key)

    for base, key, stencil in all_base_keys_sorted:
        grouped[base] = stencil_to_variants[stencil][key]
        base_order.append(base)

    has_any_selection = False
    for base in base_order:
        saved = selection_store.get(base, {}) if isinstance(selection_store, dict) else {}
        if bool(saved.get(VARIANT_ELASTIC)) or bool(saved.get(VARIANT_ANELASTIC)):
            has_any_selection = True; break

    if not has_any_selection and base_order:
        first_base = base_order[0]
        selection_store = dict(selection_store or {})
        selection_store.setdefault(first_base, {})
        if grouped[first_base].get(VARIANT_ELASTIC):
            # store as non-empty list for consistency with checklist value
            selection_store[first_base][VARIANT_ELASTIC] = ["on"]
        elif grouped[first_base].get(VARIANT_ANELASTIC):
            # pick a sensible default Q (prefer Q4 if present)
            an_paths = grouped[first_base].get(VARIANT_ANELASTIC, [])
            default_q = None
            for p in an_paths:
                try:
                    pi = parse_dataset_info(Path(p))
                    ds = pi.dataset if pi else p
                except Exception:
                    ds = p
                m_q = re.search(r"anelastic[-_]?Q(?P<n>\d+)", ds, flags=re.IGNORECASE)
                if m_q and m_q.group('n') == '4':
                    default_q = 'Q4'
                    break
            if not default_q and an_paths:
                # fallback: try to detect any Q, else pick first
                for p in an_paths:
                    try:
                        pi = parse_dataset_info(Path(p))
                        ds = pi.dataset if pi else p
                    except Exception:
                        ds = p
                    m_qc = re.search(r"anelastic[-_]?Qc(?P<n>\d+)", ds, flags=re.IGNORECASE)
                    m_q = re.search(r"anelastic[-_]?Q(?P<n>\d+)", ds, flags=re.IGNORECASE)
                    if m_qc:
                        default_q = f"C{m_qc.group('n')}"
                        break
                    if m_q:
                        # legacy token found; map to Cn for UI consistency
                        default_q = f"C{m_q.group('n')}"
                        break
            if not default_q and an_paths:
                default_q = None
            selection_store[first_base][VARIANT_ANELASTIC] = [default_q] if default_q else []


    # Column selector
    all_columns = [
        ("Test", "test"),
        ("Res", "res"),
        ("Stencil", "stencil"),
        ("Order", "order"),
        ("Domain", "domain"),
        ("PML", "pml"),
        ("Response", "response"),
        ("CG", "cg"),
    ]
    # Default selected columns (leave Test, Res, Order, CG unselected)
    default_selected = ["stencil", "domain", "pml", "response"]
    if selected_columns is None:
        selected_columns = default_selected
    canonical_column_order = [value for _, value in all_columns]
    selected_set = set(selected_columns)
    selected_columns = [value for value in canonical_column_order if value in selected_set]
    # column_selector is now in the main layout

    stencil_legend = html.Div([
        html.Div([html.Span("Stencil:", style={"fontWeight": "bold"}), html.Span(" t = traditional, u = upwind, d = upwind-drp", style={"fontStyle": "italic", "marginLeft": "6px"})]),
        html.Div([html.Span("Domain:", style={"fontWeight": "bold"}), html.Span(" a = regular, r = reference, x = ?", style={"fontStyle": "italic", "marginLeft": "6px"})])
    ], style={"marginBottom": "6px"})

    # Only include selected columns in header
    col_map = {
        "test": html.Th("Test", id={"type": "col-header", "col": "test"}, style={"textAlign": "center", "verticalAlign": "middle"}),
        "domain": html.Th("Domain", id={"type": "col-header", "col": "domain"}, style={"textAlign": "center", "verticalAlign": "middle"}),
        "stencil": html.Th("Stencil", id={"type": "col-header", "col": "stencil"}, style={"textAlign": "center", "verticalAlign": "middle"}),
        "order": html.Th("Order", id={"type": "col-header", "col": "order"}, style={"textAlign": "center", "verticalAlign": "middle"}),
        "res": html.Th("Res", id={"type": "col-header", "col": "res"}, style={"textAlign": "center", "verticalAlign": "middle"}),
        "pml": html.Th("PML", id={"type": "col-header", "col": "pml"}, style={"textAlign": "center", "verticalAlign": "middle"}),
        "response": html.Th(
            html.Div([
                html.Div("Response", style={"fontWeight": "bold", "textAlign": "center"}),
                html.Div([
                    html.Span("E", style={"paddingRight": "10px"}),
                    html.Span("C4", style={"paddingRight": "10px"}),
                    html.Span("C8", style={"paddingRight": "10px"}),
                    html.Span("F4.0", style={"paddingRight": "10px"}),
                    html.Span("F4.6"),
                ], style={"fontSize": "12px", "textAlign": "center", "marginTop": "4px"})
            ]),
            id={"type": "col-header", "col": "response"},
            style={"textAlign": "center", "verticalAlign": "middle"}
        ),
        "cg": html.Th("CG", id={"type": "col-header", "col": "cg"}, style={"textAlign": "center", "verticalAlign": "middle"}),
    }
    header = html.Thead(html.Tr([col_map[c] for c in selected_columns]))


    # Build table rows directly from base_order. Merge Elastic and Anelastic into one 'Response' column.
    rows = []
    stencil_abbr = {"traditional": "t", "upwind": "u", "upwind-drp": "d"}
    for base in base_order:
        variants = grouped.get(base, {})
        elastic_paths = variants.get(VARIANT_ELASTIC, []) or []
        anelastic_paths = variants.get(VARIANT_ANELASTIC, []) or []
        saved = selection_store.get(base, {}) if isinstance(selection_store, dict) else {}

        # Elastic checkbox (no label)
        elastic_check = dcc.Checklist(
            id={"type": "dataset-elastic", "base": base},
            options=[{"label": "", "value": "on", "disabled": not bool(elastic_paths)}],
            value=["on"] if bool(saved.get(VARIANT_ELASTIC)) and bool(elastic_paths) else [],
            inputStyle={"marginRight": "6px"},
            labelStyle={"display": "inline-block"}
        )

        # Prepare a single checklist with available Q variants (Q4/Q8)
        has_q4 = False
        has_q8 = False
        has_f40 = False
        has_f46 = False
        for p in anelastic_paths:
            try:
                info_a = parse_dataset_info(Path(p))
                ds = info_a.dataset if info_a else p
            except Exception:
                ds = p
            # Detect either new 'QcN' token or legacy 'QN' and treat both as Cn
            m_qc = re.search(r"anelastic[-_]?Qc(?P<n>\d+)", ds, flags=re.IGNORECASE)
            m_q = re.search(r"anelastic[-_]?Q(?P<n>\d+)", ds, flags=re.IGNORECASE)
            # Detect 'Qf' flavour tokens like 'Qf4-0.0', 'Qf4-0.6' or 'Qf4_4.6'
            # detect Qf anywhere in the dataset token (not necessarily after 'anelastic')
            m_qf = re.search(r"Qf(?P<n>\d+)(?:[-_](?P<val>[0-9]+(?:\.[0-9]+)?))?", ds, flags=re.IGNORECASE)
            if m_qc:
                if m_qc.group('n') == '4':
                    has_q4 = True
                if m_qc.group('n') == '8':
                    has_q8 = True
            elif m_q:
                if m_q.group('n') == '4':
                    has_q4 = True
                if m_q.group('n') == '8':
                    has_q8 = True
            if m_qf:
                n = m_qf.group('n')
                val = m_qf.group('val') or ''
                try:
                    v = float(val) if val else None
                except Exception:
                    v = None
                # Map numeric Qf variants to F placeholders:
                # - values near 0.0 => F4.0
                # - values near 0.6 or 4.6 => F4.6 (accept 0.6 as equivalent)
                if n == '4':
                    if v is None:
                        has_f40 = True
                    else:
                        # Map numeric suffix to single-digit tail: 0.0 -> 0, 0.6/4.6 -> 6
                        if abs(v - 0.0) < 1e-6:
                            has_f40 = True
                        if abs(v - 0.6) < 1e-3 or abs(v - 4.6) < 1e-3:
                            has_f46 = True

        # Always show placeholders for Q4 and Q8; disable if not present
        q_options = [
            {"label": "", "value": "C4", "disabled": not has_q4},
            {"label": "", "value": "C8", "disabled": not has_q8},
            {"label": "", "value": "F4.0", "disabled": not has_f40},
            {"label": "", "value": "F4.6", "disabled": not has_f46},
        ]

        # selection_store may contain a list for anelastic (e.g., ["Q4"])
        saved_an = saved.get(VARIANT_ANELASTIC)
        # Filter saved values to options that actually exist (ignore stale values)
        q_value_raw = saved_an if isinstance(saved_an, list) else ([saved_an] if isinstance(saved_an, str) else [])
        q_value = [q for q in q_value_raw if (q in ('C4', 'Q4') and has_q4) or (q in ('C8', 'Q8') and has_q8) or (q == 'F4.0' and has_f40) or (q == 'F4.6' and has_f46)]

        q_check = dcc.Checklist(
            id={"type": "dataset-anelastic", "base": base},
            options=q_options,
            value=q_value,
            inputStyle={"marginRight": "6px"},
            # Keep the label wrapper visible so the checkbox input still renders
            # consistently across Dash/Bootstrap versions. The labels themselves
            # are already empty strings, so no extra text is shown.
            labelStyle={"display": "inline-block", "marginBottom": "0"},
            style={"display": "flex", "flexDirection": "row", "gap": "8px", "alignItems": "center"}
        )

        # Render the three checkboxes in one horizontal line without separators
        response_cell = html.Td(
            html.Div([
                html.Div(elastic_check, style={"display": "inline-block", "marginRight": "8px"}),
                html.Div(q_check, style={"display": "inline-block"}),
            ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "gap": "6px"}),
            style={"textAlign": "center", "whiteSpace": "nowrap"}
        )

        # Try to extract display tokens from a sample dataset path for nicer columns
        sample_path = (elastic_paths[0] if elastic_paths else (anelastic_paths[0] if anelastic_paths else (next((v[0] for v in variants.values() if v), None))))
        sample_dataset = ""
        if sample_path:
            info_sample = parse_dataset_info(Path(sample_path))
            sample_dataset = info_sample.dataset if info_sample else ""

        # Build test ids directly from all datasets in this row so the column
        # always reflects every available response for the grouped configuration.
        base_nocg = re.sub(r"_cg-[^_]+$", "", base)
        test_ids = []
        seen_test_ids = set()
        for path_list in variants.values():
            for p in path_list:
                info_variant = parse_dataset_info(Path(p))
                dataset_variant = info_variant.dataset if info_variant else ""
                test_id = parse_test_id(dataset_variant)
                if test_id and test_id not in seen_test_ids:
                    seen_test_ids.add(test_id)
                    test_ids.append(test_id)
        test_ids.sort(key=test_id_sort_key)
        test_val = ",".join(test_ids) if test_ids else (parse_test_id(sample_dataset) or "-")
        cg_val = base_to_cg.get(base_nocg, "")

        # Domain: extract ver token (e.g. 'a','r','x') from the dataset; fallback to '?'
        m_ver = re.search(r"ver[-_]?([A-Za-z0-9]+)", sample_dataset, flags=re.IGNORECASE)
        domain_val = m_ver.group(1) if m_ver else "?"

        m_pml = re.search(r"pml-(?P<mode>[^_]+)", base, flags=re.IGNORECASE)
        pml_val = pml_label(m_pml.group('mode')) if m_pml else "-"
        m_res = re.search(r"res-(?P<value>[^_]+)", base, flags=re.IGNORECASE)
        res_display = m_res.group('value') if m_res else "-"
        m_stencil = re.search(r"(upwind-drp|traditional|upwind)[-_]([^_]+)", base)
        if m_stencil:
            stencil_full = m_stencil.group(1)
            stencil_val = stencil_abbr.get(stencil_full, stencil_full)
            order_val = m_stencil.group(2)
        else:
            stencil_val = "-"
            order_val = "-"

        # Only include selected columns in each row
        col_val_map = {
            "test": html.Td(test_val, style={"textAlign": "center"}),
            "domain": html.Td(domain_val, style={"textAlign": "center"}),
            "cg": html.Td(cg_val, style={"textAlign": "center"}),
            "stencil": html.Td(stencil_val, style={"textAlign": "center"}),
            "order": html.Td(order_val, style={"textAlign": "center"}),
            "res": html.Td(res_display, style={"textAlign": "center"}),
            "pml": html.Td(pml_val, style={"textAlign": "center"}),
            "response": response_cell,
        }
        row_children = [col_val_map[c] for c in selected_columns]
        rows.append(html.Tr(row_children))

    table = dbc.Table(
        [header, html.Tbody(rows)],
        bordered=True,
        hover=True,
        size="sm",
        responsive=True,
        style={"padding": "0", "margin": "0", "borderCollapse": "collapse"},
        className="table-compact"
    )
    return html.Div([stencil_legend, table]), grouped, base_order


@app.callback(
    Output("dataset-selection-store", "data"),
    [Input("clear-dataset-button", "n_clicks"),
     Input({"type": "dataset-elastic", "base": ALL}, "value"),
     Input({"type": "dataset-anelastic", "base": ALL}, "value")],
    [State("dataset-selection-store", "data")],
    prevent_initial_call=True,
)
def persist_dataset_selection(clear_clicks, elastic_values, anelastic_values, selection_store):
    triggered_id = ctx.triggered_id if ctx else None
    if not triggered_id:
        raise PreventUpdate
    if triggered_id == "clear-dataset-button":
        return {}
    if not isinstance(triggered_id, dict):
        raise PreventUpdate
    variant = VARIANT_ELASTIC if triggered_id.get("type") == "dataset-elastic" else VARIANT_ANELASTIC
    base = triggered_id.get("base")
    if not base:
        raise PreventUpdate
    # payload may be boolean (elastic) or list of selected Q values (anelastic)
    payload = ctx.triggered[0]["value"]
    store = dict(selection_store or {})
    entry = store.setdefault(base, {})
    entry[variant] = payload
    return store

@app.callback(
    Output("line-properties-store", "data"),
    [Input({"type": "width", "dataset": ALL}, "value"),
     Input({"type": "dash", "dataset": ALL}, "value"),
     Input({"type": "color", "dataset": ALL}, "value")],
    State("line-properties-store", "data"),
    prevent_initial_call=True,
)
def update_line_properties(width_values, dash_values, color_values, current_props):
    if not current_props: current_props = {}
    triggered = ctx.triggered if ctx else []
    if triggered:
        prop_id = triggered[0]['prop_id']
        import json
        id_dict = json.loads(prop_id.split('.')[0])
        ds, typ, val = id_dict['dataset'], id_dict['type'], triggered[0]['value']
        if ds not in current_props: current_props[ds] = {}
        current_props[ds][typ] = val
    return current_props

@app.callback(
    [Output("plot-container", "children"), Output("line-controls-container", "children")],
    [Input("station-dropdown", "data"),
     Input("plot-checklist", "value"),
     Input("plot-grid-mode", "value"),
     Input({"type": "dataset-elastic", "base": ALL}, "value"),
     Input({"type": "dataset-anelastic", "base": ALL}, "value"),
     Input("plot-height-slider", "value"),
     Input("plot-title-toggle", "value"),
     Input("dataset-selection-store", "data")],
    [State("dataset-path-map", "data"), State("dataset-base-order", "data"), State("line-properties-store", "data")]
)
def update_plot(station, plots_selected, plot_grid_mode, elastic_values, anelastic_values, plot_height, plot_title_toggle, selection_store, path_map, base_order, properties):
    selected_infos = []
    if base_order and isinstance(path_map, dict):
        for idx, base in enumerate(base_order):
            variants = path_map.get(base, {})
            entry = selection_store.get(base, {}) if isinstance(selection_store, dict) else {}
            # Elastic: entry may be boolean True/False
            if entry.get(VARIANT_ELASTIC):
                p_list = variants.get(VARIANT_ELASTIC)
                if p_list:
                    p = p_list[0]
                    info = parse_dataset_info(Path(p))
                    if info:
                        selected_infos.append(info)
            # Anelastic: entry is a list of selected Q strings (e.g., ['Q4'])
            anel_sel = entry.get(VARIANT_ANELASTIC)
            if anel_sel:
                p_list = variants.get(VARIANT_ANELASTIC, [])
                for qsel in (anel_sel if isinstance(anel_sel, list) else [anel_sel]):
                    # Match the selected response against the normalized dataset id
                    # used everywhere else in the app (c4, c8, f40, f46).
                    found = None
                    expected_prefix = selection_to_test_prefix(qsel)
                    for p in p_list:
                        try:
                            pi = parse_dataset_info(Path(p))
                            ds = pi.dataset if pi else p
                        except Exception:
                            ds = p
                        if parse_test_id(ds).lower().startswith(expected_prefix):
                            found = p
                            break
                    # fallback to first anelastic path if not found
                    if not found and p_list:
                        found = p_list[0]
                    if found:
                        info = parse_dataset_info(Path(found))
                        if info:
                            selected_infos.append(info)

    show_title = plot_title_toggle and "show_title" in plot_title_toggle
    # If stacked mode: create a single subplot figure with shared legend
    if plot_grid_mode == "stacked":
        plots = plots_selected or []
        n = len(plots)
        if n == 0:
            stacked_fig = go.Figure()
        else:
            # Build per-plot figures using existing make_figure then merge traces/layout
            per_plot_figs: List[go.Figure] = [make_figure(station or "", selected_infos, p, properties, plot_height, show_title=show_title) for p in plots]
            stacked_fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0)

            # Add traces from each per-plot figure into the appropriate subplot row
            for row_idx, fig in enumerate(per_plot_figs, start=1):
                for trace in fig.data:
                    # clone trace and control legend visibility only on top row
                    new_trace = trace.to_plotly_json()
                    # Ensure legendgroup so toggling affects all subplots
                    new_trace['legendgroup'] = trace.name
                    new_trace['showlegend'] = (row_idx == 1)
                    stacked_fig.add_trace(new_trace, row=row_idx, col=1)

                # Copy axis layout settings for this subplot from the per-plot fig
                # X axis: per-plot fig.layout.xaxis applies to this row
                x_layout = fig.layout.xaxis.to_plotly_json() if getattr(fig.layout, 'xaxis', None) else {}
                y_layout = fig.layout.yaxis.to_plotly_json() if getattr(fig.layout, 'yaxis', None) else {}

                # Determine axis keys for update_xaxes/update_yaxes
                # use Plotly's update_xaxes to apply settings to specific row/col
                try:
                    # Remove title from non-bottom x-axes; keep ticks only on bottom
                    if row_idx != n:
                        x_layout['showticklabels'] = False
                        x_layout.pop('title', None)
                    else:
                        # ensure bottom x-axis has title
                        x_layout['title'] = {'text': 'Time (s)'}
                except Exception:
                    pass

                # Apply axis settings
                stacked_fig.update_xaxes(**{k: v for k, v in x_layout.items() if k not in ['_name']}, row=row_idx, col=1)
                stacked_fig.update_yaxes(**{k: v for k, v in y_layout.items() if k not in ['_name']}, row=row_idx, col=1)

            # Preserve overall layout aesthetics from first per-plot figure
            base_layout = per_plot_figs[0].layout
            layout_updates = {
                'height': plot_height * max(1, n),
                'margin': dict(l=60, r=30, t=60, b=40),
                'font': base_layout.font.to_plotly_json() if getattr(base_layout, 'font', None) else None,
                'plot_bgcolor': base_layout.plot_bgcolor if getattr(base_layout, 'plot_bgcolor', None) else 'white',
                'paper_bgcolor': base_layout.paper_bgcolor if getattr(base_layout, 'paper_bgcolor', None) else 'white',
            }
            # Remove None entries
            layout_updates = {k: v for k, v in layout_updates.items() if v is not None}
            stacked_fig.update_layout(**layout_updates)

            # Ensure legend looks like per-plot (no bold/extra-wide border) and is transparent
            if getattr(base_layout, 'legend', None):
                try:
                    legend_spec = base_layout.legend.to_plotly_json()
                except Exception:
                    legend_spec = {}
            else:
                legend_spec = {}
            legend_spec.setdefault('orientation', 'h')
            legend_spec.setdefault('y', 1.02)
            legend_spec.setdefault('yanchor', 'bottom')
            legend_spec.setdefault('xanchor', 'right')
            # remove bold borders / heavy styling
            legend_spec['borderwidth'] = 0
            legend_spec['bgcolor'] = 'rgba(0,0,0,0)'
            # ensure legend font matches per-plot legend font if present
            if getattr(base_layout, 'legend', None) and getattr(base_layout.legend, 'font', None):
                try:
                    legend_spec['font'] = base_layout.legend.font.to_plotly_json()
                except Exception:
                    pass
            stacked_fig.update_layout(legend=legend_spec)

        figs = [stacked_fig]

        figs = [stacked_fig]
    else:
        figs = [make_figure(station or "", selected_infos, p, properties, plot_height, show_title=show_title) for p in plots_selected or []]

    label_to_dataset: Dict[str, str] = {}
    for info in selected_infos:
        test_id = parse_test_id(info.dataset)
        label = test_id if test_id else info.dataset
        label_to_dataset[label] = info.dataset
    selected_datasets = sorted(label_to_dataset.keys())
    if selected_datasets:
        rows = []
        for ds in selected_datasets:
            props = properties.get(ds, {})
            dataset_name = label_to_dataset.get(ds, ds)
            default_color = props.get('color') or dataset_color(dataset_name)
            # Line type dropdown replaces opacity slider
            line_type = props.get('dash', 'solid')
            line_type_dropdown = dcc.Dropdown(
                id={"type": "dash", "dataset": ds},
                options=[
                    {"label": "Solid", "value": "solid"},
                    {"label": "Dash", "value": "dash"},
                    {"label": "Dot", "value": "dot"},
                    {"label": "DashDot", "value": "dashdot"},
                    {"label": "LongDash", "value": "longdash"},
                    {"label": "LongDashDot", "value": "longdashdot"}
                ],
                value=line_type,
                clearable=False,
                style={"width": "110px"}
            )
            rows.append(html.Tr([
                html.Td(ds),
                html.Td(dcc.Slider(id={"type": "width", "dataset": ds}, min=1, max=5, value=props.get('width', 2), step=1, marks={1:'1',3:'3',5:'5'})),
                html.Td(line_type_dropdown),
                html.Td(dcc.Input(id={"type": "color", "dataset": ds}, type="color", value=default_color, style={"width": "48px", "height": "32px", "border": "none", "padding": "0"}))
            ]))
        controls = dbc.Table(
            [html.Thead(html.Tr([html.Th("Dataset"), html.Th("Line Width"), html.Th("Line Type"), html.Th("Color")])),
             html.Tbody(rows)],
            bordered=True,
            size="sm",
        )
    else:
        controls = html.Div()
        
    return [dcc.Graph(figure=fig) for fig in figs], controls

# --- 6. EXECUTION ---

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    host = os.environ.get("HOST", "0.0.0.0")
    app.run(host=host, port=port, debug=False)
