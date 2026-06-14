import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="MaintainIQ — Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
*, *::before, *::after { box-sizing: border-box; }

.stApp { background-color: #111418; font-family: 'IBM Plex Sans', sans-serif; }

/* Hide streamlit chrome & sidebar entirely */
#MainMenu, header[data-testid="stHeader"], footer,
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
section[data-testid="stSidebar"] { display: none !important; }

.block-container { padding: 0.75rem 2rem 2rem 2rem !important; max-width: 1600px; }

/* ── Topbar ── */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 20px; background: #0D1017;
    border: 1px solid #1F2937; border-radius: 4px; margin-bottom: 14px;
}
.topbar-logo { font-family:'IBM Plex Mono',monospace; font-size:14px; font-weight:500; color:#F3F4F6; letter-spacing:1px; }
.topbar-sep { width:1px; height:18px; background:#374151; margin:0 16px; }
.topbar-sub { font-size:10px; color:#6B7280; text-transform:uppercase; letter-spacing:2px; }
.topbar-right { display:flex; align-items:center; gap:20px; }
.sys-stat { display:flex; align-items:center; gap:6px; font-size:10px; color:#9CA3AF; font-family:'IBM Plex Mono',monospace; }
.led { width:6px; height:6px; border-radius:50%; background:#22C55E; box-shadow:0 0 6px #22C55E; animation:blink 3s infinite; }
.led-w { background:#EAB308; box-shadow:0 0 6px #EAB308; }
@keyframes blink { 0%,90%,100%{opacity:1} 95%{opacity:.2} }

/* ── Control Bar ── */
.ctrl-bar {
    display: flex; align-items: center; gap: 16px;
    padding: 10px 20px; background: #0D1017;
    border: 1px solid #1F2937; border-radius: 4px; margin-bottom: 14px;
}
.ctrl-label { font-family:'IBM Plex Mono',monospace; font-size:9px; color:#6B7280; letter-spacing:2px; text-transform:uppercase; margin-bottom:4px; }
.ctrl-sep { width:1px; height:36px; background:#1F2937; margin:0 4px; }

/* ── Status badge inline ── */
.machine-status {
    display:flex; align-items:center; gap:12px;
    background:#0A0E14; border:1px solid #1F2937; border-radius:4px;
    padding:8px 14px; margin-left:auto;
}
.ms-item { display:flex; flex-direction:column; }
.ms-key { font-family:'IBM Plex Mono',monospace; font-size:8px; color:#374151; letter-spacing:2px; text-transform:uppercase; margin-bottom:2px; }
.ms-val { font-family:'IBM Plex Mono',monospace; font-size:13px; color:#D1D5DB; }
.ms-val.ok   { color:#4ADE80; }
.ms-val.warn { color:#FBBF24; }
.ms-val.crit { color:#F87171; }
.ms-sep { width:1px; height:28px; background:#1F2937; }

/* ── KPI Cards ── */
.kpi-card { background:#0D1017; border:1px solid #1F2937; border-top:2px solid #1F2937; padding:14px 16px 12px; border-radius:4px; }
.kpi-card.cb { border-top-color:#2563EB; } .kpi-card.cc { border-top-color:#06B6D4; }
.kpi-card.cr { border-top-color:#DC2626; } .kpi-card.cs { border-top-color:#6366F1; }
.kpi-id { font-family:'IBM Plex Mono',monospace; font-size:9px; color:#6B7280; letter-spacing:2px; text-transform:uppercase; margin-bottom:6px; }
.kpi-v  { font-family:'IBM Plex Mono',monospace; font-size:28px; font-weight:500; line-height:1; margin-bottom:4px; }
.kpi-card.cb .kpi-v{color:#60A5FA;} .kpi-card.cc .kpi-v{color:#22D3EE;}
.kpi-card.cr .kpi-v{color:#F87171;} .kpi-card.cs .kpi-v{color:#A5B4FC;}
.kpi-d { font-size:10px; color:#6B7280; font-family:'IBM Plex Mono',monospace; }

/* ── Section Headers ── */
.sec-hdr { display:flex; align-items:baseline; gap:12px; padding:10px 0 8px; border-bottom:1px solid #1F2937; margin-bottom:14px; }
.sec-tag { font-family:'IBM Plex Mono',monospace; font-size:9px; color:#9CA3AF; letter-spacing:3px; text-transform:uppercase; background:#161B22; border:1px solid #374151; padding:2px 7px; border-radius:2px; }
.sec-ttl { font-size:13px; font-weight:500; color:#E5E7EB; }

/* ── Badges ── */
.badge { display:inline-flex; align-items:center; gap:5px; padding:3px 9px; border-radius:2px; font-family:'IBM Plex Mono',monospace; font-size:10px; font-weight:500; letter-spacing:1px; text-transform:uppercase; }
.bdot { width:5px; height:5px; border-radius:50%; }
.b-ok  { background:#052E16; color:#4ADE80; border:1px solid #166534; } .b-ok  .bdot{background:#4ADE80;}
.b-warn{ background:#1C1917; color:#FBBF24; border:1px solid #78350F; } .b-warn.bdot{background:#FBBF24;}
.b-crit{ background:#1F0000; color:#FCA5A5; border:1px solid #7F1D1D; } .b-crit .bdot{background:#FCA5A5;}

/* ── Panel ── */
.panel { background:#0D1017; border:1px solid #1F2937; border-radius:4px; overflow:hidden; }
.panel-hdr { padding:8px 14px; background:#161B22; border-bottom:1px solid #1F2937; font-family:'IBM Plex Mono',monospace; font-size:10px; color:#9CA3AF; letter-spacing:2px; text-transform:uppercase; display:flex; align-items:center; justify-content:space-between; }

/* ── Data Rows ── */
.dr { display:flex; justify-content:space-between; align-items:center; padding:8px 14px; border-bottom:1px solid #161B22; font-size:12px; }
.dr:hover { background:#161B22; }
.dk { color:#9CA3AF; font-size:11px; }
.dv { font-family:'IBM Plex Mono',monospace; font-size:11px; color:#D1D5DB; }
.dv-ok{color:#4ADE80;} .dv-warn{color:#FBBF24;} .dv-crit{color:#F87171;}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background:#0D1017; border:1px solid #1F2937; border-radius:4px; padding:3px; gap:2px; }
.stTabs [data-baseweb="tab"] { background:transparent !important; color:#6B7280 !important; border-radius:3px !important; padding:7px 20px !important; font-size:12px !important; font-weight:500 !important; }
.stTabs [aria-selected="true"] { background:#161B22 !important; color:#E5E7EB !important; border:1px solid #2563EB !important; }

/* ── Metrics ── */
[data-testid="stMetric"] { background:#0D1017; border:1px solid #1F2937; border-radius:4px; padding:12px 16px; }
[data-testid="stMetricLabel"] { color:#9CA3AF !important; font-size:10px !important; font-family:'IBM Plex Mono',monospace !important; text-transform:uppercase !important; letter-spacing:1px !important; }
[data-testid="stMetricValue"] { color:#E5E7EB !important; font-family:'IBM Plex Mono',monospace !important; font-size:18px !important; }

/* ── Number input & slider ── */
[data-testid="stNumberInput"] input,
[data-testid="stNumberInput"] button {
    background:#0A0E14 !important; border-color:#1F2937 !important;
    color:#E5E7EB !important; font-family:'IBM Plex Mono',monospace !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] { background:#2563EB !important; }
[data-testid="stSlider"] [data-baseweb="slider"] div[class*="Track"] { background:#1F2937 !important; }

/* Hide label from number_input/slider since we use custom labels */
[data-testid="stNumberInput"] label,
[data-testid="stSlider"] label { display:none !important; }

.stDataFrame { border-radius:4px; overflow:hidden; }

.hdiv { height:1px; background:#1F2937; margin:14px 0; }

.footer { text-align:center; padding:20px 0; border-top:1px solid #1F2937; margin-top:8px; }
.footer-id  { font-family:'IBM Plex Mono',monospace; font-size:10px; color:#4B5563; letter-spacing:3px; text-transform:uppercase; margin-bottom:4px; }
.footer-sub { font-size:11px; color:#374151; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Load Data ──
try:
    df = pd.read_csv("data/predictive_maintenance.csv")
except Exception:
    st.error("File data/predictive_maintenance.csv tidak ditemukan.")
    st.stop()

# ── Preprocessing ──
features = [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
]
scaled_features = MinMaxScaler().fit_transform(df[features])
similarity = cosine_similarity(scaled_features)

# ── Helpers ──
def recommend_machine(idx, top_n=5):
    scores = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)[1:top_n+1]
    result = df.iloc[[i for i, _ in scores]].copy()
    result["Similarity Score"] = [round(s, 4) for _, s in scores]
    return result

def get_risk(wear):
    if wear < 80:    return "b-ok",   "NOMINAL",  "ok"
    elif wear < 150: return "b-warn", "ADVISORY", "warn"
    else:            return "b-crit", "CRITICAL", "crit"

def val_cls(raw_key, val):
    if raw_key == "Target": return "dv-crit" if val == 1 else "dv-ok"
    if raw_key == "Tool wear [min]":
        if val > 150: return "dv-crit"
        if val > 80:  return "dv-warn"
        return "dv-ok"
    return ""

PCFG = dict(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0D1017",
    font=dict(family="IBM Plex Mono", color="#9CA3AF", size=11),
    margin=dict(l=12, r=12, t=36, b=12),
)

def apply_theme(fig, height=400):
    fig.update_layout(height=height, **PCFG)
    fig.update_xaxes(gridcolor="#1F2937", linecolor="#374151", tickfont=dict(color="#9CA3AF"))
    fig.update_yaxes(gridcolor="#1F2937", linecolor="#374151", tickfont=dict(color="#9CA3AF"))
    return fig

# ── Computed Values ──
failure_count  = int(df["Target"].sum())
avg_torque     = round(df["Torque [Nm]"].mean(), 2)
avg_toolwear   = round(df["Tool wear [min]"].mean(), 2)
total_machines = len(df)
failure_rate   = failure_count / total_machines * 100

# ── Topbar ──
st.markdown(f"""
<div class='topbar'>
    <div style='display:flex;align-items:center;'>
        <span class='topbar-logo'>⚙ MaintainIQ</span>
        <div class='topbar-sep'></div>
        <span class='topbar-sub'>Predictive Maintenance · Recommendation System</span>
    </div>
    <div class='topbar-right'>
        <div class='sys-stat'><span class='led'></span>SYS ONLINE</div>
        <div class='sys-stat'><span class='led led-w'></span>{failure_count} FAULTS</div>
        <div class='sys-stat' style='color:#60A5FA;font-weight:500;'>{total_machines:,} UNITS</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Control Bar (replaces sidebar) ──
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 3])

with ctrl_col1:
    st.markdown("<div class='ctrl-label'>Machine Index</div>", unsafe_allow_html=True)
    machine_index = st.number_input(
        "Machine Index", min_value=0, max_value=len(df)-1, value=10
    )

with ctrl_col2:
    st.markdown("<div class='ctrl-label'>Recommendation Count</div>", unsafe_allow_html=True)
    top_n = st.slider("Recommendation Count", min_value=1, max_value=10, value=5)

with ctrl_col3:
    sel  = df.iloc[machine_index]
    wear = float(sel["Tool wear [min]"])
    rc, rl, rwc = get_risk(wear)
    st.markdown(f"""
    <div class='machine-status'>
        <div class='ms-item'>
            <div class='ms-key'>Machine Index</div>
            <div class='ms-val'>#{machine_index}</div>
        </div>
        <div class='ms-sep'></div>
        <div class='ms-item'>
            <div class='ms-key'>UDI</div>
            <div class='ms-val'>{sel.get("UDI","—")}</div>
        </div>
        <div class='ms-sep'></div>
        <div class='ms-item'>
            <div class='ms-key'>Type</div>
            <div class='ms-val'>{sel.get("Type","—")}</div>
        </div>
        <div class='ms-sep'></div>
        <div class='ms-item'>
            <div class='ms-key'>Tool Wear</div>
            <div class='ms-val {rwc}'>{wear:.0f} min</div>
        </div>
        <div class='ms-sep'></div>
        <div class='ms-item'>
            <div class='ms-key'>Status</div>
            <div class='ms-val'><span class='badge {rc}'><span class='bdot'></span>{rl}</span></div>
        </div>
        <div class='ms-sep'></div>
        <div class='ms-item'>
            <div class='ms-key'>Author</div>
            <div class='ms-val' style='font-size:10px;color:#374151;'>Moreno Gibran Hardayan</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── KPI Row ──
c1, c2, c3, c4 = st.columns(4)
kpis = [
    (c1, "cb", "01 · FLEET SIZE",    f"{total_machines:,}",  "total units monitored"),
    (c2, "cr", "02 · FAULT COUNT",   f"{failure_count:,}",   f"{failure_rate:.1f}% failure rate"),
    (c3, "cc", "03 · AVG TORQUE",    f"{avg_torque}",        "Nm  rotational force"),
    (c4, "cs", "04 · AVG TOOL WEAR", f"{avg_toolwear}",      "min  cumulative wear"),
]
for col, cls, label, val, desc in kpis:
    with col:
        st.markdown(f"""
        <div class='kpi-card {cls}'>
            <div class='kpi-id'>{label}</div>
            <div class='kpi-v'>{val}</div>
            <div class='kpi-d'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Tabs ──
tab1, tab2, tab3 = st.tabs(["  Dashboard  ", "  Rekomendasi  ", "  Analytics  "])

# ── TAB 1: Dashboard ──
with tab1:
    selected = df.iloc[machine_index]
    wear_val = float(selected["Tool wear [min]"])
    rc, rl, _ = get_risk(wear_val)

    st.markdown(f"""
    <div class='sec-hdr'>
        <span class='sec-tag'>INSPECT</span>
        <span class='sec-ttl'>Machine #{machine_index} · Sensor Readout</span>
        <span style='margin-left:auto;'><span class='badge {rc}'><span class='bdot'></span>{rl}</span></span>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        display_keys = {
            "UDI": "UDI", "Product ID": "Product ID", "Type": "Machine Type",
            "Air temperature [K]": "Air Temp (K)", "Process temperature [K]": "Process Temp (K)",
            "Rotational speed [rpm]": "Rotation Speed (rpm)", "Torque [Nm]": "Torque (Nm)",
            "Tool wear [min]": "Tool Wear (min)", "Target": "Failure Flag", "Failure Type": "Failure Type",
        }
        rows = "".join(
            f"<div class='dr'><span class='dk'>{lbl}</span><span class='dv {val_cls(k,selected[k])}'>{selected[k]}</span></div>"
            for k, lbl in display_keys.items() if k in selected.index
        )
        st.markdown(f"<div class='panel'><div class='panel-hdr'>PARAMETER TABLE<span style='color:#2563EB;'>■</span></div>{rows}</div>", unsafe_allow_html=True)

    with right:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=wear_val,
            number={"suffix": " min", "font": {"family": "IBM Plex Mono", "color": "#E5E7EB", "size": 26}},
            title={"text": "TOOL WEAR INDEX", "font": {"color": "#9CA3AF", "size": 10, "family": "IBM Plex Mono"}},
            gauge={
                "axis": {"range": [0, 250], "tickfont": {"color": "#6B7280", "size": 9}, "tickcolor": "#374151"},
                "bar": {"color": "#2563EB", "thickness": 0.2},
                "bgcolor": "#0D1017", "borderwidth": 1, "bordercolor": "#1F2937",
                "steps": [
                    {"range": [0, 80],    "color": "#052E16"},
                    {"range": [80, 150],  "color": "#1C1917"},
                    {"range": [150, 250], "color": "#1F0000"},
                ],
                "threshold": {"line": {"color": "#DC2626", "width": 2}, "thickness": 0.75, "value": 200}
            }
        ))
        gauge.update_layout(height=240, **PCFG)
        st.plotly_chart(gauge, use_container_width=True)

        cats      = ["Air Temp", "Process Temp", "RPM", "Torque", "Tool Wear"]
        vals_raw  = [float(selected[f]) for f in features]
        norm_vals = [(vals_raw[i] - df[f].min()) / (df[f].max() - df[f].min() + 1e-9) for i, f in enumerate(features)]

        radar = go.Figure(go.Scatterpolar(
            r=norm_vals + [norm_vals[0]], theta=cats + [cats[0]],
            fill="toself", fillcolor="rgba(37,99,235,0.10)",
            line=dict(color="#2563EB", width=1.5),
        ))
        radar.update_layout(
            polar=dict(
                bgcolor="#0D1017",
                radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#4B5563", size=8), gridcolor="#1F2937", linecolor="#374151"),
                angularaxis=dict(tickfont=dict(color="#9CA3AF", size=10, family="IBM Plex Mono"), gridcolor="#1F2937", linecolor="#374151"),
            ),
            showlegend=False, height=240, **PCFG
        )
        st.plotly_chart(radar, use_container_width=True)

# ── TAB 2: Recommendation ──
with tab2:
    st.markdown(f"""
    <div class='sec-hdr'>
        <span class='sec-tag'>SIMILARITY</span>
        <span class='sec-ttl'>Top {top_n} Machines — Cosine Similarity Match</span>
    </div>
    """, unsafe_allow_html=True)

    recommendation = recommend_machine(machine_index, top_n)
    sim_all = similarity[machine_index]

    m1, m2, m3 = st.columns(3)
    m1.metric("Avg Similarity", f"{np.mean(sim_all):.4f}")
    m2.metric("Max Similarity", f"{np.max(sim_all):.4f}")
    m3.metric("Min Similarity", f"{np.min(sim_all):.4f}")

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    st.dataframe(recommendation, use_container_width=True, height=240, hide_index=True)
    st.markdown("<div class='hdiv'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='sec-hdr'>
        <span class='sec-tag'>RANK</span>
        <span class='sec-ttl'>Similarity Score — Ranked View</span>
    </div>
    """, unsafe_allow_html=True)

    fig_bar = px.bar(
        recommendation, x="UDI", y="Similarity Score", color="Similarity Score",
        color_continuous_scale=[[0,"#1F2937"],[0.5,"#1D4ED8"],[1.0,"#60A5FA"]],
        text=recommendation["Similarity Score"].apply(lambda v: f"{v:.4f}"),
    )
    fig_bar.update_traces(textfont_size=10, textposition="outside", marker_line_width=0, textfont_color="#E5E7EB")
    fig_bar.update_coloraxes(showscale=False)
    apply_theme(fig_bar, 360)
    fig_bar.update_layout(
        xaxis_title="Machine UDI", yaxis_title="Score",
        xaxis_title_font=dict(color="#9CA3AF"),
        yaxis_title_font=dict(color="#9CA3AF")
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── TAB 3: Analytics ──
with tab3:
    st.markdown("""
    <div class='sec-hdr'>
        <span class='sec-tag'>ANALYTICS</span>
        <span class='sec-ttl'>Fleet-Wide Failure Analysis</span>
    </div>
    """, unsafe_allow_html=True)

    colA, colB = st.columns(2, gap="large")
    with colA:
        fig_pie = px.pie(df, names="Failure Type", hole=0.55,
            color_discrete_sequence=["#2563EB","#06B6D4","#DC2626","#6366F1","#10B981","#F59E0B"])
        fig_pie.update_traces(
            textinfo="percent+label",
            textfont=dict(size=10, family="IBM Plex Mono", color="#E5E7EB"),
            marker=dict(line=dict(color="#111418", width=2))
        )
        fig_pie.update_layout(
            title=dict(text="FAILURE TYPE DISTRIBUTION", font=dict(color="#9CA3AF", size=10, family="IBM Plex Mono")),
            legend=dict(font=dict(color="#9CA3AF", size=9, family="IBM Plex Mono")),
        )
        apply_theme(fig_pie, 400)
        st.plotly_chart(fig_pie, use_container_width=True)

    with colB:
        fc = df["Failure Type"].value_counts()
        fig_bar2 = px.bar(x=fc.index, y=fc.values, color=fc.values,
            color_continuous_scale=[[0,"#161B22"],[0.4,"#1E3A8A"],[1.0,"#2563EB"]])
        fig_bar2.update_coloraxes(showscale=False)
        fig_bar2.update_traces(marker_line_width=0)
        fig_bar2.update_layout(
            title=dict(text="FAILURE FREQUENCY COUNT", font=dict(color="#9CA3AF", size=10, family="IBM Plex Mono")),
            xaxis_title="", yaxis_title="Count",
            yaxis_title_font=dict(color="#9CA3AF"),
        )
        apply_theme(fig_bar2, 400)
        st.plotly_chart(fig_bar2, use_container_width=True)

    st.markdown("<div class='hdiv'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='sec-hdr'>
        <span class='sec-tag'>MATRIX</span>
        <span class='sec-ttl'>Machine Similarity Heatmap — First 50 Units</span>
    </div>
    """, unsafe_allow_html=True)

    heatmap = px.imshow(
        similarity[:50, :50],
        color_continuous_scale=[[0,"#0D1017"],[0.3,"#1E3A8A"],[0.7,"#2563EB"],[1.0,"#93C5FD"]],
        aspect="auto", zmin=0, zmax=1
    )
    heatmap.update_layout(
        coloraxis_colorbar=dict(
            tickfont=dict(color="#9CA3AF", size=9, family="IBM Plex Mono"),
            title=dict(text="SIM", font=dict(color="#9CA3AF", size=10)),
            thickness=12, len=0.6
        ),
    )
    apply_theme(heatmap, 540)
    st.plotly_chart(heatmap, use_container_width=True)

# ── Footer ──
st.markdown("""
<div class='footer'>
    <div class='footer-id'>MAINTAINIQ · PREDICTIVE MAINTENANCE RECOMMENDATION SYSTEM</div>
    <div class='footer-sub'>Moreno Gibran Hardayan · Final Project Sistem Rekomendasi</div>
</div>
""", unsafe_allow_html=True)