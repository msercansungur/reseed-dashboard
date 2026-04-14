import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime

# ── CONFIG ───────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
CLEAN_FILE = BASE / 'data' / 'reseed_clean.csv'

# Color palette — matches pipeline
PRIMARY   = '#1B4F72'
GREEN     = '#1A7A4A'
ACCENT    = '#C0392B'
GOLD      = '#D4860B'
NEUTRAL   = '#707070'
LIGHT_BG  = '#F8FAFC'

PALETTE   = [PRIMARY, GREEN, ACCENT, GOLD, '#8E44AD', '#16A085', '#2E86AB']

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "RE-SEED Dashboard",
    page_icon   = "📊",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #F8FAFC; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1B4F72;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label {
        color: #A8C8E0 !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    
    /* KPI cards */
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-top: 4px solid #1B4F72;
        height: 110px;
    }
    .kpi-value {
        font-size: 32px;
        font-weight: 700;
        color: #1B4F72;
        line-height: 1.2;
    }
    .kpi-label {
        font-size: 12px;
        color: #707070;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    .kpi-target {
        font-size: 11px;
        color: #1A7A4A;
        margin-top: 4px;
    }
    .kpi-card-warn { border-top-color: #C0392B; }
    .kpi-card-warn .kpi-value { color: #C0392B; }
    .kpi-card-gold { border-top-color: #D4860B; }
    .kpi-card-gold .kpi-value { color: #D4860B; }
    
    /* Section headers */
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #1B4F72;
        border-bottom: 2px solid #1B4F72;
        padding-bottom: 8px;
        margin-bottom: 20px;
        margin-top: 10px;
    }
    
    /* Page title */
    .page-title {
        font-size: 26px;
        font-weight: 700;
        color: #1B4F72;
        margin-bottom: 4px;
    }
    .page-subtitle {
        font-size: 14px;
        color: #707070;
        margin-bottom: 24px;
    }
    
    /* Info box */
    .info-box {
        background: #EEF4FA;
        border-left: 4px solid #1B4F72;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 13px;
        color: #333;
        margin: 12px 0;
    }
    
    /* Warning box */
    .warn-box {
        background: #FFF3E0;
        border-left: 4px solid #D4860B;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 13px;
        color: #333;
        margin: 12px 0;
    }

    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    
    /* Plotly chart border */
    .js-plotly-plot {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(CLEAN_FILE, low_memory=False)
    numeric = ['I5','I12','C1a','I16','L2','F1e_1','F3','VOC_06','VOC_09']
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

df_full = load_data()
last_updated = datetime.fromtimestamp(CLEAN_FILE.stat().st_mtime).strftime('%d %B %Y')

# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
def get(df, col):
    if col not in df.columns:
        return pd.Series([None]*len(df), index=df.index)
    c = df[col]
    return c.iloc[:,0] if isinstance(c, pd.DataFrame) else c

def get_service_flag(df, keyword):
    """T7-primary, P2-fallback service detection"""
    t7 = get(df, 'T7').astype(str).fillna('')
    p2 = get(df, 'P2').astype(str).fillna('')
    primary = t7.where(
        ~t7.str.contains('not received|have not received', case=False, na=False),
        p2
    )
    return primary.str.contains(keyword, case=False, na=False)

def pct(df, col, val):
    s = get(df, col)
    base = s.notna().sum()
    if base == 0: return 0.0
    return round((s == val).sum() / base * 100, 1)

def freq_df(df, col):
    s = get(df, col)
    vc = s.value_counts(dropna=True).reset_index()
    vc.columns = ['Response','Count']
    vc['Percent'] = (vc['Count'] / s.notna().sum() * 100).round(1)
    return vc

def bar_chart(df, col, title, color=PRIMARY, orientation='h', top_n=10):
    fd = freq_df(df, col).head(top_n)
    if fd.empty:
        return empty_chart(title)
    if orientation == 'h':
        fig = px.bar(fd, x='Count', y='Response', orientation='h',
                     text=fd['Percent'].apply(lambda x: f'{x}%'),
                     color_discrete_sequence=[color])
        fig.update_traces(textposition='outside')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
    else:
        fig = px.bar(fd, x='Response', y='Count',
                     text=fd['Percent'].apply(lambda x: f'{x}%'),
                     color_discrete_sequence=[color])
        fig.update_traces(textposition='outside')
    return style_chart(fig, title)

def grouped_bar(df, group_col, outcome_col, outcome_val, title):
    s1 = get(df, group_col)
    s2 = get(df, outcome_col)
    sub = pd.DataFrame({'group':s1,'outcome':s2}).dropna()
    if len(sub) < 3:
        return empty_chart(title)
    result = {}
    for g in sorted(sub['group'].unique()):
        mask = sub['group'] == g
        n = mask.sum()
        if n > 0:
            result[g] = round((sub['outcome'][mask] == outcome_val).sum() / n * 100, 1)
    if not result:
        return empty_chart(title)
    fd = pd.DataFrame({'Group': list(result.keys()), 'Percent': list(result.values())})
    overall = round((sub['outcome'] == outcome_val).sum() / len(sub) * 100, 1)
    fig = px.bar(fd, x='Group', y='Percent',
                 text=fd['Percent'].apply(lambda x: f'{x}%'),
                 color_discrete_sequence=[PRIMARY, GREEN, ACCENT, GOLD, '#8E44AD'])
    fig.add_hline(y=overall, line_dash='dash', line_color=NEUTRAL,
                  annotation_text=f'Overall: {overall}%',
                  annotation_position='top right')
    fig.update_traces(textposition='outside')
    return style_chart(fig, title, yaxis_title='Percent (%)')

def pie_chart(df, col, title):
    fd = freq_df(df, col)
    if fd.empty:
        return empty_chart(title)
    fig = px.pie(fd, values='Count', names='Response',
                 color_discrete_sequence=PALETTE,
                 hole=0.4)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return style_chart(fig, title)

def stacked_bar(df, group_col, outcome_col, title):
    s1 = get(df, group_col)
    s2 = get(df, outcome_col)
    sub = pd.DataFrame({'group':s1,'outcome':s2}).dropna()
    if len(sub) < 3:
        return empty_chart(title)
    ct = pd.crosstab(sub['group'], sub['outcome'], normalize='index') * 100
    ct = ct.round(1).reset_index()
    ct_melt = ct.melt(id_vars='group', var_name='Outcome', value_name='Percent')
    fig = px.bar(ct_melt, x='group', y='Percent', color='Outcome',
                 text=ct_melt['Percent'].apply(lambda x: f'{x:.0f}%' if x > 5 else ''),
                 color_discrete_sequence=PALETTE,
                 barmode='stack')
    fig.update_traces(textposition='inside')
    return style_chart(fig, title, yaxis_title='Percent (%)')

def scatter_chart(df, x_col, y_col, color_col, title):
    sub = pd.DataFrame({
        'x': get(df, x_col),
        'y': get(df, y_col),
        'color': get(df, color_col)
    }).dropna()
    if len(sub) < 3:
        return empty_chart(title)
    fig = px.scatter(sub, x='x', y='y', color='color',
                     color_discrete_sequence=PALETTE,
                     opacity=0.7)
    return style_chart(fig, title, xaxis_title=x_col, yaxis_title=y_col)

def style_chart(fig, title, xaxis_title=None, yaxis_title=None):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=PRIMARY, family='Calibri'),
                   x=0, xanchor='left'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Calibri', size=11, color='#333'),
        margin=dict(t=50, b=40, l=40, r=40),
        showlegend=True,
        legend=dict(font=dict(size=10), orientation='h',
                    yanchor='bottom', y=-0.3, xanchor='left', x=0),
        xaxis=dict(showgrid=True, gridcolor='#F0F0F0',
                   title=xaxis_title, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#F0F0F0',
                   title=yaxis_title, zeroline=False),
        height=380,
    )
    fig.update_traces(marker=dict(line=dict(width=0.5, color='white')))
    return fig

def empty_chart(title):
    fig = go.Figure()
    fig.add_annotation(text="No data available for current filters",
                       xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(size=13, color=NEUTRAL))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=PRIMARY)),
        plot_bgcolor='white', paper_bgcolor='white', height=380,
        margin=dict(t=50, b=40, l=40, r=40),
    )
    return fig

def kpi_card(label, value, target=None, unit='%', style='normal'):
    style_class = {'normal':'kpi-card', 'warn':'kpi-card kpi-card-warn',
                   'gold':'kpi-card kpi-card-gold'}.get(style, 'kpi-card')
    target_html = f'<div class="kpi-target">Target: {target}{unit}</div>' if target else ''
    return f"""
    <div class="{style_class}">
        <div class="kpi-value">{value}{unit}</div>
        <div class="kpi-label">{label}</div>
        {target_html}
    </div>
    """

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="text-align:center; padding: 10px 0 20px 0;">
            <div style="font-size:22px; font-weight:700; color:white;">RE-SEED</div>
            <div style="font-size:12px; color:#A8C8E0;">Employability Survey Dashboard</div>
            <div style="font-size:11px; color:#7AAEC8; margin-top:4px;">GIZ · Türkiye</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:11px; color:#A8C8E0; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">Navigation</div>', unsafe_allow_html=True)

    page = st.radio("", [
        "📊 Overview & KPIs",
        "👥 Demographics & Coverage",
        "💼 Employment Outcomes",
        "🎓 Services & Training",
        "👶 Childcare & Inclusion",
        "📣 Accountability & Cohesion",
    ], label_visibility='collapsed')

    st.markdown("---")
    st.markdown('<div style="font-size:11px; color:#A8C8E0; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">Filters</div>', unsafe_allow_html=True)

    # City filter
    cities = ['All'] + sorted(get(df_full, 'I3').dropna().unique().tolist())
    sel_city = st.multiselect("City", cities[1:], placeholder="All cities")

    # Gender filter
    genders = get(df_full, 'I4').dropna().unique().tolist()
    sel_gender = st.multiselect("Gender", genders, placeholder="All genders")

    # Nationality filter
    nats = get(df_full, 'I6').dropna().unique().tolist()
    sel_nat = st.multiselect("Nationality", nats, placeholder="All nationalities")

    # Employment filter
    sel_emp = st.selectbox("Employment status", ["All", "Employed", "Not employed"])

    # Has children filter
    sel_children = st.selectbox("Has children", ["All", "Yes", "No"])

    # Service received filter
    sel_service = st.selectbox("Service received", [
        "All", "Vocational training", "Language course", "Advisory support",
        "Formalisation support"
    ])

    st.markdown("---")
    st.markdown(f'<div style="font-size:10px; color:#7AAEC8;">Data updated: {last_updated}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:10px; color:#7AAEC8;">Total respondents: {len(df_full)}</div>', unsafe_allow_html=True)

# ── APPLY FILTERS ─────────────────────────────────────────────────────────────
df = df_full.copy()

if sel_city:
    df = df[get(df, 'I3').isin(sel_city)]
if sel_gender:
    df = df[get(df, 'I4').isin(sel_gender)]
if sel_nat:
    df = df[get(df, 'I6').isin(sel_nat)]
if sel_emp == "Employed":
    df = df[get(df, 'F1') == 'Yes']
elif sel_emp == "Not employed":
    df = df[get(df, 'F1') == 'No']
if sel_children != "All":
    df = df[get(df, 'C1') == sel_children]
if sel_service == "Vocational training":
    df = df[get_service_flag(df, 'Vocational')]
elif sel_service == "Language course":
    df = df[get_service_flag(df, 'language course')]
elif sel_service == "Advisory support":
    df = df[get_service_flag(df, 'Legal')]
elif sel_service == "Formalisation support":
    df = df[get_service_flag(df, 'Formalisation')]

N = len(df)

# ── FILTER INDICATOR BAR ──────────────────────────────────────────────────────
active_filters = []
if sel_city:      active_filters.append(f"City: {', '.join(sel_city)}")
if sel_gender:    active_filters.append(f"Gender: {', '.join(sel_gender)}")
if sel_nat:       active_filters.append(f"Nationality: {', '.join(sel_nat)}")
if sel_emp != "All":      active_filters.append(f"Employment: {sel_emp}")
if sel_children != "All": active_filters.append(f"Children: {sel_children}")
if sel_service != "All":  active_filters.append(f"Service: {sel_service}")

if active_filters:
    st.markdown(f'<div class="info-box">🔍 Active filters: {" · ".join(active_filters)} · <b>N = {N} respondents</b></div>',
                unsafe_allow_html=True)
else:
    st.markdown(f'<div class="info-box">Showing all <b>{N} respondents</b> — use sidebar filters to drill down</div>',
                unsafe_allow_html=True)

if N == 0:
    st.warning("No respondents match the current filters. Please adjust your selection.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW & KPIs
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview & KPIs":
    st.markdown('<div class="page-title">Overview & Key Performance Indicators</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">High-level summary of RE-SEED project outcomes and coverage</div>', unsafe_allow_html=True)

    # KPI row 1
    emp_pct     = pct(df, 'F1', 'Yes')
    formal_pct  = pct(df, 'F1c', 'Formal')
    female_pct  = pct(df, 'I4', 'Female')
    refugee_pct = round(get(df,'I6').isin(['Syrian','Iraqi','Iranian','Afghani']).sum() / N * 100, 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Total Respondents", N, unit=''), unsafe_allow_html=True)
    c2.markdown(kpi_card("Employment Rate", emp_pct,
                         style='normal' if emp_pct >= 60 else 'warn'), unsafe_allow_html=True)
    c3.markdown(kpi_card("Formal Employment", formal_pct,
                         style='normal' if formal_pct >= 60 else 'gold'), unsafe_allow_html=True)
    c4.markdown(kpi_card("Female Respondents", female_pct, target=35,
                         style='normal' if female_pct >= 35 else 'warn'), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI row 2
    refugee_target_met = refugee_pct >= 65
    sustained_pct = round(
        ((get(df,'F1')=='Yes') &
         (get(df,'F1e').isin(['6 months','More than 6 months']) |
          (pd.to_numeric(get(df,'F1e_1'), errors='coerce') >= 6))
        ).sum() / N * 100, 1)
    cities_n = get(df,'I3').nunique()
    mean_age = round(get(df,'I5').mean(), 1) if get(df,'I5').notna().any() else 'N/A'

    c5, c6, c7, c8 = st.columns(4)
    c5.markdown(kpi_card("Refugee/Displaced", refugee_pct, target=65,
                         style='normal' if refugee_target_met else 'warn'), unsafe_allow_html=True)
    c6.markdown(kpi_card("Sustained 6m+", sustained_pct, target=60,
                         style='normal' if sustained_pct >= 60 else 'gold'), unsafe_allow_html=True)
    c7.markdown(kpi_card("Cities Covered", cities_n, unit='', style='gold'), unsafe_allow_html=True)
    c8.markdown(kpi_card("Mean Age", mean_age, unit=' yrs', style='normal'), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Respondent Profile</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(pie_chart(df, 'I4', 'Gender Distribution'),
                        use_container_width=True)
    with col2:
        st.plotly_chart(pie_chart(df, 'F1', 'Employment Status'),
                        use_container_width=True)
    with col3:
        st.plotly_chart(bar_chart(df, 'I3', 'Respondents by City', color=PRIMARY, orientation='v'),
                        use_container_width=True)

    st.markdown('<div class="section-header">Nationality & Service Mix</div>', unsafe_allow_html=True)
    col4, col5 = st.columns(2)
    with col4:
        st.plotly_chart(bar_chart(df, 'I6', 'Nationality Breakdown', color=GREEN),
                        use_container_width=True)
    with col5:
        # Services received breakdown
        services = {
            'Vocational training': get_service_flag(df, 'Vocational').sum(),
            'Language course':     get_service_flag(df, 'language course').sum(),
            'Advisory support':    get_service_flag(df, 'Legal').sum(),
            'Soft skills':         get_service_flag(df, 'soft skills').sum(),
            'Formalisation':       get_service_flag(df, 'Formalisation').sum(),
        }
        svc_df = pd.DataFrame({'Service': list(services.keys()), 'Count': list(services.values())})
        svc_df['Percent'] = (svc_df['Count'] / N * 100).round(1)
        fig = px.bar(svc_df, x='Count', y='Service', orientation='h',
                     text=svc_df['Percent'].apply(lambda x: f'{x}%'),
                     color_discrete_sequence=[GOLD])
        fig.update_traces(textposition='outside')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(style_chart(fig, 'Services Received'), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: DEMOGRAPHICS & COVERAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Demographics & Coverage":
    st.markdown('<div class="page-title">Demographics & Coverage</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Respondent profile and TPM coverage compliance</div>', unsafe_allow_html=True)

    # Coverage targets
    female_pct  = pct(df, 'I4', 'Female')
    refugee_pct = round(get(df,'I6').isin(['Syrian','Iraqi','Iranian','Afghani']).sum() / N * 100, 1)

    st.markdown('<div class="section-header">TPM Coverage Targets</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Female", female_pct, target=35,
                         style='normal' if female_pct>=35 else 'warn'), unsafe_allow_html=True)
    c2.markdown(kpi_card("Refugee/Displaced", refugee_pct, target=65,
                         style='normal' if refugee_pct>=65 else 'warn'), unsafe_allow_html=True)
    c3.markdown(kpi_card("Syrian", pct(df,'I6','Syrian'), unit='%', style='gold'), unsafe_allow_html=True)
    c4.markdown(kpi_card("With Disability", pct(df,'I15','Yes'), unit='%', style='normal'), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Age and Household</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        age_s = get(df, 'I5').dropna()
        if len(age_s) > 0:
            fig = px.histogram(age_s, nbins=20, color_discrete_sequence=[PRIMARY])
            fig.add_vline(x=age_s.mean(), line_dash='dash', line_color=ACCENT,
                          annotation_text=f'Mean: {age_s.mean():.1f}',
                          annotation_position='top right')
            st.plotly_chart(style_chart(fig, 'Age Distribution',
                                        xaxis_title='Age', yaxis_title='Count'),
                            use_container_width=True)
    with col2:
        st.plotly_chart(bar_chart(df, 'I12', 'Household Size', color=GREEN, orientation='v'),
                        use_container_width=True)

    st.markdown('<div class="section-header">Background Characteristics</div>', unsafe_allow_html=True)
    col3, col4, col5 = st.columns(3)
    with col3:
        st.plotly_chart(bar_chart(df, 'I10', 'Marital Status', color=PRIMARY),
                        use_container_width=True)
    with col4:
        st.plotly_chart(bar_chart(df, 'I18', 'Earthquake Impact', color=ACCENT),
                        use_container_width=True)
    with col5:
        st.plotly_chart(bar_chart(df, 'I15', 'Disability in Household', color=GOLD, orientation='v'),
                        use_container_width=True)

    st.markdown('<div class="section-header">City × Gender Breakdown</div>', unsafe_allow_html=True)
    s1 = get(df,'I3'); s2 = get(df,'I4')
    sub = pd.DataFrame({'City':s1,'Gender':s2}).dropna()
    if len(sub) > 0:
        ct = pd.crosstab(sub['City'], sub['Gender']).reset_index()
        ct_melt = ct.melt(id_vars='City', var_name='Gender', value_name='Count')
        fig = px.bar(ct_melt, x='City', y='Count', color='Gender',
                     color_discrete_sequence=[PRIMARY, GREEN],
                     barmode='group', text='Count')
        fig.update_traces(textposition='outside')
        st.plotly_chart(style_chart(fig, 'Respondents by City and Gender',
                                    yaxis_title='Count'),
                        use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: EMPLOYMENT OUTCOMES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💼 Employment Outcomes":
    st.markdown('<div class="page-title">Employment Outcomes</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Current employment status, formality, work permits, and sustainability</div>', unsafe_allow_html=True)

    emp_pct    = pct(df,'F1','Yes')
    formal_pct = pct(df,'F1c','Formal')
    permit_pct = pct(df[get(df,'F1b') != 'Not applicable — Turkish national'], 'F1b', 'Yes')
    iskur_pct  = pct(df,'F1f','Yes')

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Employed", emp_pct), unsafe_allow_html=True)
    c2.markdown(kpi_card("Formal Employment", formal_pct,
                         style='normal' if formal_pct>=60 else 'gold'), unsafe_allow_html=True)
    c3.markdown(kpi_card("Work Permit (non-Turkish)", permit_pct,
                         style='normal' if permit_pct>=70 else 'gold'), unsafe_allow_html=True)
    c4.markdown(kpi_card("Registered with İŞKUR", iskur_pct, style='gold'), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Disaggregation selector
    st.markdown('<div class="section-header">Employment by Demographic Group</div>', unsafe_allow_html=True)
    disagg = st.selectbox("Disaggregate by:",
                          ["Gender", "Nationality", "City", "Has children", "Protection status"],
                          key='emp_disagg')
    col_map = {"Gender":"I4","Nationality":"I6","City":"I3",
               "Has children":"C1","Protection status":"I7b"}

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(grouped_bar(df, col_map[disagg], 'F1', 'Yes',
                                    f'Employment Rate by {disagg}'),
                        use_container_width=True)
    with col2:
        st.plotly_chart(grouped_bar(df, col_map[disagg], 'F1c', 'Formal',
                                    f'Formal Employment Rate by {disagg}'),
                        use_container_width=True)

    st.markdown('<div class="section-header">Employment Details</div>', unsafe_allow_html=True)
    col3, col4, col5 = st.columns(3)
    with col3:
        st.plotly_chart(bar_chart(df,'F1a','How Job Was Obtained', color=GREEN),
                        use_container_width=True)
    with col4:
        st.plotly_chart(bar_chart(df,'F1e','Employment Duration', color=PRIMARY, orientation='v'),
                        use_container_width=True)
    with col5:
        st.plotly_chart(bar_chart(df,'F1g','Barriers to Employment', color=ACCENT),
                        use_container_width=True)

    st.markdown('<div class="section-header">Job Search Duration</div>', unsafe_allow_html=True)
    col6, col7 = st.columns(2)
    with col6:
        st.plotly_chart(bar_chart(df,'F1a_3','Job Search Duration', color=GOLD, orientation='v'),
                        use_container_width=True)
        # 18-month flag
        fa3 = get(df,'F1a_3')
        over_18 = (fa3 == 'More than 18 months').sum()
        if over_18 > 0:
            st.markdown(f'<div class="warn-box">⚠ <b>{over_18} respondents</b> searched for more than 18 months — legally considered to have left the Turkish labour market.</div>',
                        unsafe_allow_html=True)
    with col7:
        st.plotly_chart(stacked_bar(df,'F1a_3','F1c',
                                    'Job Search Duration × Work Formality'),
                        use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: SERVICES & TRAINING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎓 Services & Training":
    st.markdown('<div class="page-title">Services & Training</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Quality and impact of vocational training, language courses, and advisory support</div>', unsafe_allow_html=True)

    # Service subsamples
    voc_df = df[get_service_flag(df, 'Vocational')]
    adv_df = df[get_service_flag(df, 'Legal')]

    # KPIs
    voc_n   = len(voc_df)
    adv_n   = len(adv_df)
    adv_sat = pct(adv_df,'A5','Satisfied') + pct(adv_df,'A5','Very satisfied')
    voc_imp = round(get(voc_df,'VOC_06').mean(), 2) if voc_n > 0 and get(voc_df,'VOC_06').notna().any() else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Vocational Participants", voc_n, unit='', style='normal'), unsafe_allow_html=True)
    c2.markdown(kpi_card("Advisory Participants", adv_n, unit='', style='normal'), unsafe_allow_html=True)
    c3.markdown(kpi_card("Advisory Satisfaction", round(adv_sat,1),
                         style='normal' if adv_sat>=70 else 'gold'), unsafe_allow_html=True)
    c4.markdown(kpi_card("Mean VOC Impact Score", voc_imp, unit='/5', style='gold'), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔧 Vocational Training", "📋 Advisory Support", "🔍 Compare Services"])

    with tab1:
        if voc_n == 0:
            st.info("No vocational training participants match current filters.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(bar_chart(voc_df,'VOC_01','Training Type', color=PRIMARY),
                                use_container_width=True)
                st.plotly_chart(bar_chart(voc_df,'VOC_04','Training Quality', color=GREEN, orientation='v'),
                                use_container_width=True)
            with col2:
                st.plotly_chart(bar_chart(voc_df,'VOC_02','Training Duration', color=GOLD, orientation='v'),
                                use_container_width=True)
                st.plotly_chart(bar_chart(voc_df,'VOC_07','Certificate Type', color=PRIMARY),
                                use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                voc6 = get(voc_df,'VOC_06').dropna()
                if len(voc6) > 0:
                    fig = px.histogram(voc6, nbins=5, color_discrete_sequence=[PRIMARY],
                                       range_x=[0.5,5.5])
                    fig.add_vline(x=voc6.mean(), line_dash='dash', line_color=ACCENT,
                                  annotation_text=f'Mean: {voc6.mean():.2f}')
                    st.plotly_chart(style_chart(fig,'Training Impact on Employment (1-5)',
                                                xaxis_title='Score', yaxis_title='Count'),
                                    use_container_width=True)
            with col4:
                st.plotly_chart(bar_chart(voc_df,'VOC_09c','Suggested Improvements', color=ACCENT),
                                use_container_width=True)

    with tab2:
        if adv_n == 0:
            st.info("No advisory support participants match current filters.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(bar_chart(adv_df,'A5','Satisfaction with Advisory', color=GREEN, orientation='v'),
                                use_container_width=True)
                st.plotly_chart(bar_chart(adv_df,'A8','Confidence Increase', color=PRIMARY, orientation='v'),
                                use_container_width=True)
            with col2:
                st.plotly_chart(bar_chart(adv_df,'A6','Advisory Relevance', color=GOLD),
                                use_container_width=True)
                st.plotly_chart(bar_chart(adv_df,'A10','Actions Taken After Counselling', color=GREEN),
                                use_container_width=True)
            col3, col4 = st.columns(2)
            with col3:
                st.plotly_chart(bar_chart(adv_df,'A11','Overall Usefulness', color=PRIMARY, orientation='v'),
                                use_container_width=True)
            with col4:
                st.plotly_chart(bar_chart(adv_df,'A3','Service Delivery Method', color=GOLD, orientation='v'),
                                use_container_width=True)

    with tab3:
        st.markdown("**Employment rate by service received:**")
        any_service = (get_service_flag(df, 'Vocational') |
                       get_service_flag(df, 'Legal') |
                       get_service_flag(df, 'language course'))
        svc_emp = {
            'Vocational training': pct(df[get_service_flag(df, 'Vocational')], 'F1', 'Yes'),
            'Advisory support':    pct(df[get_service_flag(df, 'Legal')], 'F1', 'Yes'),
            'Formalisation':       pct(df[get_service_flag(df, 'Formalisation')], 'F1', 'Yes'),
            'No services':         pct(df[~any_service], 'F1', 'Yes'),
        }
        svc_df2 = pd.DataFrame({'Service':list(svc_emp.keys()), 'Employment Rate (%)':list(svc_emp.values())})
        fig = px.bar(svc_df2, x='Service', y='Employment Rate (%)',
                     text=svc_df2['Employment Rate (%)'].apply(lambda x: f'{x}%'),
                     color_discrete_sequence=[PRIMARY, GREEN, NEUTRAL])
        fig.update_traces(textposition='outside')
        st.plotly_chart(style_chart(fig, 'Employment Rate by Service Received',
                                    yaxis_title='Employed (%)'),
                        use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: CHILDCARE & INCLUSION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👶 Childcare & Inclusion":
    st.markdown('<div class="page-title">Childcare & Inclusion</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Childcare barriers, gender intersection, and participation challenges</div>', unsafe_allow_html=True)

    has_ch_pct  = pct(df,'C1','Yes')
    no_care_pct = round(
        ((get(df,'C1')=='Yes') &
         get(df,'C4').astype(str).str.contains('No, I do', na=False)
        ).sum() / N * 100, 1)
    childcare_prev_pct = round(
        get(df,'T6').isin(['Yes, frequently','Occasionally']).sum() / N * 100, 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Has Children", has_ch_pct), unsafe_allow_html=True)
    c2.markdown(kpi_card("No Childcare Access", no_care_pct,
                         style='warn' if no_care_pct>30 else 'normal'), unsafe_allow_html=True)
    c3.markdown(kpi_card("Childcare Prevented Participation", childcare_prev_pct,
                         style='warn' if childcare_prev_pct>20 else 'gold'), unsafe_allow_html=True)
    c4.markdown(kpi_card("Has Children & Unemployed",
                         round(((get(df,'C1')=='Yes') & (get(df,'F1')=='No')).sum()/N*100,1),
                         style='warn'), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Childcare × Gender Analysis</div>', unsafe_allow_html=True)

    # Four-group employment analysis
    groups = {
        'Female\nwith children': df[(get(df,'I4')=='Female') & (get(df,'C1')=='Yes')],
        'Female\nno children':   df[(get(df,'I4')=='Female') & (get(df,'C1')=='No')],
        'Male\nwith children':   df[(get(df,'I4')=='Male') & (get(df,'C1')=='Yes')],
        'Male\nno children':     df[(get(df,'I4')=='Male') & (get(df,'C1')=='No')],
    }
    four_group = pd.DataFrame({
        'Group': list(groups.keys()),
        'N': [len(v) for v in groups.values()],
        'Employment Rate (%)': [pct(v,'F1','Yes') for v in groups.values()],
        'Childcare Burden (%)': [round(get(v,'T6').isin(['Yes, frequently','Occasionally']).sum()/max(len(v),1)*100,1) for v in groups.values()],
    })

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(four_group, x='Group', y='Employment Rate (%)',
                     text=four_group['Employment Rate (%)'].apply(lambda x: f'{x}%'),
                     color='Group', color_discrete_sequence=PALETTE)
        fig.add_hline(y=pct(df,'F1','Yes'), line_dash='dash', line_color=NEUTRAL,
                      annotation_text=f'Overall: {pct(df,"F1","Yes")}%')
        fig.update_traces(textposition='outside', showlegend=False)
        st.plotly_chart(style_chart(fig,'Employment Rate: Gender × Childcare',
                                    yaxis_title='Employed (%)'), use_container_width=True)
    with col2:
        fig2 = px.bar(four_group, x='Group', y='Childcare Burden (%)',
                      text=four_group['Childcare Burden (%)'].apply(lambda x: f'{x}%'),
                      color='Group', color_discrete_sequence=PALETTE)
        fig2.update_traces(textposition='outside', showlegend=False)
        st.plotly_chart(style_chart(fig2,'Childcare Prevented Participation: Gender × Children',
                                    yaxis_title='Childcare burden (%)'), use_container_width=True)

    st.markdown('<div class="section-header">Childcare Details</div>', unsafe_allow_html=True)
    col3, col4, col5 = st.columns(3)
    with col3:
        st.plotly_chart(bar_chart(df,'C4b','Barriers to Childcare Access', color=ACCENT),
                        use_container_width=True)
    with col4:
        st.plotly_chart(bar_chart(df,'C5','Importance of Childcare for Employment',
                                  color=GREEN, orientation='v'), use_container_width=True)
    with col5:
        st.plotly_chart(bar_chart(df,'C6','How Childcare Would Help', color=PRIMARY),
                        use_container_width=True)

    st.markdown('<div class="section-header">Participation Challenges</div>', unsafe_allow_html=True)
    col6, col7 = st.columns(2)
    with col6:
        st.plotly_chart(bar_chart(df,'T4','Participation Challenges', color=ACCENT),
                        use_container_width=True)
    with col7:
        st.plotly_chart(grouped_bar(df,'I4','T6','Yes, frequently',
                                    'Childcare Frequently Prevented Participation by Gender'),
                        use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: ACCOUNTABILITY & SOCIAL COHESION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📣 Accountability & Cohesion":
    st.markdown('<div class="page-title">Accountability & Social Cohesion</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Feedback mechanisms, inclusion, and social cohesion indicators</div>', unsafe_allow_html=True)

    informed_pct = round(get(df,'V1').isin(['Yes, clearly','Yes, I was partially informed']).sum()/N*100,1)
    hotline_pct  = pct(df,'V2','Yes')
    comfort_pct  = round(get(df,'V2a').isin(['Comfortable','Very comfortable','Somewhat comfortable']).sum()/N*100,1)
    equal_pct    = round(get(df,'V3').isin(['Mostly','Fully']).sum()/N*100,1)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Informed About Feedback", informed_pct,
                         style='normal' if informed_pct>=70 else 'gold'), unsafe_allow_html=True)
    c2.markdown(kpi_card("Hotline Awareness", hotline_pct,
                         style='normal' if hotline_pct>=50 else 'warn'), unsafe_allow_html=True)
    c3.markdown(kpi_card("Comfortable Sharing Feedback", comfort_pct,
                         style='normal' if comfort_pct>=60 else 'gold'), unsafe_allow_html=True)
    c4.markdown(kpi_card("Equal Access (Mostly/Fully)", equal_pct,
                         style='normal' if equal_pct>=70 else 'gold'), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Accountability Indicators</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(bar_chart(df,'V1','Informed About Feedback Channels',
                                  color=PRIMARY, orientation='v'), use_container_width=True)
    with col2:
        st.plotly_chart(bar_chart(df,'V2a','Comfort Sharing Feedback',
                                  color=GREEN, orientation='v'), use_container_width=True)
    with col3:
        st.plotly_chart(bar_chart(df,'V3','Equal Access and Treatment',
                                  color=GOLD, orientation='v'), use_container_width=True)

    st.markdown('<div class="section-header">Social Cohesion & Inclusion</div>', unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3)
    with col4:
        st.plotly_chart(bar_chart(df,'V4','Staff Inclusiveness',
                                  color=PRIMARY, orientation='v'), use_container_width=True)
    with col5:
        st.plotly_chart(bar_chart(df,'V5','Social Cohesion Training Impact',
                                  color=GREEN, orientation='v'), use_container_width=True)
    with col6:
        st.plotly_chart(bar_chart(df,'V2c','Free Expression During Activities',
                                  color=GOLD, orientation='v'), use_container_width=True)

    st.markdown('<div class="section-header">Disaggregated Accountability</div>', unsafe_allow_html=True)
    col7, col8 = st.columns(2)
    with col7:
        st.plotly_chart(grouped_bar(df,'I4','V2a','Comfortable',
                                    'Feedback Comfort by Gender'),
                        use_container_width=True)
    with col8:
        st.plotly_chart(grouped_bar(df,'I6','V3','Fully',
                                    'Equal Access (Fully) by Nationality'),
                        use_container_width=True)