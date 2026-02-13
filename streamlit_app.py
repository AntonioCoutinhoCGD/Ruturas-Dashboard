import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/9/9c/CGD_Logo_2017.png"

st.set_page_config(
    page_title="Ruturas Dashboard",
    page_icon=LOGO_URL,
    layout="wide",
)

# Título com logo
c_logo, c_title = st.columns([0.12, 0.88])
with c_logo:
    st.image(LOGO_URL, width=72)
with c_title:
    st.markdown("<h1 style='margin-bottom:0;'>Ruturas Dashboard</h1>", unsafe_allow_html=True)
st.write("Carrega um ficheiro CSV")

# -----------------------------------------------------------------------------
# Helpers / Constantes
# -----------------------------------------------------------------------------
def base_name(colname: str) -> str:
    """Remove sufixos pandas .1, .2 e devolve lowercase trim."""
    return str(colname).strip().split(".", 1)[0].lower()

def normalize_text_pt(s: str) -> str:
    """Remove acentos PT para matching robusto."""
    repl = str.maketrans("áàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÜÇ", "aaaaeeiooouucAAAAEEIOOOUUC")
    return str(s).translate(repl).strip().lower()

def base_norm(colname: str) -> str:
    return normalize_text_pt(base_name(colname))

EXPECTED_BASENORMS = [
    "ruturas vtm", "indisponiveis vtm", "anomalias vtm",
    "ruturas atm", "indisponiveis atm", "anomalias atm",
]

DISPLAY_FONTE = {
    "GERAL": "Geral",
    "Agências": "Agências",
    "Esegur": "Fornecedores",  # display name pedido
}

# -----------------------------------------------------------------------------
# Leitura e normalização segundo o novo layout (Agências + Esegur + Just/Registos)
# -----------------------------------------------------------------------------
def read_uploaded_csv_v2(file):
    """
    Novo CSV: [DATA + MÉTRICAS Agências (VTM/ATM)] + [MÉTRICAS Esegur (VTM/ATM)] +
              [DATA + Justificações] + [DATA + Registos detalhados]
    header=1 para ler a segunda linha como nomes de coluna.
    """
    # tentar UTF-8 -> fallback Latin-1
    try:
        df = pd.read_csv(file, sep=";", header=1, engine="python")
    except UnicodeDecodeError:
        file.seek(0)
        df = pd.read_csv(file, sep=";", header=1, engine="python", encoding="latin-1")

    df.columns = [str(c).strip() for c in df.columns]
    cols = df.columns.tolist()

    # Localizar todas as posições de 'Data'
    bases = [base_name(c) for c in cols]
    data_positions = [i for i, b in enumerate(bases) if b == "data"]

    if len(data_positions) < 2:
        raise ValueError("Estrutura inesperada: esperava pelo menos 2 colunas 'Data'.")

    # ---------------------- BLOCO PRINCIPAL (Agências + Esegur) ----------------------
    start_main = data_positions[0]
    end_main = data_positions[1]
    df_mainblk = df.iloc[:, start_main:end_main].copy()

    # A primeira coluna deve ser 'Data'. Normalizar datas.
    data_col = df_mainblk.columns[0]
    df_mainblk.rename(columns={data_col: "Data"}, inplace=True)
    df_mainblk["Data"] = pd.to_datetime(df_mainblk["Data"], errors="coerce").dt.normalize()
    df_mainblk = df_mainblk.dropna(subset=["Data"]).reset_index(drop=True)

    # Mapear colunas duplicadas: 1ª ocorrência -> Agências; 2ª -> Esegur
    occ_map = {bn: [] for bn in EXPECTED_BASENORMS}
    for c in df_mainblk.columns[1:]:
        bn = base_norm(c)
        if bn in occ_map:
            occ_map[bn].append(c)

    def colmeta_from_basename(bn: str):
        # "ruturas vtm" -> ("VTM","Ruturas")
        parts = bn.split()
        metrica = parts[0].capitalize()      # Ruturas / Indisponiveis / Anomalias
        canal = parts[-1].upper()            # VTM / ATM
        return canal, metrica

    map_agencias = {}
    map_esegur = {}
    for bn, col_list in occ_map.items():
        canal, metrica = colmeta_from_basename(bn)
        if len(col_list) >= 1:
            map_agencias[col_list[0]] = (canal, metrica)
        if len(col_list) >= 2:
            map_esegur[col_list[1]] = (canal, metrica)

    def melt_fonte(df_blk: pd.DataFrame, map_cols: dict, fonte_label: str) -> pd.DataFrame:
        if not map_cols:
            return pd.DataFrame(columns=["Data","Fonte","Canal","Metrica","Valor"])
        keep = ["Data"] + list(map_cols.keys())
        dfx = df_blk[keep].copy()
        dfl = dfx.melt(id_vars=["Data"], var_name="Col", value_name="Valor")
        dfl["Fonte"] = fonte_label
        dfl[["Canal","Metrica"]] = dfl["Col"].apply(lambda c: pd.Series(map_cols[c]))
        dfl.drop(columns=["Col"], inplace=True)
        dfl["Valor"] = pd.to_numeric(dfl["Valor"], errors="coerce").fillna(0)
        return dfl[["Data","Fonte","Canal","Metrica","Valor"]]

    df_ag = melt_fonte(df_mainblk, map_agencias, "Agências")
    df_es = melt_fonte(df_mainblk, map_esegur, "Esegur")

    # Se Esegur não existir no CSV para alguma métrica, criar zeros (mesmas datas/categorias)
    if df_es.empty and not df_ag.empty:
        uniq = df_ag[["Data","Canal","Metrica"]].drop_duplicates()
        uniq["Fonte"] = "Esegur"
        uniq["Valor"] = 0
        df_es = uniq[["Data","Fonte","Canal","Metrica","Valor"]].copy()

    df_daily = pd.concat([df_ag, df_es], ignore_index=True)

    # Adicionar linha "GERAL" (ATM+VTM) por Fonte e Métrica
    df_geral = (
        df_daily.groupby(["Data","Fonte","Metrica"], as_index=False)["Valor"].sum()
                .assign(Canal="GERAL")
                .loc[:, ["Data","Fonte","Canal","Metrica","Valor"]]
    )
    df_daily = pd.concat([df_daily, df_geral], ignore_index=True)

    # ---------------------- BLOCO JUSTIFICAÇÕES (matriz diária) ----------------------
    df_just, just_has_date = None, False
    try:
        start_j = data_positions[1]
        end_j = data_positions[2] if len(data_positions) >= 3 else len(cols)
        df_right = df.iloc[:, start_j:end_j].copy()
        df_right.columns = [str(c).strip() for c in df_right.columns]

        data_col_j = None
        for c in df_right.columns:
            if base_name(c) == "data":
                data_col_j = c
                break

        if data_col_j:
            df_right.rename(columns={data_col_j: "Data"}, inplace=True)
            df_right["Data"] = pd.to_datetime(df_right["Data"], errors="coerce").dt.normalize()
            df_right = df_right.dropna(subset=["Data"])
            for c in [c for c in df_right.columns if c != "Data"]:
                df_right[c] = pd.to_numeric(df_right[c], errors="coerce").fillna(0)
            df_just = df_right.copy()
            just_has_date = True
        else:
            for c in df_right.columns:
                df_right[c] = pd.to_numeric(df_right[c], errors="coerce").fillna(0)
            df_just = df_right.copy()
            just_has_date = False
    except Exception:
        df_just, just_has_date = None, False

    # ---------------------- BLOCO EVENTOS DETALHADOS ----------------------
    df_events = None
    if len(data_positions) >= 3:
        start_e = data_positions[2]
        df_events_blk = df.iloc[:, start_e:].copy()

        rename_map = {}
        for c in df_events_blk.columns:
            n = normalize_text_pt(c)
            if n == "data":
                rename_map[c] = "Data"
            elif n.startswith("hora"):
                rename_map[c] = "Hora_" + c.split()[1] if len(c.split()) > 1 else "Hora"
            elif "duracao" in n:
                rename_map[c] = "Duracao_" + str(len(rename_map))
            elif n in ("agencia/empresa", "agencia/ empresa", "agenciaempresa", "agencia_empresa"):
                rename_map[c] = "AgenciaEmpresa"
            elif n.startswith("maquina"):
                rename_map[c] = "Maquina"
            elif n.startswith("justific"):
                rename_map[c] = "Justificacao"

        dfe = df_events_blk.rename(columns=rename_map)
        keep = [c for c in ["Data","AgenciaEmpresa","Maquina","Justificacao"] if c in dfe.columns]
        if keep:
            dfe = dfe[keep].copy()
            if "Data" in dfe.columns:
                dfe["Data"] = pd.to_datetime(dfe["Data"], errors="coerce").dt.normalize()

            # Classificação da Fonte a partir de AgenciaEmpresa
            if "AgenciaEmpresa" in dfe.columns:
                dfe["Fonte"] = np.where(
                    dfe["AgenciaEmpresa"].fillna("").str.strip().str.lower() == "esegur",
                    "Esegur", "Agências"
                )
            df_events = dfe.dropna(how="all")
        else:
            df_events = None

    return df_daily, df_just, just_has_date, df_events

# -----------------------------------------------------------------------------
# Upload
# -----------------------------------------------------------------------------
file = st.file_uploader("Carrega aqui", type=["csv"])
if not file:
    st.info("A aguardar ficheiro…")
    st.stop()

try:
    df_daily, df_just, just_has_date, df_events = read_uploaded_csv_v2(file)
except Exception as e:
    st.error(f"Erro a ler o CSV: {e}")
    st.stop()

st.success("Dados carregados com sucesso.")

# -----------------------------------------------------------------------------
# Utilitários de KPIs
# -----------------------------------------------------------------------------
last_date = df_daily["Data"].max()

def ma7_from_series(s: pd.Series) -> float:
    """Média móvel de 7 (excluindo o dia de referência; já deve ser filtrado)."""
    return float(s.tail(7).mean()) if not s.empty else float("nan")

def today_and_ma7(df_daily: pd.DataFrame, fonte_tab: str, canal: str, metrica: str, ref_date: pd.Timestamp):
    """
    Devolve (valor_hoje, m7) para a combinação pedida.
    fonte_tab: "GERAL" (soma das fontes) | "Agências" | "Esegur"
    canal: "ATM" | "VTM" | "GERAL"
    metrica: "Ruturas" | "Indisponiveis" | "Anomalias"
    """
    if fonte_tab == "GERAL":
        # soma das fontes
        df_today = df_daily[
            (df_daily["Data"] == ref_date) &
            (df_daily["Canal"] == canal) &
            (df_daily["Metrica"] == metrica)
        ]
        v_hoje = float(df_today.groupby("Data")["Valor"].sum().sum())

        df_hist = df_daily[
            (df_daily["Data"] < ref_date) &
            (df_daily["Canal"] == canal) &
            (df_daily["Metrica"] == metrica)
        ]
        s_hist = df_hist.groupby("Data")["Valor"].sum().sort_index()
        m7 = ma7_from_series(s_hist)
    else:
        df_today = df_daily[
            (df_daily["Data"] == ref_date) &
            (df_daily["Fonte"] == fonte_tab) &
            (df_daily["Canal"] == canal) &
            (df_daily["Metrica"] == metrica)
        ]
        v_hoje = float(df_today["Valor"].sum())

        df_hist = df_daily[
            (df_daily["Data"] < ref_date) &
            (df_daily["Fonte"] == fonte_tab) &
            (df_daily["Canal"] == canal) &
            (df_daily["Metrica"] == metrica)
        ].sort_values("Data")
        m7 = ma7_from_series(df_hist["Valor"])
    return v_hoje, m7

def render_main_kpi(metrica: str, v_geral: float, m7_geral: float):
    """Mostra KPI principal (GERAL) com delta vs M7 (inverse)."""
    delta = None if pd.isna(m7_geral) else f"{'+' if (v_geral - m7_geral) >= 0 else ''}{int(round(v_geral - m7_geral))}"
    st.metric(metrica, int(v_geral), delta=delta, delta_color="inverse" if delta else "off")

# -----------------------------------------------------------------------------
# KPIs — Destaque GERAL (soma ATM+VTM) + detalhe estético ATM/VTM em expander
# -----------------------------------------------------------------------------
st.header(f"Indicadores — {last_date.date().isoformat()}")

tab_geral, tab_ag, tab_for = st.tabs([DISPLAY_FONTE["GERAL"], DISPLAY_FONTE["Agências"], DISPLAY_FONTE["Esegur"]])

# Em cada tab: mostrar KPIs da soma (Canal="GERAL") como destaque
for tab, fonte_tab in zip([tab_geral, tab_ag, tab_for], ["GERAL", "Agências", "Esegur"]):
    with tab:
        if fonte_tab == "GERAL":
            sub_today = df_daily[(df_daily["Data"] == last_date) & (df_daily["Canal"] == "GERAL")]
        else:
            sub_today = df_daily[
                (df_daily["Data"] == last_date) &
                (df_daily["Fonte"] == fonte_tab) &
                (df_daily["Canal"] == "GERAL")
            ]
        if sub_today.empty:
            st.info(f"Sem dados para {DISPLAY_FONTE[fonte_tab]} no dia {last_date.date().isoformat()}.")
            continue

        # KPIs principais (sem "— GERAL" nos títulos)
        cols = st.columns(3)
        for metrica, cc in zip(["Ruturas","Indisponiveis","Anomalias"], cols):
            v_geral, m7_geral = today_and_ma7(df_daily, fonte_tab, "GERAL", metrica, last_date)
            with cc:
                render_main_kpi(metrica, v_geral, m7_geral)

        # Detalhe estético por canal (ATM/VTM) num expander
        with st.expander("Detalhe por canal (ATM / VTM)"):
            c1, c2, c3 = st.columns(3)
            for metrica, cont in zip(["Ruturas","Indisponiveis","Anomalias"], [c1, c2, c3]):
                v_atm, m7_atm = today_and_ma7(df_daily, fonte_tab, "ATM", metrica, last_date)
                v_vtm, m7_vtm = today_and_ma7(df_daily, fonte_tab, "VTM", metrica, last_date)
                with cont:
                    st.caption(f"**{metrica}**")
                    d_atm = None if pd.isna(m7_atm) else f"{'+' if (v_atm - m7_atm) >= 0 else ''}{int(round(v_atm - m7_atm))}"
                    d_vtm = None if pd.isna(m7_vtm) else f"{'+' if (v_vtm - m7_vtm) >= 0 else ''}{int(round(v_vtm - m7_vtm))}"
                    st.metric("ATM", int(v_atm), delta=d_atm, delta_color="inverse" if d_atm else "off")
                    st.metric("VTM", int(v_vtm), delta=d_vtm, delta_color="inverse" if d_vtm else "off")
        st.markdown("---")

# -----------------------------------------------------------------------------
# Evolução diária — escolher Fonte (Geral/Agências/Fornecedores) e Métrica,
# mostrar três linhas: ATM, VTM e GERAL (soma)
# -----------------------------------------------------------------------------
st.header("Evolução diária")
fonte_sel_label = st.radio("Fonte", [DISPLAY_FONTE["GERAL"], DISPLAY_FONTE["Agências"], DISPLAY_FONTE["Esegur"]],
                           horizontal=True, index=0)
# map back to internal labels
label_to_internal = {v: k for k, v in DISPLAY_FONTE.items()}
fonte_sel = label_to_internal[fonte_sel_label]
met_sel = st.selectbox("Métrica", ["Ruturas","Indisponiveis","Anomalias"], index=0)

if fonte_sel == "GERAL":
    # Somar fontes para ATM/VTM e manter GERAL (soma final)
    base = df_daily[(df_daily["Metrica"] == met_sel) & (df_daily["Canal"].isin(["ATM","VTM","GERAL"]))].copy()
    # ATM/VTM: somar por data e canal; GERAL já existe como soma por fonte (vamos somar também por data)
    df_atm_vtm = (base[base["Canal"].isin(["ATM","VTM"])]
                  .groupby(["Data", "Canal"], as_index=False)["Valor"].sum())
    df_all = (base[base["Canal"] == "GERAL"]
              .groupby(["Data"], as_index=False)["Valor"].sum().assign(Canal="GERAL"))
    df_chart = pd.concat([df_atm_vtm, df_all], ignore_index=True)
else:
    df_chart = df_daily[
        (df_daily["Fonte"] == fonte_sel)
        & (df_daily["Metrica"] == met_sel)
        & (df_daily["Canal"].isin(["ATM","VTM","GERAL"]))
    ].groupby(["Data","Canal"], as_index=False)["Valor"].sum()

if df_chart.empty:
    st.info("Sem dados para o filtro escolhido.")
else:
    chart = (
        alt.Chart(df_chart)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("Data:T", title="Data", axis=alt.Axis(format="%Y-%m-%d")),
            y=alt.Y("Valor:Q", title="Valor"),
            color=alt.Color("Canal:N", title=None, scale=alt.Scale(scheme="tableau10")),
            tooltip=[alt.Tooltip("Data:T", title="Data", format="%Y-%m-%d"),
                     alt.Tooltip("Canal:N", title="Canal"),
                     alt.Tooltip("Valor:Q", title=met_sel, format=".0f")]
        ).properties(height=340)
    )
    st.altair_chart(chart, use_container_width=True)

# -----------------------------------------------------------------------------
# JUSTIFICAÇÕES — Matriz diária (geral) + Gráfico (Agências) via registos
# -----------------------------------------------------------------------------
st.header("Justificações")

# 1) Matriz diária (geral)
if df_just is None or df_just.empty:
    st.info("Sem dados de justificações (matriz diária).")
else:
    sem_col_candidates = [c for c in df_just.columns if normalize_text_pt(c).startswith("sem justific")]
    sem_col = sem_col_candidates[0] if sem_col_candidates else None

    # Top 2 do último dia (exclui 'Sem justificação')
    st.subheader("Top 2 — último dia (geral)")
    if "Data" in df_just.columns:
        last_date_just = df_just["Data"].max()
        df_last = df_just[df_just["Data"] == last_date_just].copy()
        cand_cols = [c for c in df_last.columns if c != "Data" and c != sem_col]
        if df_last.empty or not cand_cols:
            st.info("Sem dados de justificações para o último dia.")
        else:
            s_vals = df_last[cand_cols].iloc[0].astype(float)
            top2 = s_vals.sort_values(ascending=False).head(2)
            colA, colB = st.columns(2)
            with colA:
                st.metric(top2.index[0], int(top2.iloc[0]))
            with colB:
                if len(top2) > 1:
                    st.metric(top2.index[1], int(top2.iloc[1]))
            with st.expander("Ver todas as categorias (último dia)"):
                st.dataframe(
                    s_vals.sort_values(ascending=False).reset_index().rename(columns={"index":"Categoria",0:"Valor"}),
                    use_container_width=True, hide_index=True
                )

    # Acumulado por período (exclui 'Sem justificação')
    st.subheader("Acumulado por período (geral)")
    periodo = st.selectbox("Período", ["1 semana", "Mês", "Ano", "1-3 Anos"], index=1, key="per_just")
    days_map = {"1 semana": 7, "Mês": 30, "Ano": 365, "1-3 Anos": 365*3}
    dias = days_map[periodo]
    if "Data" in df_just.columns:
        end_date = df_just["Data"].max().normalize()
        start_date = end_date - pd.Timedelta(days=dias-1)
        dfj_win = df_just[(df_just["Data"] >= start_date) & (df_just["Data"] <= end_date)].copy()
        st.caption(f"Período: {start_date.date().isoformat()} a {end_date.date().isoformat()} ({periodo})")
    else:
        dfj_win = df_just.copy()
        st.caption("Período não disponível")

    cols_sum = [c for c in dfj_win.columns if c != "Data" and c != sem_col]
    total_just = dfj_win[cols_sum].sum().sort_values(ascending=False)
    st.bar_chart(total_just)

# 2) Gráfico — Justificações só das Agências (registos detalhados) — remover Esegur e tabelas
st.subheader("Justificações — Agências (registos detalhados)")
if (df_events is None) or df_events.empty or ("Justificacao" not in df_events.columns):
    st.info("Sem registos detalhados para calcular este gráfico.")
else:
    col1, col2 = st.columns(2)
    with col1:
        periodo_ev = st.selectbox("Período", ["Tudo", "1 semana", "Mês", "Ano", "1-3 Anos"], index=2, key="per_ev_ag_only")
    with col2:
        excluir_sem = st.checkbox("Excluir 'Sem justificação'", value=True, key="excluir_sem_ag_only")

    def filtro_periodo(df, periodo_label: str):
        days_map2 = {"1 semana": 7, "Mês": 30, "Ano": 365, "1-3 Anos": 365*3, "Tudo": None}
        dias2 = days_map2[periodo_label]
        if (dias2 is None) or ("Data" not in df.columns) or (df["Data"].dropna().empty):
            return df
        end_d = df["Data"].max().normalize()
        start_d = end_d - pd.Timedelta(days=dias2-1)
        return df[(df["Data"] >= start_d) & (df["Data"] <= end_d)].copy()

    ev = df_events.copy()
    # Filtrar apenas Agências
    ev = ev[ev["Fonte"] == "Agências"]

    # Aplicar janela temporal
    ev = filtro_periodo(ev, periodo_ev)

    # >>> Correção robusta para "Sem justificação" (evitar KeyError/booleans)
    if excluir_sem and "Justificacao" in ev.columns:
        ev = ev[~ev["Justificacao"].fillna("").astype(str).apply(normalize_text_pt).eq("sem justificacao")]

    if ev.empty:
        st.info("Sem registos para os filtros.")
    else:
        top_by_ag = (
            ev.groupby(["Justificacao"], dropna=False).size()
              .rename("Ocorrencias").reset_index()
              .sort_values("Ocorrencias", ascending=False).head(10)
        )
        chart_ag = (
            alt.Chart(top_by_ag)
            .mark_bar(color="#4C78A8")
            .encode(
                x=alt.X("Ocorrencias:Q", title="Ocorrências"),
                y=alt.Y("Justificacao:N", title=None, sort="-x"),
                tooltip=["Justificacao:N", alt.Tooltip("Ocorrencias:Q", title="Ocorrências")]
            ).properties(height=280)
        )
        st.altair_chart(chart_ag, use_container_width=True)

# -----------------------------------------------------------------------------
# Top 5 piores agências (nº de ocorrências) — exclui Fornecedores (Esegur)
# -----------------------------------------------------------------------------
st.header("Top 5 piores agências (nº de ocorrências)")

if (df_events is None) or df_events.empty or ("AgenciaEmpresa" not in df_events.columns):
    st.info("Sem registos detalhados com 'AgenciaEmpresa'.")
else:
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        periodo_top = st.selectbox("Período", ["Tudo", "1 semana", "Mês", "Ano", "1-3 Anos"], index=2, key="periodo_top")
    with col_t2:
        just_opts2 = ["Todas"] + sorted([j for j in df_events["Justificacao"].dropna().unique()]) if "Justificacao" in df_events.columns else ["Todas"]
        just_sel2 = st.selectbox("Justificação", just_opts2, index=0, key="just_top")

    def filtro_periodo_top(df, periodo_label: str):
        days_map = {"1 semana": 7, "Mês": 30, "Ano": 365, "1-3 Anos": 365*3, "Tudo": None}
        dias = days_map[periodo_label]
        if dias is None or "Data" not in df.columns or df["Data"].dropna().empty:
            return df
        end_date = df["Data"].max().normalize()
        start_date = end_date - pd.Timedelta(days=dias-1)
        return df[(df["Data"] >= start_date) & (df["Data"] <= end_date)].copy()

    top_df = df_events.copy()
    # Excluir Fornecedores (Esegur)
    top_df = top_df[top_df["Fonte"] == "Agências"]
    top_df = filtro_periodo_top(top_df, periodo_top)
    if just_sel2 != "Todas" and "Justificacao" in top_df.columns:
        top_df = top_df[top_df["Justificacao"] == just_sel2]

    if top_df.empty:
        st.info("Sem dados para o filtro selecionado.")
    else:
        topN = (
            top_df.dropna(subset=["AgenciaEmpresa"])
                  .groupby("AgenciaEmpresa", dropna=False)
                  .size()
                  .sort_values(ascending=False)
                  .head(5)
                  .rename("Ocorrencias")
                  .reset_index()
        )

        chart_top = (
            alt.Chart(topN)
            .mark_bar(color="#E76F51")
            .encode(
                x=alt.X("Ocorrencias:Q", title="Ocorrências"),
                y=alt.Y("AgenciaEmpresa:N", title="Agência", sort="-x"),
                tooltip=["AgenciaEmpresa:N", alt.Tooltip("Ocorrencias:Q", title="Ocorrências")]
            ).properties(height=240)
        )
        labels = (
            alt.Chart(topN)
            .mark_text(align="left", dx=4, color="#333")
            .encode(x="Ocorrencias:Q", y=alt.Y("AgenciaEmpresa:N", sort="-x"), text="Ocorrencias:Q")
        )
        st.altair_chart((chart_top + labels), use_container_width=True)
        with st.expander("Ver tabela"):
            st.dataframe(topN, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Recomendações (bullet points)
# -----------------------------------------------------------------------------
st.header("Recomendações")

try:
    recs = []
    # 1) KPI vs MA7 para cada Fonte no GERAL (ATM+VTM somados)
    last_rows = df_daily[(df_daily["Data"] == last_date) & (df_daily["Canal"] == "GERAL")]
    for fonte in ["Agências","Esegur"]:
        for metrica in ["Ruturas","Indisponiveis","Anomalias"]:
            hoje = float(last_rows.loc[(last_rows["Fonte"] == fonte) & (last_rows["Metrica"] == metrica), "Valor"].sum())
            hist = df_daily[(df_daily["Fonte"] == fonte) & (df_daily["Canal"] == "GERAL") & (df_daily["Metrica"] == metrica)]
            media7 = ma7_from_series(hist.loc[hist["Data"] < last_date, "Valor"])
            if not pd.isna(media7):
                if hoje > media7:
                    recs.append(f"**{DISPLAY_FONTE[fonte]} — {metrica}** acima da M7 ({int(hoje)} vs {int(round(media7))}). Reforçar diagnóstico e mitigação.")
                else:
                    recs.append(f"**{DISPLAY_FONTE[fonte]} — {metrica}** ≤ M7 ({int(hoje)} ≤ {int(round(media7))}). Manter práticas e monitorizar.")

    # 2) Categoria mais incidente por Fonte (registos detalhados)
    if (df_events is not None) and ("Justificacao" in df_events.columns):
        for fonte in ["Agências","Esegur"]:
            sub = df_events[df_events["Fonte"] == fonte]
            if not sub.empty:
                s = sub.groupby("Justificacao").size().sort_values(ascending=False)
                if not s.empty:
                    top_cat = s.index[0]
                    recs.append(f"**{DISPLAY_FONTE[fonte]} — Justificação principal**: _{top_cat}_. Endereçar ações (processo, formação, manutenção).")

    if recs:
        st.markdown("\n".join(f"- {r}" for r in recs))
    else:
        st.info("Sem recomendações automáticas no momento (dados insuficientes).")

except Exception as e:
    st.info(f"Não foi possível gerar recomendações automáticas ({e}).")

# -----------------------------------------------------------------------------
# Downloads
# -----------------------------------------------------------------------------
st.header("Download dos dados")

# Tabela diária normalizada (Fonte x Canal x Métrica)
csv_daily = df_daily.sort_values(["Data","Fonte","Canal","Metrica"]).to_csv(index=False, sep=";").encode("utf-8-sig")
st.download_button("Baixar CSV (diário — Fonte/Canal/Métrica)", csv_daily, file_name="ruturas_diario_fonte_canal.csv", mime="text/csv")

# Justificações (matriz diária)
if (df_just is not None) and (not df_just.empty):
    csv_just = df_just.to_csv(index=False, sep=";").encode("utf-8-sig")
    st.download_button("Baixar CSV (justificações — matriz)", csv_just, file_name="ruturas_justificacoes_matriz.csv", mime="text/csv")

# Registos detalhados normalizados
if (df_events is not None) and (not df_events.empty):
    csv_ev = df_events.to_csv(index=False, sep=";").encode("utf-8-sig")
    st.download_button("Baixar CSV (registos detalhados)", csv_ev, file_name="ruturas_registos_detalhados.csv", mime="text/csv")