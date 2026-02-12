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
    page_icon=LOGO_URL,   # ícone da página = logotipo da Caixa
    layout="wide",
)

# Título com logo ao lado
c_logo, c_title = st.columns([0.12, 0.88])
with c_logo:
    st.image(LOGO_URL, width=72)
with c_title:
    st.markdown("<h1 style='margin-bottom:0;'>Ruturas Dashboard</h1>", unsafe_allow_html=True)

# Texto minimal
st.write("Carrega um ficheiro CSV")

# -----------------------------------------------------------------------------
# Helpers / Constantes
# -----------------------------------------------------------------------------
MAIN_COLS = ["Data", "Ruturas", "Indisponiveis", "Anomalias"]

def base_name(colname: str) -> str:
    """Nome base da coluna (remove sufixos pandas .1, .2), trim e minúsculas."""
    return str(colname).strip().split(".", 1)[0].lower()

def normalize_text_pt(s: str) -> str:
    """Remove acentos PT para matching robusto."""
    repl = str.maketrans("áàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÜÇ", "aaaaeeiooouucAAAAEEIOOOUUC")
    return str(s).translate(repl).strip().lower()

def find_first_col_by_base(cols, target_base: str) -> str | None:
    tb = target_base.lower()
    for c in cols:
        if base_name(c) == tb:
            return c
    return None

def normalize_main(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza a tabela principal (Data, Ruturas, Indisponiveis, Anomalias),
    garantindo tipos corretos, retirando horas e preenchendo zeros para colunas em falta.
    """
    cols = df.columns.tolist()
    data_col = find_first_col_by_base(cols, "data")
    if data_col is None:
        raise ValueError("Falta a coluna 'Data' na tabela principal.")

    keep = [data_col]
    for c in ["Ruturas","Indisponiveis","Anomalias"]:
        if c in cols:
            keep.append(c)

    out = df[keep].copy()
    out.rename(columns={data_col: "Data"}, inplace=True)

    for c in ["Ruturas","Indisponiveis","Anomalias"]:
        if c not in out.columns:
            out[c] = 0

    out["Data"] = pd.to_datetime(out["Data"], errors="coerce").dt.normalize()
    for c in ["Ruturas","Indisponiveis","Anomalias"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    out = out.dropna(subset=["Data"]).sort_values("Data").reset_index(drop=True)
    return out

def normalize_just_block(df_right: pd.DataFrame):
    """
    Normaliza o bloco central de justificações (Data + categorias).
    Aceita qualquer número de categorias; datas sem horas.
    """
    if df_right is None or df_right.empty:
        return None, False

    dfj = df_right.copy()
    dfj.columns = [str(c).strip() for c in dfj.columns]

    data_col = find_first_col_by_base(dfj.columns, "data")
    if data_col is None:
        # sem Data -> não conseguimos distribuir por dia
        for c in dfj.columns:
            dfj[c] = pd.to_numeric(dfj[c], errors="coerce").fillna(0)
        if "Registo" not in dfj.columns:
            dfj.insert(0, "Registo", np.arange(1, len(dfj)+1))
        return dfj, False

    if data_col != "Data":
        dfj.rename(columns={data_col: "Data"}, inplace=True)

    dfj["Data"] = pd.to_datetime(dfj["Data"], errors="coerce").dt.normalize()
    dfj = dfj.dropna(subset=["Data"])

    cat_cols = [c for c in dfj.columns if c != "Data"]
    for c in cat_cols:
        dfj[c] = pd.to_numeric(dfj[c], errors="coerce").fillna(0)
    return dfj, True

def normalize_events_block(df_events: pd.DataFrame) -> pd.DataFrame | None:
    """
    Normaliza o bloco à direita (Registos detalhados): Data, Hora, Agencia, Justificação.
    (Data normalizada, Hora convertida para inteiro 0..23 em 'Hora_int')
    """
    if df_events is None or df_events.empty:
        return None

    # Mapear nomes (tolerante a espaços/acentos/capitalização)
    rename_map = {}
    for c in df_events.columns:
        n = normalize_text_pt(c)
        if n == "data":
            rename_map[c] = "Data"
        elif n == "hora":
            rename_map[c] = "Hora"
        elif n == "agencia":
            rename_map[c] = "Agencia"
        elif n.startswith("justific"):
            rename_map[c] = "Justificacao"
    dfe = df_events.rename(columns=rename_map)

    keep = [c for c in ["Data","Hora","Agencia","Justificacao"] if c in dfe.columns]
    if not keep:
        return None

    dfe = dfe[keep].copy()
    if "Data" in dfe.columns:
        dfe["Data"] = pd.to_datetime(dfe["Data"], errors="coerce").dt.normalize()

    # Converter Hora para HH:MM:SS e extrair hora inteira 0..23
    if "Hora" in dfe.columns:
        hora_dt = pd.to_datetime(dfe["Hora"], errors="coerce", format="%H:%M:%S")
        bad = hora_dt.isna() & dfe["Hora"].notna()
        if bad.any():
            hora_dt.loc[bad] = pd.to_datetime("1900-01-01 " + dfe.loc[bad, "Hora"].astype(str), errors="coerce")
        dfe["Hora_int"] = hora_dt.dt.hour

    return dfe.dropna(how="all")

def find_sem_just_col(cols) -> str | None:
    targets = ["sem justificacao", "sem justificação"]
    for c in cols:
        n = normalize_text_pt(c)
        if n in targets or n.startswith("sem justific"):
            return c
    return None

# -----------------------------------------------------------------------------
# Leitura robusta do upload (CSV com 3 blocos na mesma sheet)
# -----------------------------------------------------------------------------
def read_uploaded_csv(file):
    """
    Lê CSV (sep=';', header=1) com a estrutura:
      [MAIN: Data,Ruturas,Indisponiveis,Anomalias] ; [JUST: Data + categorias...] ; [EVENTS: Data,Hora,Agencia,Justificação]
    Detecta automaticamente as 3 posições de 'Data' (1ª=main, 2ª=just, 3ª=events).
    """
    # tentar UTF-8 → fallback Latin-1
    try:
        df = pd.read_csv(file, sep=";", header=1, engine="python")
    except UnicodeDecodeError:
        file.seek(0)
        df = pd.read_csv(file, sep=";", header=1, engine="python", encoding="latin-1")

    df.columns = [str(c).strip() for c in df.columns]
    cols = df.columns.tolist()

    # Localizar todas as colunas cujo "base" é 'data' (ex.: 'Data', 'Data.1', 'Data.2')
    bases = [base_name(c) for c in cols]
    data_positions = [i for i, b in enumerate(bases) if b == "data"]

    # MAIN — usar a 1ª 'Data' + colunas conhecidas
    left_data_col = find_first_col_by_base(cols, "data")
    main_candidates = [left_data_col] if left_data_col else []
    for c in ["Ruturas","Indisponiveis","Anomalias"]:
        if c in cols:
            main_candidates.append(c)
    df_main = normalize_main(df[main_candidates]) if main_candidates else None

    # JUST — começa na 2ª 'Data' (se existir) e termina na 3ª 'Data' (exclusivo)
    df_just, just_has_date = None, False
    if len(data_positions) >= 2:
        start_j = data_positions[1]
        end_j = data_positions[2] if len(data_positions) >= 3 else len(cols)
        df_right = df.iloc[:, start_j:end_j]
        df_just, just_has_date = normalize_just_block(df_right)

    # EVENTS — começa na 3ª 'Data' (se existir) até ao fim
    df_events = None
    if len(data_positions) >= 3:
        start_e = data_positions[2]
        df_events_blk = df.iloc[:, start_e:]
        df_events = normalize_events_block(df_events_blk)

    return df_main, df_just, just_has_date, df_events

# -----------------------------------------------------------------------------
# Upload
# -----------------------------------------------------------------------------
file = st.file_uploader("Carrega aqui", type=["csv"])
if not file:
    st.info("A aguardar ficheiro…")
    st.stop()

df_main, df_just, just_has_date, df_events = read_uploaded_csv(file)
if df_main is None:
    st.error("Não consegui ler a tabela principal (Data, Ruturas, Indisponiveis, Anomalias).")
    st.stop()

st.success("Dados carregados com sucesso.")

# -----------------------------------------------------------------------------
# KPIs — mostrar a DATA e delta simples (+N / -N) vs MA7 (exclui o dia atual)
# -----------------------------------------------------------------------------
last_date_main = df_main["Data"].max()
st.header(f"Indicadores — {last_date_main.date().isoformat()}")

def ma7_excluindo_hoje(series_col: pd.Series, ref_date: pd.Timestamp) -> float:
    hist = df_main.loc[df_main["Data"] < ref_date, series_col.name]
    return float(hist.tail(7).mean()) if not hist.empty else float("nan")

last_row = df_main.loc[df_main["Data"] == last_date_main].iloc[-1]

k1, k2, k3 = st.columns(3)
for col, metric in zip([k1, k2, k3], ["Ruturas", "Indisponiveis", "Anomalias"]):
    hoje = float(last_row[metric]) if pd.notna(last_row[metric]) else 0.0
    media7 = ma7_excluindo_hoje(df_main[metric], last_date_main)
    if pd.isna(media7):
        delta_txt = ""
        delta_color = "off"
    else:
        dif = hoje - media7
        delta_txt = f"{'+' if dif >= 0 else ''}{dif:.0f}"  # apenas +/-N
        # 'inverse' = verde quando delta <= 0 (melhor), vermelho quando > 0
        delta_color = "inverse"
    with col:
        st.metric(metric, f"{int(hoje)}", delta=delta_txt, delta_color=delta_color)

# -----------------------------------------------------------------------------
# Evolução diária — linhas limpas; valor só em hover
# -----------------------------------------------------------------------------
st.header("Evolução diária")

df_chart = df_main.melt(
    id_vars=["Data"],
    value_vars=["Ruturas", "Indisponiveis", "Anomalias"],
    var_name="Categoria",
    value_name="Valor"
)

hover = alt.selection_point(fields=["Data", "Categoria"], nearest=True, on="mouseover", empty="none")

base = alt.Chart(df_chart).encode(
    x=alt.X("Data:T", title="Data", axis=alt.Axis(format="%Y-%m-%d")),
    y=alt.Y("Valor:Q", title="Valor"),
    color=alt.Color("Categoria:N", legend=alt.Legend(title=None))
)

lines = base.mark_line(strokeWidth=2)
points_on_hover = base.mark_point(size=60).transform_filter(hover)

chart = (lines + points_on_hover).encode(
    tooltip=[
        alt.Tooltip("Data:T", title="Data", format="%Y-%m-%d"),
        alt.Tooltip("Categoria:N", title="Série"),
        alt.Tooltip("Valor:Q", title="Valor", format=".0f")
    ]
).add_params(hover).properties(height=340)

st.altair_chart(chart, use_container_width=True)

# -----------------------------------------------------------------------------
# JUSTIFICAÇÕES — Destaque selecionável + Acumulado por período (exclui "Sem justificação")
# -----------------------------------------------------------------------------
st.header("Justificações")

if df_just is None or df_just.empty:
    st.info("Sem dados de justificações.")
else:
    # ---------------------------
    # Destaque de uma categoria (um único KPI com delta por baixo do valor)
    # ---------------------------
    categorias = [c for c in df_just.columns if c != "Data"]
    default_cat = find_sem_just_col(categorias) or (categorias[0] if categorias else None)

    st.subheader("Destaque de uma categoria")
    selected_cat = st.selectbox(
        "Seleciona a categoria para analisar",
        options=categorias,
        index=categorias.index(default_cat) if default_cat in categorias else 0
    )

    def ma7_cat_excl_hoje(dfj: pd.DataFrame, cat: str, ref_date: pd.Timestamp) -> float:
        hist = dfj.loc[dfj["Data"] < ref_date, cat]
        return float(hist.tail(7).mean()) if not hist.empty else float("nan")

    if selected_cat:
        last_date_just = df_just["Data"].max()
        df_last_sel = df_just[df_just["Data"] == last_date_just].copy()
        last_val = int(float(df_last_sel[selected_cat].iloc[0])) if not df_last_sel.empty else 0
        ma7_sel = ma7_cat_excl_hoje(df_just, selected_cat, last_date_just)
        dif = None if pd.isna(ma7_sel) else (last_val - ma7_sel)

        cols_sem = st.columns(2)
        with cols_sem[0]:
            st.metric(
                f"{selected_cat} (dia {last_date_just.date().isoformat()})",
                last_val,
                delta=(f"{'+' if dif >= 0 else ''}{int(dif)}" if dif is not None else None),
                delta_color="inverse" if dif is not None else "off"
            )
        with cols_sem[1]:
            st.write("")  # vazio propositado para manter o layout

        # Série diária da categoria selecionada
        st.markdown(f"Distribuição diária de '{selected_cat}'")
        c_daily = (
            alt.Chart(df_just.rename(columns={selected_cat: "ValorSel"}))
            .mark_bar(color="#004B87")
            .encode(
                x=alt.X("Data:T", title="Data", axis=alt.Axis(format="%Y-%m-%d")),
                y=alt.Y("ValorSel:Q", title="Contagem"),
                tooltip=[alt.Tooltip("Data:T", title="Data", format="%Y-%m-%d"),
                         alt.Tooltip("ValorSel:Q", title=selected_cat)]
            ).properties(height=220)
        )
        st.altair_chart(c_daily, use_container_width=True)

    st.markdown("---")

    # ---------------------------
    # Justificações do último dia — Top 2 (exceto "Sem justificação"), resto oculto
    # ---------------------------
    st.subheader("Top 2 - Justificações")

    last_date_just = df_just["Data"].max()
    df_last = df_just[df_just["Data"] == last_date_just].copy()

    # Excluir 'Data' e 'Sem justificação' da seleção
    sem_col = find_sem_just_col(df_just.columns)
    cand_cols = [c for c in df_last.columns if c != "Data" and c != sem_col]

    def ma7_cat(cat: str) -> float:
        hist = df_just.loc[df_just["Data"] < last_date_just, cat]
        return float(hist.tail(7).mean()) if not hist.empty else float("nan")

    if df_last.empty or not cand_cols:
        st.info("Sem dados de justificações para o último dia.")
    else:
        s_vals = df_last[cand_cols].iloc[0].astype(float)
        top2 = s_vals.sort_values(ascending=False).head(2)

        colA, colB = st.columns(2)
        for col, cat in zip([colA, colB], top2.index.tolist()):
            val = int(top2[cat])
            ma7_val = ma7_cat(cat)
            if pd.isna(ma7_val):
                delta_txt = None
                delta_color = "off"
            else:
                dif = val - ma7_val
                delta_txt = f"{'+' if dif >= 0 else ''}{int(dif)}"
                delta_color = "inverse"  # verde se <=0, vermelho se >0
            with col:
                st.metric(cat, val, delta=delta_txt, delta_color=delta_color)

        with st.expander("Ver todas as categorias do último dia (detalhe)"):
            s_all = df_last[[c for c in df_last.columns if c != "Data"]].iloc[0].astype(float).sort_values(ascending=False)
            df_table = s_all.reset_index()
            df_table.columns = ["Categoria", "Valor"]
            st.dataframe(df_table, use_container_width=True, hide_index=True)

    # ---------------------------
    # Acumulado das justificações (período) — excluir "Sem justificação"
    # ---------------------------
    st.subheader("Acumulado das justificações (período)")
    periodo = st.selectbox("Período", ["1 semana", "Mês", "Ano", "1-3 Anos"], index=1)
    days_map = {"1 semana": 7, "Mês": 30, "Ano": 365, "1-3 Anos": 365*3}
    dias = days_map[periodo]
    end_date = df_just["Data"].max().normalize()
    start_date = end_date - pd.Timedelta(days=dias-1)
    dfj_win = df_just[(df_just["Data"] >= start_date) & (df_just["Data"] <= end_date)].copy()

    sem_col = find_sem_just_col(dfj_win.columns)
    cols_sum = [c for c in dfj_win.columns if c != "Data" and (c != sem_col)]
    total_just = dfj_win[cols_sum].sum().sort_values(ascending=False)

    st.caption(f"Período: {start_date.date().isoformat()} a {end_date.date().isoformat()} ({periodo})")
    st.bar_chart(total_just)

# -----------------------------------------------------------------------------
# Rutura por hora do dia — desde sempre, todas as justificações, apenas linha
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Rutura por hora do dia — desde sempre, todas as justificações, apenas linha
# -----------------------------------------------------------------------------
st.header("Rutura por hora do dia")

if (df_events is None) or df_events.empty or (("Hora" not in df_events.columns) and ("Hora_int" not in df_events.columns)):
    st.info("Sem registos detalhados com hora para calcular o gráfico por hora.")
else:
    # Garantir Hora_int
    if "Hora_int" not in df_events.columns:
        hora_dt = pd.to_datetime(df_events["Hora"], errors="coerce", format="%H:%M:%S")
        bad = hora_dt.isna() & df_events["Hora"].notna()
        if bad.any():
            hora_dt.loc[bad] = pd.to_datetime("1900-01-01 " + df_events.loc[bad, "Hora"].astype(str), errors="coerce")
        df_events["Hora_int"] = hora_dt.dt.hour

    # Contagem por hora 0..23 (desde sempre; todas as justificações)
    contagem = (
        df_events.dropna(subset=["Hora_int"])
                 .assign(Hora=df_events["Hora_int"].astype(int))
                 .groupby("Hora", as_index=False)
                 .size()
                 .rename(columns={"size": "Ocorrencias"})
    )

    # garantir ticks 0..24 (duplicamos o último valor em 24 para fechar o eixo)
    horas_full = pd.DataFrame({"Hora": list(range(24))})
    contagem = horas_full.merge(contagem, on="Hora", how="left").fillna({"Ocorrencias": 0})
    contagem["Ocorrencias"] = contagem["Ocorrencias"].astype(int)
    contagem_full = contagem.copy()
    contagem_full.loc[len(contagem_full)] = [24, contagem_full["Ocorrencias"].iloc[-1]]

    # Apenas linha (sem barras) + labels do eixo X em 90º (verticais)
    line = (
        alt.Chart(contagem_full)
        .mark_line(color="#1F4E79", strokeWidth=2)
        .encode(
            x=alt.X(
                "Hora:O",
                title="Hora",
                axis=alt.Axis(labelAngle=0)  # <- VERTICAL (90º)
            ),
            y=alt.Y("Ocorrencias:Q", title="Ocorrências"),
            tooltip=[
                alt.Tooltip("Hora:O", title="Hora"),
                alt.Tooltip("Ocorrencias:Q", title="Ocorrências")
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(line, use_container_width=True)

# -----------------------------------------------------------------------------
# Top 5 piores agências (nº de ocorrências) — mantido
# -----------------------------------------------------------------------------
st.header("Top 5 piores agências (nº de ocorrências)")

if (df_events is None) or df_events.empty or ("Agencia" not in df_events.columns):
    st.info("Sem registos detalhados com 'Agencia' para calcular o ranking.")
else:
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        periodo_top = st.selectbox("Período", ["Tudo", "1 semana", "Mês", "Ano", "1-3 Anos"], index=2, key="periodo_top")
    with col_t2:
        just_opts2 = ["Todas"] + sorted([j for j in df_events["Justificacao"].dropna().unique()]) if "Justificacao" in df_events.columns else ["Todas"]
        just_sel2 = st.selectbox("Justificação", just_opts2, index=0, key="just_top")

    def filtro_periodo(df, col_data: str, periodo_label: str):
        days_map = {"1 semana": 7, "Mês": 30, "Ano": 365, "1-3 Anos": 365*3, "Tudo": None}
        dias = days_map[periodo_label]
        if dias is None:
            return df
        end_date = df[col_data].max().normalize()
        start_date = end_date - pd.Timedelta(days=dias-1)
        return df[(df[col_data] >= start_date) & (df[col_data] <= end_date)].copy()

    top_df = df_events.copy()
    if "Data" in top_df.columns:
        top_df = filtro_periodo(top_df, "Data", periodo_top)
    if just_sel2 != "Todas" and "Justificacao" in top_df.columns:
        top_df = top_df[top_df["Justificacao"] == just_sel2]

    topN = (
        top_df.dropna(subset=["Agencia"])
              .groupby("Agencia", dropna=False)
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
            y=alt.Y("Agencia:N", title="Agência", sort="-x"),
            tooltip=["Agencia:N", alt.Tooltip("Ocorrencias:Q", title="Ocorrências")]
        )
        .properties(height=240)
    )
    labels = (
        alt.Chart(topN)
        .mark_text(align="left", dx=4, color="#333")
        .encode(x="Ocorrencias:Q", y=alt.Y("Agencia:N", sort="-x"), text="Ocorrencias:Q")
    )
    st.altair_chart((chart_top + labels), use_container_width=True)
    with st.expander("Ver tabela"):
        st.dataframe(topN, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Recomendações de decisão (bullet points)
# -----------------------------------------------------------------------------
st.header("Recomendações")

try:
    recs = []

    # 1) KPI vs MA7
    for metric in ["Ruturas", "Indisponiveis", "Anomalias"]:
        hoje = float(last_row[metric]) if pd.notna(last_row[metric]) else 0.0
        media7 = ma7_excluindo_hoje(df_main[metric], last_date_main)
        if not pd.isna(media7):
            if hoje > media7:
                recs.append(f"**{metric}** acima da média 7 dias ({int(hoje)} vs {int(round(media7))}). "
                            f"Reforçar recursos/diagnóstico root-cause para reduzir {metric.lower()} nos próximos dias.")
            else:
                recs.append(f"**{metric}** abaixo ou em linha com a média 7 dias ({int(hoje)} ≤ {int(round(media7))}). "
                            f"Manter práticas atuais e monitorizar.")

    # 2) Hora de pico (desde sempre)
    if (df_events is not None) and ("Hora_int" in df_events.columns):
        cont = (df_events.dropna(subset=["Hora_int"])
                         .assign(Hora=df_events["Hora_int"].astype(int))
                         .groupby("Hora").size().sort_values(ascending=False))
        if not cont.empty:
            hora_pico = int(cont.index[0])
            vol_pico = int(cont.iloc[0])
            recs.append(f"**Hora de pico**: {hora_pico:02d}h (≈ {vol_pico} ocorrências). "
                        f"Alocar equipa técnica/abastecimento preventivo antes deste período e reforçar monitorização entre "
                        f"**{hora_pico:02d}:00–{(hora_pico+1)%24:02d}:00**.")

    # 3) Categoria mais incidente (exclui 'Sem justificação')
    if (df_just is not None) and (not df_just.empty):
        sem_col = find_sem_just_col(df_just.columns)
        cats = [c for c in df_just.columns if c not in ["Data", sem_col]]
        if cats:
            tot = df_just[cats].sum().sort_values(ascending=False)
            if not tot.empty:
                top_cat = tot.index[0]
                recs.append(f"**Categoria mais incidente**: _{top_cat}_. Priorizar ações específicas (procedimentos, formação, manutenção preventiva).")

    # 4) Top 3 agências
    if (df_events is not None) and ("Agencia" in df_events.columns):
        worst = (df_events.dropna(subset=["Agencia"])
                           .groupby("Agencia").size()
                           .sort_values(ascending=False).head(3))
        if not worst.empty:
            lista = "; ".join([f"{ag} ({int(cnt)})" for ag, cnt in worst.items()])
            recs.append(f"**Agências com pior desempenho (Top 3)**: {lista}. "
                        f"Agendar **visita técnica** e plano de ação dedicado a cada uma.")

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
csv_main = df_main.to_csv(index=False, sep=";").encode("utf-8-sig")
st.download_button("Baixar CSV (tabela principal)", csv_main, file_name="ruturas_principal.csv", mime="text/csv")

if (df_just is not None) and (not df_just.empty):
    csv_just = df_just.to_csv(index=False, sep=";").encode("utf-8-sig")
    st.download_button("Baixar CSV (justificações)", csv_just, file_name="ruturas_justificacoes.csv", mime="text/csv")

if (df_events is not None) and (not df_events.empty):
    csv_ev = df_events.to_csv(index=False, sep=";").encode("utf-8-sig")
    st.download_button("Baixar CSV (registos detalhados)", csv_ev, file_name="ruturas_registos.csv", mime="text/csv")