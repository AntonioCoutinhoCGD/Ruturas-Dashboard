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
    page_icon=LOGO_URL,   # <- ícone da página = logotipo da Caixa
    layout="wide",
)

# Título com logo ao lado (sem emojis)
c_logo, c_title = st.columns([0.12, 0.88])
with c_logo:
    st.image(LOGO_URL, width=72)
with c_title:
    st.markdown("<h1 style='margin-bottom:0;'>Ruturas Dashboard</h1>", unsafe_allow_html=True)

# Texto simplificado
st.write("Carrega um ficheiro CSV")

# -----------------------------------------------------------------------------
# Helpers / Constantes
# -----------------------------------------------------------------------------
MAIN_COLS = ["Data", "Ruturas", "Indisponiveis", "Anomalias"]

def base_name(colname: str) -> str:
    """Remove sufixos do pandas (ex.: 'Data.1' -> 'Data'), trim e minúsculas."""
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

    # Retirar horas
    out["Data"] = pd.to_datetime(out["Data"], errors="coerce").dt.normalize()
    for c in ["Ruturas","Indisponiveis","Anomalias"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    out = out.dropna(subset=["Data"]).sort_values("Data").reset_index(drop=True)

    # Aviso se vierem a zero
    msg = []
    for c in ["Indisponiveis","Anomalias"]:
        if (out[c].fillna(0) == 0).all():
            msg.append(c)
    if msg:
        st.warning(
            "As colunas " + " e ".join(msg) + " vieram vazias no ficheiro. "
            "Se tinhas valores no Excel, faz upload do .xlsx ou exporta um CSV principal separado."
        )
    return out

def normalize_just(df_right: pd.DataFrame):
    """
    Normaliza o bloco da direita (justificações).
    Aceita qualquer número de categorias.
    Suporta Data + categorias ou apenas categorias (com 'Registo').
    Retira horas quando existe Data.
    """
    if df_right is None or df_right.empty:
        return None, False

    dfj = df_right.copy()
    dfj.columns = [str(c).strip() for c in dfj.columns]

    # Tem Data (ex.: 'Data' ou 'Data.1')?
    has_data = any(base_name(c) == "data" for c in dfj.columns)
    if has_data:
        data_col = find_first_col_by_base(dfj.columns, "data")
        if data_col != "Data":
            dfj.rename(columns={data_col: "Data"}, inplace=True)

        dfj["Data"] = pd.to_datetime(dfj["Data"], errors="coerce").dt.normalize()
        dfj = dfj.dropna(subset=["Data"])

        cat_cols = [c for c in dfj.columns if c != "Data"]
        for c in cat_cols:
            dfj[c] = pd.to_numeric(dfj[c], errors="coerce").fillna(0)
        return dfj, True

    # Agregado sem Data
    for c in dfj.columns:
        dfj[c] = pd.to_numeric(dfj[c], errors="coerce").fillna(0)
    if "Registo" not in dfj.columns:
        dfj.insert(0, "Registo", np.arange(1, len(dfj)+1))
    return dfj, False

def find_sem_just_col(cols) -> str | None:
    # encontra a coluna "Sem Justificação" com robustez a acentos/maiúsculas
    for c in cols:
        n = normalize_text_pt(c)
        if n == "sem justificacao":
            return c
    for c in cols:
        n = normalize_text_pt(c)
        if n.startswith("sem justific"):
            return c
    return None

# -----------------------------------------------------------------------------
# Leitura robusta do upload
# -----------------------------------------------------------------------------
def read_uploaded(file):
    """
    Lê CSV ou Excel e separa bloco esquerdo (principal) e bloco direito (justificações),
    mesmo quando há coluna vazia entre blocos e a 2.ª 'Data' vem como 'Data.1'.
    """
    name = file.name.lower()

    # -------- CSV --------
    if name.endswith(".csv"):
        # tentar UTF-8 → fallback Latin-1
        try:
            df = pd.read_csv(file, sep=";", header=1, engine="python")
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, sep=";", header=1, engine="python", encoding="latin-1")

        df.columns = [str(c).strip() for c in df.columns]
        cols = df.columns.tolist()

        # MAIN
        left_data_col = find_first_col_by_base(cols, "data")
        if left_data_col is None:
            return None, None, False

        main_candidates = [left_data_col] + [c for c in ["Ruturas","Indisponiveis","Anomalias"] if c in cols]
        df_main = normalize_main(df[main_candidates])

        # JUSTIFICAÇÕES: 2.ª 'Data' (ex.: 'Data.1')
        bases = [base_name(c) for c in cols]
        data_positions = [i for i, b in enumerate(bases) if b == "data"]

        if len(data_positions) < 2:
            return df_main, None, False

        right_start = data_positions[1]
        df_right = df.iloc[:, right_start:]
        df_just, just_has_date = normalize_just(df_right)
        return df_main, df_just, just_has_date

    # -------- EXCEL --------
    if name.endswith(".xlsx"):
        try:
            df = pd.read_excel(file, header=1, engine="openpyxl")
        except ImportError:
            st.error(
                "Este ambiente não tem 'openpyxl' para ler Excel. "
                "Solução: guarda como CSV (;) com cabeçalho na 2.ª linha e volta a carregar."
            )
            return None, None, False
        except Exception as e:
            st.error(f"Erro a ler Excel: {e}")
            return None, None, False

        df.columns = [str(c).strip() for c in df.columns]
        cols = df.columns.tolist()

        left_data_col = find_first_col_by_base(cols, "data")
        if left_data_col is None:
            return None, None, False

        main_candidates = [left_data_col] + [c for c in ["Ruturas","Indisponiveis","Anomalias"] if c in cols]
        df_main = normalize_main(df[main_candidates])

        bases = [base_name(c) for c in cols]
        data_positions = [i for i, b in enumerate(bases) if b == "data"]

        if len(data_positions) < 2:
            return df_main, None, False

        right_start = data_positions[1]
        df_right = df.iloc[:, right_start:]
        df_just, just_has_date = normalize_just(df_right)
        return df_main, df_just, just_has_date

    st.error("Formato não suportado. Usa CSV ou XLSX.")
    return None, None, False

# -----------------------------------------------------------------------------
# Upload
# -----------------------------------------------------------------------------
file = st.file_uploader("Carrega aqui", type=["csv", "xlsx"])
if not file:
    st.info("A aguardar ficheiro…")
    st.stop()

df_main, df_just, just_has_date = read_uploaded(file)
if df_main is None:
    st.stop()

st.success("Dados carregados com sucesso.")

# -----------------------------------------------------------------------------
# KPIs — mostrar a DATA em vez de 'último dia'
# -----------------------------------------------------------------------------
last_date_main = df_main["Data"].max()
st.header(f"Indicadores — {last_date_main.date().isoformat()}")

last_row = df_main.loc[df_main["Data"] == last_date_main].iloc[-1]
k1, k2, k3 = st.columns(3)
k1.metric("Ruturas", int(last_row["Ruturas"]) if pd.notna(last_row["Ruturas"]) else 0)
k2.metric("Indisponíveis", int(last_row["Indisponiveis"]) if pd.notna(last_row["Indisponiveis"]) else 0)
k3.metric("Anomalias", int(last_row["Anomalias"]) if pd.notna(last_row["Anomalias"]) else 0)

# -----------------------------------------------------------------------------
# Evolução diária — linhas limpas; mostrar valor só em hover
# -----------------------------------------------------------------------------
st.header("Evolução diária")

df_chart = df_main.melt(
    id_vars=["Data"],
    value_vars=["Ruturas", "Indisponiveis", "Anomalias"],
    var_name="Categoria",
    value_name="Valor"
)

# Seleção por hover: ativa tooltip e ponto apenas sobre a série/dia do rato
hover = alt.selection_point(fields=["Data", "Categoria"], nearest=True, on="mouseover", empty="none")

base = alt.Chart(df_chart).encode(
    x=alt.X("Data:T", title="Data", axis=alt.Axis(format="%Y-%m-%d")),
    y=alt.Y("Valor:Q", title="Valor"),
    color=alt.Color("Categoria:N", legend=alt.Legend(title=None))
)

# Linha simples
lines = base.mark_line(strokeWidth=2)

# Ponto apenas em hover
points_on_hover = base.mark_point(size=60).transform_filter(hover)

# Tooltip apenas em hover (sem labels fixos/linhas extra)
chart = (lines + points_on_hover).encode(
    tooltip=[
        alt.Tooltip("Data:T", title="Data", format="%Y-%m-%d"),
        alt.Tooltip("Categoria:N", title="Série"),
        alt.Tooltip("Valor:Q", title="Valor", format=".0f")
    ]
).add_params(hover).properties(height=340)

st.altair_chart(chart, use_container_width=True)

# -----------------------------------------------------------------------------
# JUSTIFICAÇÕES — Destaque selecionável (default: Sem Justificação) + Acumulado com seletor período
# -----------------------------------------------------------------------------
st.header("Justificações")

if df_just is None or df_just.empty:
    st.info("Sem dados de justificações.")
else:
    # Conjunto de categorias disponíveis
    if just_has_date:
        categorias = [c for c in df_just.columns if c != "Data"]
    else:
        categorias = [c for c in df_just.columns if c != "Registo"]

    # Default para "Sem Justificação" se existir
    default_cat = find_sem_just_col(categorias) or (categorias[0] if categorias else None)

    # ---- seletor de categoria para destaque ----
    st.subheader("Destaque de uma categoria")
    selected_cat = st.selectbox(
        "Seleciona a categoria para analisar",
        options=categorias,
        index=categorias.index(default_cat) if default_cat in categorias else 0
    )

    if just_has_date and selected_cat:
        last_date_just = df_just["Data"].max()
        df_last = df_just[df_just["Data"] == last_date_just].copy()
        last_val = int(float(df_last[selected_cat].iloc[0])) if not df_last.empty else 0

        cols_sem = st.columns(2)

        # KPI 1: valor no último dia
        with cols_sem[0]:
            st.metric(f"{selected_cat} (dia {last_date_just.date().isoformat()})", last_val)

        # KPI 2: MÉDIA DIÁRIA DESDE SEMPRE (floor, inteiro, sem decimais)
        with cols_sem[1]:
            total_sel = float(df_just[selected_cat].sum())
            num_dias = int(df_just["Data"].nunique())
            media_diaria = int(np.floor(total_sel / max(1, num_dias)))
            st.metric(f"{selected_cat} (média diária — desde sempre)", media_diaria)

        # Gráfico diário da categoria escolhida (sem horas)
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
    else:
        st.info("Não foi possível destacar a categoria (faltam 'Data' ou não há categorias).")

    st.markdown("---")

    # 2) Secção completa — último dia (todas as categorias)
    st.subheader("Justificações do último dia (todas as categorias)")
    if just_has_date:
        last_date_just = df_just["Data"].max()
        df_last = df_just[df_just["Data"] == last_date_just].copy()
        st.markdown(f"Último dia: {last_date_just.date().isoformat()}")

        cat_cols = [c for c in df_last.columns if c != "Data"]
        cols = st.columns(len(cat_cols)) if cat_cols else []
        for col, c in zip(cols, cat_cols):
            value = int(float(df_last[c].iloc[0])) if not pd.isna(df_last[c].iloc[0]) else 0
            col.metric(label=c.rstrip(), value=value)
    else:
        st.warning("As justificações não têm coluna 'Data' — não é possível destacar o último dia.")

    # 3) Acumulado com seletor de período — SEM HORAS
    st.subheader("Acumulado das justificações (período)")
    if just_has_date:
        periodo = st.selectbox("Período", ["1 semana", "Mês", "Ano", "1-3 Anos"], index=1)
        days_map = {"1 semana": 7, "Mês": 30, "Ano": 365, "1-3 Anos": 365*3}
        dias = days_map[periodo]
        end_date = df_just["Data"].max().normalize()
        start_date = end_date - pd.Timedelta(days=dias-1)
        dfj_win = df_just[(df_just["Data"] >= start_date) & (df_just["Data"] <= end_date)].copy()

        total_just = dfj_win[[c for c in dfj_win.columns if c != "Data"]].sum().sort_values(ascending=False)
        st.caption(f"Período: {start_date.date().isoformat()} a {end_date.date().isoformat()} ({periodo})")
        st.bar_chart(total_just)
    else:
        total_just = df_just[[c for c in df_just.columns if c != "Registo"]].sum().sort_values(ascending=False)
        st.bar_chart(total_just)

    # 4) Tabela completa
    with st.expander("Ver tabela completa de justificações"):
        st.dataframe(df_just, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Downloads
# -----------------------------------------------------------------------------
st.header("Download dos dados")
csv_main = df_main.to_csv(index=False, sep=";").encode("utf-8-sig")
st.download_button("Baixar CSV (tabela principal)", csv_main, file_name="ruturas_principal.csv", mime="text/csv")

if df_just is not None and not df_just.empty:
    csv_just = df_just.to_csv(index=False, sep=";").encode("utf-8-sig")
    st.download_button("Baixar CSV (justificações)", csv_just, file_name="ruturas_justificacoes.csv", mime="text/csv")