import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import minhastats as ms

st.set_page_config(page_title="Laboratório Estatístico Interativo", page_icon="📊", layout="wide")
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dados" / "dataset.csv"

st.title("📊 Laboratório Estatístico Interativo")
st.caption("Matemática e Estatística para Computação — Sistematização 2026")

if not DATA.exists():
    st.warning("O dataset ainda não está na pasta dados/.")
    st.code("pip install -r requirements.txt\npython baixar_dados.py\nstreamlit run app.py", language="bash")
    st.stop()

df = pd.read_csv(DATA)

# Renomeação amigável apenas na interface.
labels = {
    "temp": "Temperatura normalizada",
    "atemp": "Sensação térmica normalizada",
    "hum": "Umidade normalizada",
    "windspeed": "Velocidade do vento normalizada",
    "casual": "Usuários casuais",
    "registered": "Usuários registrados",
    "cnt": "Total de bicicletas alugadas",
    "hr": "Hora",
    "mnth": "Mês",
    "season": "Estação",
    "weathersit": "Situação climática",
    "workingday": "Dia útil",
    "holiday": "Feriado",
    "weekday": "Dia da semana",
    "yr": "Ano",
}

numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in ["instant"]]
categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in ["dteday"]]

st.sidebar.header("Navegação")
modulo = st.sidebar.radio("Módulo", [
    "0 — Dataset", "1 — Núcleo estatístico", "2 — Descritiva", "3 — LGN", "4 — TCL e distribuições", "5 — Correlação e regressão", "6 — Descobertas"
])

if modulo == "0 — Dataset":
    st.header("Módulo 0 — Dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Registros", f"{len(df):,}".replace(",", "."))
    c2.metric("Variáveis", len(df.columns))
    c3.metric("Nulos", int(df.isna().sum().sum()))
    st.write("Fonte: UCI Machine Learning Repository — Bike Sharing.")
    st.dataframe(df.head(20), use_container_width=True)
    st.subheader("Tipos e valores ausentes")
    resumo = pd.DataFrame({"tipo": df.dtypes.astype(str), "nulos": df.isna().sum(), "únicos": df.nunique()})
    st.dataframe(resumo, use_container_width=True)

elif modulo == "1 — Núcleo estatístico":
    st.header("Módulo 1 — Núcleo estatístico 'na unha'")
    col = st.selectbox("Escolha uma variável numérica", numeric_cols, format_func=lambda x: labels.get(x, x))
    dados = df[col].dropna().tolist()
    q1, med, q3 = ms.quartis(dados)
    vals = {
        "Média": ms.media(dados), "Mediana": med, "Variância amostral": ms.variancia(dados),
        "Desvio padrão": ms.desvio_padrao(dados), "Amplitude": ms.amplitude(dados),
        "CV (%)": ms.coef_variacao(dados), "Q1": q1, "Q3": q3,
    }
    cols = st.columns(4)
    for i, (k, v) in enumerate(vals.items()):
        cols[i % 4].metric(k, f"{v:.4f}")
    st.info("As medidas exibidas são calculadas pelas funções de minhastats.py; NumPy é usado somente nos testes de validação.")

elif modulo == "2 — Descritiva":
    st.header("Módulo 2 — Descritiva interativa")
    col = st.selectbox("Variável numérica", numeric_cols, format_func=lambda x: labels.get(x, x))
    dados = df[col].dropna().tolist()
    media, mediana, desvio = ms.media(dados), ms.mediana(dados), ms.desvio_padrao(dados)
    c1, c2, c3 = st.columns(3)
    c1.metric("Média", f"{media:.2f}")
    c2.metric("Mediana", f"{mediana:.2f}")
    c3.metric("Desvio padrão", f"{desvio:.2f}")
    q1, _, q3 = ms.quartis(dados)
    lo, hi = ms.limites_iqr(dados)
    st.write(f"**Q1:** {q1:.2f} · **Q3:** {q3:.2f} · **IQR:** {q3-q1:.2f} · **Limites de outlier:** [{lo:.2f}, {hi:.2f}]")
    if media > mediana + 0.5 * desvio:
        leitura = "Assimetria à direita: valores altos puxam a média."
    elif media < mediana - 0.5 * desvio:
        leitura = "Assimetria à esquerda: valores baixos puxam a média."
    else:
        leitura = "Distribuição aproximadamente simétrica pela regra adotada."
    st.info(leitura)
    fig, ax = plt.subplots()
    ax.hist(dados, bins=ms.sturges(len(dados)), density=True)
    ax.set_title(f"Histograma — {labels.get(col, col)}")
    ax.set_xlabel(labels.get(col, col)); ax.set_ylabel("Densidade")
    st.pyplot(fig); plt.close(fig)
    fig2, ax2 = plt.subplots(); ax2.boxplot(dados, vert=False); ax2.set_title("Boxplot e dispersão")
    st.pyplot(fig2); plt.close(fig2)
    cat = st.selectbox("Variável categórica", categorical_cols, format_func=lambda x: labels.get(x, x))
    freq = df[cat].value_counts().sort_index()
    st.bar_chart(freq)

elif modulo == "3 — LGN":
    st.header("Módulo 3 — Lei dos Grandes Números")
    n = st.slider("Número de lançamentos", 100, 5000, 5000, 100)
    rng = np.random.default_rng(42)
    lanc = rng.integers(0, 2, size=n)
    freq = np.cumsum(lanc) / np.arange(1, n + 1)
    fig, ax = plt.subplots(); ax.plot(np.arange(1, n+1), freq); ax.axhline(0.5, linestyle="--")
    ax.set_xlabel("Número de lançamentos"); ax.set_ylabel("Frequência relativa de cara")
    ax.set_title("Convergência da frequência para 0,5")
    st.pyplot(fig); plt.close(fig)
    st.write(f"Após {n:,} lançamentos, a frequência observada foi **{freq[-1]:.4f}**.")

elif modulo == "4 — TCL e distribuições":
    st.header("Módulos 3 e 4 — TCL e distribuições")
    col = st.selectbox("Variável assimétrica para o TCL", [c for c in numeric_cols if c in ["casual", "registered", "cnt", "windspeed", "hum"]], format_func=lambda x: labels.get(x, x))
    dados = np.array(df[col].dropna().tolist(), dtype=float)
    n = st.slider("Tamanho da amostra", 2, 100, 30)
    reps = st.slider("Número de repetições", 100, 3000, 1000, 100)
    rng = np.random.default_rng(123)
    medias = [ms.media(rng.choice(dados, size=n, replace=True).tolist()) for _ in range(reps)]
    fig, ax = plt.subplots(); ax.hist(medias, bins=30, density=True)
    ax.set_title(f"Distribuição das médias — n={n}, repetições={reps}")
    ax.set_xlabel("Média amostral"); ax.set_ylabel("Densidade")
    st.pyplot(fig); plt.close(fig)
    st.info("À medida que n cresce, a distribuição das médias tende a se aproximar de uma forma Normal, conforme o TCL.")

    st.subheader("Sobreposição de distribuições teóricas")
    dist = st.selectbox("Distribuição", ["Normal", "Exponencial", "Uniforme"])
    x = np.linspace(float(np.min(dados)), float(np.max(dados)), 300)
    m = ms.media(dados); s = ms.desvio_padrao(dados)
    if dist == "Normal":
        dens = np.exp(-0.5*((x-m)/s)**2)/(s*np.sqrt(2*np.pi)) if s else np.zeros_like(x)
    elif dist == "Exponencial":
        lam = 1/m if m > 0 else 0
        dens = lam*np.exp(-lam*x) if lam else np.zeros_like(x)
        dens[x < 0] = 0
    else:
        a, b = float(np.min(dados)), float(np.max(dados)); dens = np.ones_like(x)/(b-a) if b > a else np.zeros_like(x)
    fig, ax = plt.subplots(); ax.hist(dados, bins=ms.sturges(len(dados)), density=True, alpha=0.6, label="Dados"); ax.plot(x, dens, linewidth=2, label=dist)
    ax.legend(); ax.set_title(f"Dados × {dist}"); st.pyplot(fig); plt.close(fig)

elif modulo == "5 — Correlação e regressão":
    st.header("Módulo 5 — Correlação e regressão")
    xcol = st.selectbox("X", numeric_cols, index=numeric_cols.index("temp") if "temp" in numeric_cols else 0, format_func=lambda x: labels.get(x, x))
    ycol = st.selectbox("Y", numeric_cols, index=numeric_cols.index("cnt") if "cnt" in numeric_cols else min(1, len(numeric_cols)-1), format_func=lambda x: labels.get(x, x))
    dados = df[[xcol, ycol]].dropna()
    x, y = dados[xcol].tolist(), dados[ycol].tolist()
    r = ms.correlacao(x, y); b0, b1, r2 = ms.regressao_linear(x, y)
    c1, c2, c3 = st.columns(3); c1.metric("Correlação de Pearson", f"{r:.4f}"); c2.metric("R²", f"{r2:.4f}"); c3.metric("Inclinação b1", f"{b1:.4f}")
    xmin, xmax = min(x), max(x)
    pred_x = st.number_input("Valor de X para predição", float(xmin), float(xmax), float((xmin+xmax)/2))
    pred_y = b0 + b1*pred_x
    st.metric("Predição de Y", f"{pred_y:.2f}")
    fig, ax = plt.subplots(); ax.scatter(x, y, s=8, alpha=0.25)
    xx = np.linspace(xmin, xmax, 100); ax.plot(xx, [b0+b1*z for z in xx], linewidth=2)
    ax.set_xlabel(labels.get(xcol,xcol)); ax.set_ylabel(labels.get(ycol,ycol)); ax.set_title("Regressão linear por mínimos quadrados")
    st.pyplot(fig); plt.close(fig)
    st.warning("Correlação indica associação linear; não demonstra causalidade.")
    st.write(f"Interpretação: cada unidade a mais de **{labels.get(xcol,xcol)}** está associada, em média, a **{b1:.4f}** unidade(s) de **{labels.get(ycol,ycol)}**.")

elif modulo == "6 — Descobertas":
    st.header("Módulo 6 — Três descobertas")
    # Descoberta 1: grupos por estação
    nomes_est = {1:"Inverno",2:"Primavera",3:"Verão",4:"Outono"}
    grp = df.groupby("season")["cnt"].median().sort_values(ascending=False)
    st.subheader("1. Contraste entre estações")
    st.write(f"As medianas de aluguel total por estação variam de **{grp.min():.0f}** a **{grp.max():.0f}** bicicletas por registro.")
    st.bar_chart(grp.rename(index=nomes_est))
    # Descoberta 2: correlação temperatura-total
    r = ms.correlacao(df["temp"].tolist(), df["cnt"].tolist())
    st.subheader("2. Relação entre temperatura e demanda")
    st.write(f"A correlação de Pearson entre temperatura normalizada e total alugado é **{r:.3f}**. Isso descreve associação linear, não causalidade.")
    # Descoberta 3: extremos/outliers de demanda
    dados_cnt = df["cnt"].dropna().tolist(); lo, hi = ms.limites_iqr(dados_cnt)
    out = df[(df["cnt"] < lo) | (df["cnt"] > hi)]
    st.subheader("3. Pontos fora da regra do IQR")
    st.write(f"A regra do IQR identifica **{len(out)}** registros fora de [{lo:.0f}, {hi:.0f}]. Eles merecem investigação, mas não devem ser tratados automaticamente como erros.")
    st.dataframe(out[["dteday", "hr", "season", "weathersit", "cnt"]].head(20), use_container_width=True)

st.divider()
st.caption("Fonte do dataset: Fanaee-T (2013), Bike Sharing, UCI Machine Learning Repository. Projeto acadêmico — dados históricos de 2011–2012.")
