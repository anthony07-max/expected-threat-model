from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

P = Path(__file__).resolve().parents[1] / "data" / "processed"
st.set_page_config(page_title="Expected Threat (xT)", layout="wide")
st.title("Expected Threat (xT): Premier League 2015-16")
st.caption("Possession value model on StatsBomb open data. By Anthony Adusei-Poku (@anthony07-max)")
pr = pd.read_csv(P / "player_xt.csv")
grid = np.load(P / "xt_grid.npy")

c1, c2, c3 = st.columns(3)
teams = ["All"] + sorted(pr.team.unique())
team = c1.selectbox("Team", teams, index=teams.index("Leicester City"))
pos = c2.multiselect("Position group", sorted(pr.pos_group.unique()), default=[p for p in pr.pos_group.unique() if p != "Goalkeeper"])
mm = c3.slider("Minimum minutes", 900, 3400, 900, 100)
f = pr[(pr.minutes >= mm) & pr.pos_group.isin(pos)]
if team != "All":
    f = f[f.team == team]
st.dataframe(f[["player", "team", "pos_group", "minutes", "xt_p90", "xt_pass_p90", "xt_carry_p90", "np_xg_p90", "goals"]].round(3),
             hide_index=True, use_container_width=True)
st.plotly_chart(px.scatter(f, x="xt_p90", y="np_xg_p90", hover_name="player", color="pos_group", size="minutes",
                           title="Creators vs finishers"), use_container_width=True)
st.plotly_chart(px.imshow(grid.T, color_continuous_scale="magma_r", aspect="equal", title="xT grid (attacking left to right)",
                          labels=dict(color="xT")), use_container_width=True)
