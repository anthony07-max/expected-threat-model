"""Fit xT, value every pass and carry, rank players and teams, make charts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc
from xt import fit_xt, value_moves, L, W

ROOT = Path(__file__).resolve().parents[1]
P, C = ROOT / "data" / "processed", ROOT / "charts"
C.mkdir(exist_ok=True)
MIN_MINUTES = 900

def pitch(ax):
    ax.add_patch(Rectangle((0, 0), 120, 80, fill=False, lw=1))
    for x0 in (0, 102): ax.add_patch(Rectangle((x0, 18), 18, 44, fill=False, lw=1))
    ax.plot([60, 60], [0, 80], "k", lw=1); ax.add_patch(Arc((60, 40), 20, 20, lw=1))
    ax.set_xlim(-1, 121); ax.set_ylim(81, -1); ax.set_aspect("equal"); ax.axis("off")

def main():
    moves = pd.read_parquet(P / "moves.parquet")
    shots = pd.read_parquet(P / "shots.parquet")
    mins = pd.read_parquet(P / "minutes.parquet")
    # Penalties are excluded from g (set pieces from the spot would inflate one zone)
    open_shots = shots[shots.subtype != "Penalty"]
    grid, parts = fit_xt(moves, open_shots)
    np.save(P / "xt_grid.npy", grid)
    moves["xt"] = value_moves(moves, grid)

    # --- xT grid chart ---
    fig, ax = plt.subplots(figsize=(10, 6.5))
    im = ax.imshow(grid.T, extent=[0, 120, 80, 0], cmap="magma_r", origin="upper")
    for i in range(L):
        for j in range(W):
            ax.text((i + .5) * 120 / L, (j + .5) * 80 / W, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=5,
                    color="white" if grid[i, j] > grid.max() * .5 else "black")
    pitch(ax); plt.colorbar(im, ax=ax, shrink=.7, label="xT (probability of scoring within the possession chain)")
    ax.set_title("Expected Threat grid, Premier League 2015-16 (attacking left to right)")
    fig.tight_layout(); fig.savefig(C / "01_xt_grid.png", dpi=150); plt.close()

    # --- Player rankings ---
    pm = mins.groupby(["player_id"]).agg(player=("player", "first"), minutes=("minutes", "sum"),
                                         team=("team", lambda s: s.mode().iat[0]),
                                         position=("position", lambda s: s.dropna().mode().iat[0] if s.notna().any() else None)).reset_index()
    agg = moves.groupby("player_id").agg(xt=("xt", "sum"),
                                         xt_pass=("xt", lambda s: s[moves.loc[s.index, "type"] == "Pass"].sum()),
                                         xt_carry=("xt", lambda s: s[moves.loc[s.index, "type"] == "Carry"].sum()),
                                         actions=("xt", "size"))
    sh = shots.groupby("player_id").agg(np_xg=("xg", lambda s: s[shots.loc[s.index, "subtype"] != "Penalty"].sum()),
                                        goals=("goal", "sum"))
    pr = pm.join(agg, on="player_id").join(sh, on="player_id").fillna({"np_xg": 0, "goals": 0})
    pr = pr[pr.minutes >= MIN_MINUTES].copy()
    for c in ["xt", "xt_pass", "xt_carry", "np_xg"]:
        pr[f"{c}_p90"] = pr[c] / pr.minutes * 90
    pr["pos_group"] = pr.position.fillna("").map(lambda p: "Goalkeeper" if "Goalkeeper" in p else
        "Defender" if ("Back" in p) else "Midfielder" if "Midfield" in p else "Forward/Winger" if p else "Unknown")
    pr = pr.sort_values("xt_p90", ascending=False)
    pr.to_csv(P / "player_xt.csv", index=False)

    top = pr.head(20)[::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top.player + " (" + top.team + ")", top.xt_pass_p90, label="from passes", color="#264653")
    ax.barh(top.player + " (" + top.team + ")", top.xt_carry_p90, left=top.xt_pass_p90, label="from carries", color="#e9c46a")
    ax.set_xlabel("xT added per 90"); ax.legend(); ax.set_title(f"Top 20 players by xT per 90 (min {MIN_MINUTES} mins), PL 2015-16")
    fig.tight_layout(); fig.savefig(C / "02_top_players_xt.png", dpi=150); plt.close()

    # xT vs non-penalty xG: creators vs finishers
    fig, ax = plt.subplots(figsize=(9, 7))
    a = pr[pr.pos_group != "Goalkeeper"]
    ax.scatter(a.xt_p90, a.np_xg_p90, s=12, alpha=.5, color="#8d99ae")
    for r in pd.concat([a.nlargest(8, "xt_p90"), a.nlargest(6, "np_xg_p90")]).drop_duplicates("player_id").itertuples():
        ax.annotate(r.player.split()[-1], (r.xt_p90, r.np_xg_p90), fontsize=8)
    ax.set_xlabel("xT added per 90 (ball progression and creation)"); ax.set_ylabel("Non-penalty xG per 90 (shooting threat)")
    ax.set_title("Creators vs finishers"); fig.tight_layout(); fig.savefig(C / "03_xt_vs_xg.png", dpi=150); plt.close()

    # Teams
    tm = moves.groupby("team").agg(xt=("xt", "sum"), passes=("type", lambda s: (s == "Pass").sum()))
    tm["xt_per_game"] = tm.xt / 38
    tm["xt_per_100_passes"] = tm.xt / tm.passes * 100
    tm = tm.sort_values("xt_per_game", ascending=False)
    tm.to_csv(P / "team_xt.csv")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(tm.passes / 38, tm.xt_per_game, color=["#0053a0" if t == "Leicester City" else "#8d99ae" for t in tm.index], s=40)
    for t, r in tm.iterrows(): ax.annotate(t, (r.passes / 38, r.xt_per_game), fontsize=7)
    ax.set_xlabel("Passes per game"); ax.set_ylabel("xT generated per game (passes + carries)")
    ax.set_title("Volume vs threat: Leicester's title came from efficiency, not possession")
    fig.tight_layout(); fig.savefig(C / "04_team_xt.png", dpi=150); plt.close()

    # Where Leicester vs league create xT
    def zone_map(df):
        h = np.zeros((L, W)); zi = np.clip((df.x / 120 * L).astype(int), 0, L - 1); zj = np.clip((df.y / 80 * W).astype(int), 0, W - 1)
        np.add.at(h, (zi, zj), df.xt); return h
    lei = moves[moves.team == "Leicester City"]
    diff = zone_map(lei) / 38 - zone_map(moves) / (38 * 20)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    v = np.abs(diff).max()
    im = ax.imshow(diff.T, extent=[0, 120, 80, 0], cmap="RdBu_r", vmin=-v, vmax=v)
    pitch(ax); plt.colorbar(im, ax=ax, shrink=.7, label="xT per game vs league-average team")
    ax.set_title("Leicester City 2015-16: where their moves generated more (red) or less (blue) threat")
    fig.tight_layout(); fig.savefig(C / "05_leicester_xt_zones.png", dpi=150); plt.close()

    # Sanity check: team xT vs actual goals
    goals = shots[shots.goal].groupby("team").size()
    corr = float(np.corrcoef(tm.xt_per_game, goals.reindex(tm.index))[0, 1])
    xgc = float(np.corrcoef(tm.xt_per_game, shots.groupby("team").xg.sum().reindex(tm.index))[0, 1])

    kf = {"matches": int(moves.match_id.nunique()), "moves_valued": int(len(moves)), "shots": int(len(shots)),
          "iterations": parts["iterations"], "max_zone_xt": float(grid.max()), "own_box_xt": float(grid[0].mean()),
          "top_players": pr.head(10)[["player", "team", "minutes", "xt_p90", "xt_pass_p90", "xt_carry_p90"]].round(3).to_dict("records"),
          "top_carriers": pr.nlargest(5, "xt_carry_p90")[["player", "team", "xt_carry_p90"]].round(3).to_dict("records"),
          "top_defenders": pr[pr.pos_group == "Defender"].head(5)[["player", "team", "xt_p90"]].round(3).to_dict("records"),
          "teams": tm.reset_index()[["team", "xt_per_game", "xt_per_100_passes", "passes"]].round(3).to_dict("records"),
          "corr_team_xt_goals": corr, "corr_team_xt_xg": xgc,
          "carry_share_of_xt": float(moves[moves.type == "Carry"].xt.sum() / moves.xt.sum())}
    (P / "key_findings.json").write_text(json.dumps(kf, indent=2))
    print(json.dumps({k: v for k, v in kf.items() if k != "teams"}, indent=1)); print(tm.round(3))

if __name__ == "__main__":
    main()
