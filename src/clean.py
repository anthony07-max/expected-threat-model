"""Build tidy tables: on-ball moves (passes, carries), shots, and player minutes per match."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"

def minutes(ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mid, g in ev.groupby("match_id"):
        end = g.minute.max() + 1
        on, pos = {}, {}
        for r in g[g.type == "Starting XI"].itertuples():
            for pid, name, p in json.loads(r.extra):
                on[pid] = [name, r.team, 0, end]; pos[pid] = p
        for r in g[g.type == "Substitution"].itertuples():
            rid, rname = json.loads(r.extra)
            if r.player_id in on: on[r.player_id][3] = r.minute
            on[rid] = [rname, r.team, r.minute, end]; pos.setdefault(rid, r.position)
        for pid, (name, team, a, b) in on.items():
            rows.append({"match_id": mid, "player_id": pid, "player": name, "team": team, "minutes": max(b - a, 0), "position": pos.get(pid)})
    return pd.DataFrame(rows)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = pd.concat([pd.read_parquet(f).assign(competition=f.stem) for f in RAW.glob("*.parquet")])
    moves = ev[ev.type.isin(["Pass", "Carry"]) & ev.x.notna() & ev.end_x.notna()].copy()
    # Successful pass: StatsBomb leaves outcome empty. Carries are always "successful" moves.
    moves["success"] = moves.outcome.isna()
    shots = ev[(ev.type == "Shot") & ev.x.notna()].copy()
    shots["goal"] = shots.outcome == "Goal"
    cols = ["competition", "match_id", "index", "period", "minute", "type", "subtype", "team", "player_id", "player",
            "position", "x", "y", "end_x", "end_y", "success", "cross", "under_pressure"]
    moves[cols].to_parquet(OUT / "moves.parquet", index=False)
    shots[["competition", "match_id", "team", "player_id", "player", "x", "y", "xg", "goal", "subtype", "body_part"]].to_parquet(OUT / "shots.parquet", index=False)
    mins = minutes(ev)
    mins.to_parquet(OUT / "minutes.parquet", index=False)
    print(len(moves), "moves", len(shots), "shots", len(mins), "player-matches")

if __name__ == "__main__":
    main()
