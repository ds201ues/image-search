"""Judged eval on people query types with the scores view swapped. Usage: uv run python -m exp.vlm_alone.eval_people <view> <label>"""
import sys
from app import search as S
from eval import run_eval
S.TABLES["scores"] = sys.argv[1]
sys.argv = ["run_eval", "--types", "person,person_context,real", "--label", sys.argv[2]]
run_eval.main()
