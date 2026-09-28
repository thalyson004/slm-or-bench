"""Create the Gurobi environment and execute one policy-checked program."""

from __future__ import annotations

import os
import runpy
import sys

import gurobipy as gp


def main(script_path: str) -> None:
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        credentials = ("WLSACCESSID", "WLSSECRET", "LICENSEID")
        if all(os.getenv(name) for name in credentials):
            for name in credentials:
                value = os.environ[name]
                env.setParam(name, int(value) if name == "LICENSEID" else value)
        env.start()
        runpy.run_path(script_path, init_globals={"env": env}, run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1])
