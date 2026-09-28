"""Frozen zero-shot and few-shot prompts for the three pipelines.

The prompt structure follows the explicit Input/Task/Steps/Output Contract
style used in comparable OR-agent papers. The benchmark still disallows
semantic review, feedback, and correction after a model response.
"""

from __future__ import annotations

SYSTEM = """You are an Operations Research Agent specializing in mathematical modeling and solver-backed verification.

You receive one optimization problem at a time. Preserve every datum, unit,
index, time convention, logical condition, objective direction, and requested
output quantity from the problem statement. Work only with the information in
the supplied inputs. Do not ask questions or use external tools.

The caller applies a strict output contract. Return only the artifact requested
by the task; do not add a preface, postscript, or alternative answer."""

SHOT_CONDITIONS = ("zero_shot", "few_shot")

# Adapted from English MAMO Easy records 1 and 11 (one-based file order).
# MAMO Easy is excluded from the ComplexOR, LogiOR, and IndustryOR denominator.
DEMONSTRATION_PROBLEM = """A marketing company allocates nonnegative integer resource units to projects X
and Y. The total allocation cannot exceed 1,000 units. Project X must receive
at least 200 units more than project Y. Project X costs $50 per unit and is
limited to 700 units; project Y costs $30 per unit and is limited to 500 units.
What is the minimum total allocation cost?"""

DEMONSTRATION_FORMULATION = """Sets and indices: P = {X, Y}, the two projects.
Parameters: c_X = 50 and c_Y = 30 dollars per unit; U_X = 700 and U_Y = 500;
total allocation limit B = 1000; required excess D = 200.
Decision variables and domains: x_X, x_Y in Z_+, with x_X <= U_X and
x_Y <= U_Y.
Objective and direction: minimize Z = 50 x_X + 30 x_Y.
Constraints: x_X + x_Y <= 1000; x_X - x_Y >= 200; x_X <= 700;
x_Y <= 500.
Requested reported quantity: the optimal objective value Z in dollars."""

DEMONSTRATION_CODE = '''import gurobipy as gp
from gurobipy import GRB

model = gp.Model(env=env)
x = model.addVar(lb=0, ub=700, vtype=GRB.INTEGER, name="x")
y = model.addVar(lb=0, ub=500, vtype=GRB.INTEGER, name="y")
model.addConstr(x + y <= 1000, name="total_allocation")
model.addConstr(x - y >= 200, name="required_excess")
model.setObjective(50 * x + 30 * y, GRB.MINIMIZE)
model.optimize()

if model.Status == GRB.OPTIMAL:
    print(f"FINAL_OBJECTIVE={model.ObjVal:.17g}")'''

DEMONSTRATION_2_PROBLEM = """An energy company chooses nonnegative integer numbers of solar and wind
projects. A solar project costs 7 units and a wind project costs 5 units. The
portfolio must satisfy 2 solar + 3 wind >= 10, while grid capacity requires
4 solar + wind <= 15. What is the minimum total investment cost?"""

DEMONSTRATION_2_FORMULATION = """Sets and indices: P = {S, W}, representing solar and wind projects.
Parameters: c_S = 7 and c_W = 5 cost units; output coefficients a_S = 2 and
a_W = 3; minimum output R = 10; grid coefficients g_S = 4 and g_W = 1;
grid limit G = 15.
Decision variables and domains: x_S, x_W in Z_+.
Objective and direction: minimize Z = 7 x_S + 5 x_W.
Constraints: 2 x_S + 3 x_W >= 10; 4 x_S + x_W <= 15.
Requested reported quantity: the optimal objective value Z in cost units."""

DEMONSTRATION_2_CODE = '''import gurobipy as gp
from gurobipy import GRB

model = gp.Model(env=env)
solar = model.addVar(lb=0, vtype=GRB.INTEGER, name="solar")
wind = model.addVar(lb=0, vtype=GRB.INTEGER, name="wind")
model.addConstr(2 * solar + 3 * wind >= 10, name="minimum_output")
model.addConstr(4 * solar + wind <= 15, name="grid_capacity")
model.setObjective(7 * solar + 5 * wind, GRB.MINIMIZE)
model.optimize()

if model.Status == GRB.OPTIMAL:
    print(f"FINAL_OBJECTIVE={model.ObjVal:.17g}")'''

FEW_SHOT_DIRECT = f"""FEW-SHOT DEMONSTRATIONS

EXAMPLE 1
=========

Input
-----
{DEMONSTRATION_PROBLEM}

Worked solution
---------------
Because x_X - x_Y >= 200 and both variables are nonnegative, every feasible
solution has x_X >= 200. All objective coefficients are positive, so the
minimum is attained at x_Y = 0 and x_X = 200. The objective is
50(200) + 30(0) = 10,000 dollars.

Expected response
-----------------
FINAL_OBJECTIVE=10000

EXAMPLE 2
=========

Input
-----
{DEMONSTRATION_2_PROBLEM}

Worked solution
---------------
For x_S = 0, the minimum feasible integer wind allocation is x_W = 4, with
cost 5(4) = 20. Grid capacity limits x_S to at most 3. For x_S = 1, 2, and 3,
the minimum feasible wind allocations are 3, 2, and 2, with costs 22, 24, and
31. Thus x_S = 0 and x_W = 4 are optimal.

Expected response
-----------------
FINAL_OBJECTIVE=20

END FEW-SHOT DEMONSTRATIONS
"""

FEW_SHOT_CODE = f"""FEW-SHOT DEMONSTRATIONS

EXAMPLE 1
=========

Input
-----
{DEMONSTRATION_PROBLEM}

Expected response
-----------------
```python
{DEMONSTRATION_CODE}
```

EXAMPLE 2
=========

Input
-----
{DEMONSTRATION_2_PROBLEM}

Expected response
-----------------
```python
{DEMONSTRATION_2_CODE}
```

END FEW-SHOT DEMONSTRATIONS
"""

FEW_SHOT_FORMULATION = f"""FEW-SHOT DEMONSTRATIONS

EXAMPLE 1
=========

Input
-----
{DEMONSTRATION_PROBLEM}

Expected response
-----------------
{DEMONSTRATION_FORMULATION}

EXAMPLE 2
=========

Input
-----
{DEMONSTRATION_2_PROBLEM}

Expected response
-----------------
{DEMONSTRATION_2_FORMULATION}

END FEW-SHOT DEMONSTRATIONS
"""

FEW_SHOT_FORMULATION_CODE = f"""FEW-SHOT DEMONSTRATIONS

EXAMPLE 1
=========

Original problem
----------------
{DEMONSTRATION_PROBLEM}

Supplied formulation
--------------------
{DEMONSTRATION_FORMULATION}

Expected response
-----------------
```python
{DEMONSTRATION_CODE}
```

EXAMPLE 2
=========

Original problem
----------------
{DEMONSTRATION_2_PROBLEM}

Supplied formulation
--------------------
{DEMONSTRATION_2_FORMULATION}

Expected response
-----------------
```python
{DEMONSTRATION_2_CODE}
```

END FEW-SHOT DEMONSTRATIONS
"""


def _shot_block(condition: str, few_shot: str) -> str:
    if condition == "zero_shot":
        return """ZERO-SHOT CONDITION
--------------------
No demonstration is supplied. Solve the target problem directly.
"""
    if condition == "few_shot":
        return few_shot
    raise ValueError(f"Unknown shot condition: {condition}")


def direct_prompt(question: str, shot_condition: str = "zero_shot") -> str:
    return f"""INPUT
-----
You receive one natural-language operations-research problem.

TASK
----
Determine the optimal value of the objective requested by the statement.

STEP 1: Interpret the problem
------------------------------
Identify the decision quantities, objective direction, units, domains, and
constraints needed to answer the stated question. Preserve the original data.

STEP 2: Solve the target instance
----------------------------------
Reason internally and compute the requested objective. Do not return a model,
Python code, or an unrequested solution vector.

STEP 3: Produce the final output
--------------------------------
Return one finite numeric value using the exact marker below.

OUTPUT CONTRACT
---------------
Return exactly one line and nothing else:
FINAL_OBJECTIVE=<finite numeric value>

{_shot_block(shot_condition, FEW_SHOT_DIRECT)}
TARGET PROBLEM
--------------
{question}
"""


def code_prompt(question: str, shot_condition: str = "zero_shot") -> str:
    return f"""INPUT
-----
You receive one natural-language operations-research problem.

TASK
----
Translate the problem into one complete Python program using gurobipy and solve
the requested objective.

STEP 1: Build the model
-----------------------
Extract the sets, parameters, decision variables and domains, objective, and
constraints. Encode every numerical datum from the statement directly.

STEP 2: Apply implementation rules
-----------------------------------
- Import gurobipy as gp and GRB.
- Create the model with the runtime-provided environment exactly as
  gp.Model(env=env).
- Preserve objective direction, units, indexing, integrality, and all logical
  or temporal restrictions.
- Call model.optimize().

STEP 3: Produce the final output
--------------------------------
Return only one Python program, optionally inside one python code fence. If and
only if the requested solution is available and optimal, print exactly one line
FINAL_OBJECTIVE=<finite numeric value>.

RULES
-----
Do not read or write files, access the network, spawn processes, install
packages, call input(), use dynamic evaluation, or revise the model after an
error. The caller performs policy checking and solver verification separately.

{_shot_block(shot_condition, FEW_SHOT_CODE)}
TARGET PROBLEM
--------------
{question}
"""


def formulation_prompt(question: str, shot_condition: str = "zero_shot") -> str:
    return f"""INPUT
-----
You receive one natural-language operations-research problem.

TASK
----
Formulate the optimization problem using the standard five-element structure.

STEP 1: Build the five-element formulation
--------------------------------------------
1. Sets and indices.
2. Parameters, constants, units, and data domains.
3. Decision variables and domains (binary, integer, continuous, and indices).
4. Objective and direction (minimize or maximize).
5. Constraints, including logical, temporal, capacity, balance, and integrality
   requirements.

STEP 2: Apply formulation rules
-------------------------------
- Use standard OR notation such as sum, for-all, and indexed expressions.
- Define every symbol before using it and preserve the statement's exact data.
- Use auxiliary variables and Big-M only when required; define their bounds.
- Do not solve numerically, generate Python code, or introduce assumptions.
- If the statement is ambiguous, record the ambiguity in the formulation text
  without changing the problem or requesting feedback.

STEP 3: Produce the final output
--------------------------------
Return only the six headings below, in this order, with complete content:
Sets and indices; Parameters; Decision variables and domains; Objective and
direction; Constraints; Requested reported quantity.

{_shot_block(shot_condition, FEW_SHOT_FORMULATION)}
TARGET PROBLEM
--------------
{question}
"""


def formulation_code_prompt(
    question: str, formulation: str, shot_condition: str = "zero_shot"
) -> str:
    return f"""INPUT A: ORIGINAL PROBLEM
--------------------------
You receive the original natural-language operations-research problem below.

INPUT B: SUPPLIED FORMULATION
-----------------------------
You also receive a mathematical formulation produced by the preceding stage.

TASK
----
Translate the supplied formulation into one complete Python/gurobipy program
for the original problem.

STEP 1: Read the formulation literally
---------------------------------------
Map its sets, parameters, variables, domains, objective, and constraints to
code. Keep the original data and requested reported quantity.

STEP 2: Apply implementation rules
-----------------------------------
- Import gurobipy as gp and GRB.
- Create the model with the runtime-provided environment exactly as
  gp.Model(env=env).
- Resolve only Python or API syntax needed for implementation. Do not revise,
  repair, reinterpret, or optimize the supplied mathematics.
- Call model.optimize().

STEP 3: Produce the final output
--------------------------------
Return only one Python program, optionally inside one python code fence. If and
only if the requested solution is available and optimal, print exactly one line
FINAL_OBJECTIVE=<finite numeric value>.

RULES
-----
Do not read or write files, access the network, spawn processes, install
packages, call input(), use dynamic evaluation, or send feedback to another
stage. A formulation error that reaches the program remains attributable to
this pipeline.

{_shot_block(shot_condition, FEW_SHOT_FORMULATION_CODE)}
ORIGINAL PROBLEM
----------------
{question}

SUPPLIED FORMULATION
--------------------
{formulation}
"""


def appendix_prompt_documents() -> dict[str, str]:
    """Return the exact prompt messages used by the benchmark."""

    question_placeholder = "{{question}}"
    formulation_placeholder = "{{formulation}}"
    return {
        "system-prompt.txt": SYSTEM,
        "p1-zero-shot.txt": direct_prompt(question_placeholder, "zero_shot"),
        "p1-few-shot.txt": direct_prompt(question_placeholder, "few_shot"),
        "p2-zero-shot.txt": code_prompt(question_placeholder, "zero_shot"),
        "p2-few-shot.txt": code_prompt(question_placeholder, "few_shot"),
        "p3-formulation-zero-shot.txt": formulation_prompt(
            question_placeholder, "zero_shot"
        ),
        "p3-formulation-few-shot.txt": formulation_prompt(
            question_placeholder, "few_shot"
        ),
        "p3-code-zero-shot.txt": formulation_code_prompt(
            question_placeholder, formulation_placeholder, "zero_shot"
        ),
        "p3-code-few-shot.txt": formulation_code_prompt(
            question_placeholder, formulation_placeholder, "few_shot"
        ),
    }
