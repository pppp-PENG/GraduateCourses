from coptpy import *

# Create environment and model
env = Envr()
model = env.createModel("TSP_5holes")

# Data
hole_num = 5
d = [
    [0, 13, 8, 15, 7],
    [13, 0, 5, 7, 14],
    [8, 5, 0, 15, 17],
    [15, 7, 15, 0, 8],
    [7, 14, 17, 8, 0],
] # Distance matrix
S = [
    {0},
    {1},
    {2},
    {3},
    {4},
    {0, 1},
    {0, 2},
    {0, 3},
    {0, 4},
    {1, 2},
    {1, 3},
    {1, 4},
    {2, 3},
    {2, 4},
    {3, 4},
    {0, 1, 2},
    {0, 1, 3},
    {0, 1, 4},
    {0, 2, 3},
    {0, 2, 4},
    {0, 3, 4},
    {1, 2, 3},
    {1, 2, 4},
    {1, 3, 4},
    {2, 3, 4},
    {0, 1, 2, 3},
    {0, 1, 2, 4},
    {0, 1, 3, 4},
    {0, 2, 3, 4},
    {1, 2, 3, 4},
] # Subtour elimination sets

# Decision variables
x = [
    [model.addVar(name=f"x_{i}{j}", vtype=COPT.BINARY) for i in range(hole_num)]
    for j in range(hole_num)
]

# Constraints
for i in range(hole_num): # Each row and column sums to 1
    model.addConstr(sum(x[:][i]) == 1, name="column_sum")
    model.addConstr(sum(x[i][:]) == 1, name="row_sum")
for s in S: # Subtour elimination constraints
    model.addConstr(
        sum(x[i][j] for i in s for j in s) <= len(s) - 1,
        name=f"subtour_elimination_{s}",
    )

# Objective function
distance = sum(d[i][j] * x[i][j] for i in range(hole_num) for j in range(hole_num))
model.setObjective(distance, COPT.MINIMIZE)

# Solve
model.solve()

# Report
print("(b): ")
for i in range(hole_num):
    for j in range(hole_num):
        if x[i][j].x > 0.5:
            print(f"From hole {i+1} to hole {j+1}")
print(f"Optimal distance: {model.objval}")
