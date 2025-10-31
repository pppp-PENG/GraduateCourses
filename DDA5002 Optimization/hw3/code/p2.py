from coptpy import *

# Create environment and model
env = Envr()
model = env.createModel("LGUAirline_Seating")

# Data
scenarios = 3
seat_types = 3
max_ticket = [[20, 50, 200], [10, 25, 175], [5, 10, 150]]

# Decision variables
x = [model.addVar(lb=0, name=f"x_{t}") for t in range(seat_types)]

# Other variables
n = [
    [model.addVar(lb=0, name=f"n_{s}_{t}") for t in range(seat_types)]
    for s in range(scenarios)
]

# Constraints
model.addConstr(2 * x[0] + 1.5 * x[1] + x[2] <= 200, name="capacity") # capacity constraint
for s in range(scenarios): # demand and supply constraints
    for t in range(seat_types):
        model.addConstr(n[s][t] <= x[t], name=f"supply^{s}_{t}")
        model.addConstr(n[s][t] <= max_ticket[s][t], name=f"demand^{s}_{t}")

# Objective function
profit = (1 / 3) * sum(3 * n[s][0] + 2 * n[s][1] + n[s][2] for s in range(scenarios))
model.setObjective(profit, COPT.MAXIMIZE)

# Solve
model.solve()

# Report
print("(b): ")
print(f"Optimal partition: x*=({x[0].x:.4f}, {x[1].x:.4f}, {x[2].x:.4f})")
for s in range(scenarios):
    print(
        f"In Scenario {s+1}: Sell first-class:{n[s][0].x:.4f}, business-class:{n[s][1].x:.4f}, coach-fare:{n[s][2].x:.4f}"
    )
print(f"Optimal profit: {model.objval:.4f}")

print("(c): ")
capacity_constr = model.getConstrByName("capacity")
y_star = capacity_constr.pi
print(f"Shadow price of capacity: {y_star:.4f}")

print("(d): ")
z_star = model.objval + y_star * (201 - 200)
print(f"z_star: {z_star:.4f}")

print("(e): ")
capacity_constr.remove()
model.addConstr(2 * x[0] + 1.5 * x[1] + x[2] <= 201, name="new capacity")
model.solve()
print(f"The new optimal profit: {model.objval:.4f}")

print("(f): ")
shadow_price = [[0.0 for _ in range(seat_types)] for _ in range(scenarios)]

for s in range(scenarios):
    for t in range(seat_types):
        constr = model.getConstrByName(f"demand^{s}_{t}")
        shadow_price[s][t] = constr.pi

for s in range(scenarios):
    print(
        f"Scenario {s + 1}: Shadow price for first-class, business-class and coach-fare are {shadow_price[s][0]:.4f}, {shadow_price[s][1]:.4f}, {shadow_price[s][2]:.4f} respectively."
    )
