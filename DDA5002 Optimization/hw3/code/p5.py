from coptpy import *

# Create environment and model
env = Envr()
model = env.createModel("Course_Selection")

# Data
course_num = 7

# Decision variables
x = [model.addVar(name=f"x_{i}", vtype=COPT.BINARY) for i in range(course_num)]

# Constraints
model.addConstr(x[0] + x[1] + x[2] + x[3] + x[6] >= 2, name="Math_constr")
model.addConstr(x[1] + x[3] + x[4] + x[6] >= 2, name="OR_constr")
model.addConstr(x[2] + x[4] + x[5] >= 2, name="Computer_constr")
model.addConstr(x[2] <= x[5], name="prereq_constr1")
model.addConstr(x[3] <= x[0], name="prereq_constr2")
model.addConstr(x[4] <= x[5], name="prereq_constr3")
model.addConstr(x[6] <= x[3], name="prereq_constr4")

# Objective function
courses = sum(x[i] for i in range(course_num))
model.setObjective(courses, COPT.MINIMIZE)

# Solve
model.solve()

# Report
print("(b): ")
for i in range(course_num):
    if x[i].x > 0.5:
        print(f"Choose course {i + 1}")
print(f"Optimal courses choice: {model.objval}")
