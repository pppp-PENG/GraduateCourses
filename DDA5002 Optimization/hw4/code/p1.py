import coptpy as cp
from coptpy import COPT


class Node:
    def __init__(self, model, constr_list=None, name=""):
        self.name = name
        self.model = model
        self.constr_list = constr_list if constr_list else []


def solve_ip():
    record = []
    env = cp.Envr()
    model0 = env.createModel("IP")
    x1 = model0.addVar(lb=0, name="x1")
    x2 = model0.addVar(lb=0, name="x2")
    x3 = model0.addVar(lb=0, name="x3")
    x4 = model0.addVar(lb=0, name="x4")
    model0.setObjective(2 * x1 + 3 * x2 + 4 * x3 + 7 * x4, sense=COPT.MAXIMIZE)
    model0.addConstr(4 * x1 + 6 * x2 - 2 * x3 + 8 * x4 == 20)
    model0.addConstr(x1 + 2 * x2 - 6 * x3 + 7 * x4 == 10)

    best_z = -float("inf")
    best_sol = None
    nodes = [Node(model0, name="root")]

    while nodes:
        node = nodes.pop()
        model = node.model
        model.solve()
        if model.status != COPT.OPTIMAL:
            record.append(f"node {node.name} is infeasible, backtrack")
            continue
        record.append(f"node {node.name} optimal value: {model.objval}")
        z = model.objval
        x1_val = model.getVarByName("x1").x
        x2_val = model.getVarByName("x2").x
        x3_val = model.getVarByName("x3").x
        x4_val = model.getVarByName("x4").x
        record.append(
            f"    solution: z={z}, x1={x1_val}, x2={x2_val}, x3={x3_val}, x4={x4_val}"
        )

        # check solution
        if all(val.is_integer() for val in [x1_val, x2_val, x3_val, x4_val]):
            if z > best_z:
                best_z = z
                best_sol = (x1_val, x2_val, x3_val, x4_val)
            continue

        if not x1_val.is_integer():
            print("branching on x1")
            model_left = model.clone()
            x1_left = model_left.getVarByName("x1")
            model_left.addConstr(x1_left <= int(x1_val))
            nodes.append(Node(model_left, name=f"left(x1<={int(x1_val)})"))

            model_right = model.clone()
            x1_right = model_right.getVarByName("x1")
            model_right.addConstr(x1_right >= int(x1_val) + 1)
            nodes.append(Node(model_right, name=f"right(x1>={int(x1_val)+1})"))

        elif not x2_val.is_integer():
            print("branching on x2")
            model_left = model.clone()
            x2_left = model_left.getVarByName("x2")
            model_left.addConstr(x2_left <= int(x2_val))
            nodes.append(Node(model_left, name=f"left(x2<={int(x2_val)})"))

            model_right = model.clone()
            x2_right = model_right.getVarByName("x2")
            model_right.addConstr(x2_right >= int(x2_val) + 1)
            nodes.append(Node(model_right, name=f"right(x2>={int(x2_val)+1})"))

        elif not x3_val.is_integer():
            print("branching on x3")
            model_left = model.clone()
            x3_left = model_left.getVarByName("x3")
            model_left.addConstr(x3_left <= int(x3_val))
            nodes.append(Node(model_left, name=f"left(x3<={int(x3_val)})"))

            model_right = model.clone()
            x3_right = model_right.getVarByName("x3")
            model_right.addConstr(x3_right >= int(x3_val) + 1)
            nodes.append(Node(model_right, name=f"right(x3>={int(x3_val)+1})"))

        elif not x4_val.is_integer():
            print("branching on x4")
            model_left = model.clone()
            x4_left = model_left.getVarByName("x4")
            model_left.addConstr(x4_left <= int(x4_val))
            nodes.append(Node(model_left, name=f"left(x4<={int(x4_val)})"))

            model_right = model.clone()
            x4_right = model_right.getVarByName("x4")
            model_right.addConstr(x4_right >= int(x4_val) + 1)
            nodes.append(Node(model_right, name=f"right(x4>={int(x4_val)+1})"))

    if best_sol:
        print(
            f"optimal solution: z={best_z}, x1={best_sol[0]}, x2={best_sol[1]}, x3={best_sol[2]}, x4={best_sol[3]}"
        )
    else:
        print("no feasible integer solution found")

    return record


record = solve_ip()
for line in record:
    print(line)
