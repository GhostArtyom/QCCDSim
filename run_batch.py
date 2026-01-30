import subprocess as sp

PATH = "./programs"

PROG = ["BV32", "BV64", "BV128", "GHZ32", "GHZ64", "GHZ128"]

MACHINE = ["G2x3"]

IONS = ["14"]
# for i in range(14, 35, 2):
#     IONS.append(str(i))
# print(IONS)

mapper = "Greedy"
reorder = "Naive"

for p in PROG:
    for m in MACHINE:
        for i in IONS:
            output_file = open(f"./output/{p}_{m}_{i}.log", "w")
            sp.call(["python", "run.py", f"{PATH}/{p}.qasm", m, i, mapper, reorder, "1", "0", "0", "FM", "GateSwap"], stdout=output_file)
