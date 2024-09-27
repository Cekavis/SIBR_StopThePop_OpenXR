
import sys, os

if len(sys.argv) != 2:
    print("missing argument")
    exit()

USER_DIR = f"user_{sys.argv[1]}"
if not os.path.exists(USER_DIR):
    print(f"{USER_DIR} does not exist")
    exit()

with open(f"{USER_DIR}/selection.csv") as f:
    print(f.readlines()[-1])

for m_i in range(4):
    for s_i in range(2):
        while (True):
            print(f"Running Method {m_i+1} with config {s_i+1}: {USER_DIR}\\run_{m_i+1}_method_{s_i+1}.bat")
            os.system(f".\\{USER_DIR}\\run_{m_i+1}_method_{s_i+1}.bat >> {USER_DIR}\\log.txt")

            in_txt = input("repeat or next (r/n)? ")
            if (in_txt == "n"):
                break