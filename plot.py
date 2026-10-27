from fitting_script import data_folder

import matplotlib.pyplot as plt
import numpy as np

data_path = (
    data_folder
    / "output_all_vol_integral"
    / "time_separation.txt"
)
with data_path.open(encoding="utf-8") as data_file:
    eccentricity = float(data_file.readline().split("=", 1)[1])
    time, separation = np.loadtxt(data_file, unpack=True)

fig, ax = plt.subplots()
ax.plot(time, separation)
ax.set_xlabel(r"Time ($GM_\odot/c^3$)")
ax.set_ylabel(r"Separation ($GM_\odot/c^2$)")
ax.set_title(f"Star separation versus time (e = {eccentricity:.6f})")
ax.grid(True)
fig.tight_layout()
fig.savefig(data_path.parent / "separation_vs_time.png", dpi=300)
plt.close(fig)
