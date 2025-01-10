import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("sv.csv")

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(3.5,6.5), dpi=600, layout='constrained')

ax = axs[0]
ax.plot(df['Freq (1/cm)'], df['Sv_tot (cm)'], label = "Total")
ax.plot(df['Freq (1/cm)'], df['Sv_trn (cm)'], label = "Trn")
ax.plot(df['Freq (1/cm)'], df['Sv_rot (cm)'], label = "Rot")
ax.plot(df['Freq (1/cm)'], df['Sv_vib (cm)'], label = "Vib")

ax.set_xlim((0, 1000))
#ax.set_ylim((0, 12))
ax.set_xlabel(r"$\nu$ (1/cm)")
ax.set_ylabel(r"S($\nu$) (cm)")
ax.legend()

ax = axs[1]
ax.plot(df['Freq (1/cm)'], df['Sv_trn (cm)'], label = "Trn")
ax.plot(df['Freq (1/cm)'], df['Sv_trn_g (cm)'], label = "Trn(g)")
ax.plot(df['Freq (1/cm)'], df['Sv_trn_s (cm)'], label = "Trn(s)")

ax.set_xlim((0, 500))
#ax.set_ylim((0, 12))
ax.set_xlabel(r"$\nu$ (1/cm)")
ax.set_ylabel(r"S($\nu$) (cm)")

ax.legend()

ax = axs[2]
ax.plot(df['Freq (1/cm)'], df['Sv_rot (cm)'], label = "Rot")
ax.plot(df['Freq (1/cm)'], df['Sv_rot_g (cm)'], label = "Rot(g)")
ax.plot(df['Freq (1/cm)'], df['Sv_rot_s (cm)'], label = "Rot(s)")

ax.set_xlim((0, 1000))
#ax.set_ylim((0, 4))
ax.set_xlabel(r"$\nu$ (1/cm)")
ax.set_ylabel(r"S($\nu$) (cm)")

ax.legend()

fig.savefig('Sv.png')
