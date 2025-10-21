"""
agent_visualization.py — Agent profile heterogeneity visualization
Author: [Your Name]
Description:
    Generates three publication-quality visualizations for agent profiles:
    1. Radar chart (trait comparison)
    2. Bar chart (speed vs hazard sensitivity)
    3. Pie chart (population ratio)
"""

import matplotlib.pyplot as plt
import numpy as np

# ================================
# 🎨 Project Colour Palette (Tailwind-inspired)
# ================================
palette = {
    "Licorice": "#220901",
    "Blood Red": "#621708",
    "Ultra Violet": "#5b618a",
    "Syracuse Red Orange": "#d74009",
    "Orange Web": "#f6aa1c"
}

# ================================
# 🧠 Agent Profile Definitions
# ================================
profiles = {
    'cautious': {
        'speed_multiplier': 0.83,
        'risk_tolerance': 0.1,
        'w_hazard_multiplier': 1.5,
        'lookahead_multiplier': 1.5,
        'profile_multiplier': 1.5
    },
    'average': {
        'speed_multiplier': 1.0, 
        'risk_tolerance': 0.3,
        'w_hazard_multiplier': 1.0,
        'lookahead_multiplier': 1.0,
        'profile_multiplier': 1.0
    },
    'bold': {
        'speed_multiplier': 1.17,
        'risk_tolerance': 0.6,
        'w_hazard_multiplier': 0.5,
        'lookahead_multiplier': 0.5,
        'profile_multiplier': 0.5
    },
}

ratios = {'cautious': 0.3, 'average': 0.5, 'bold': 0.2}

# Derived hazard weight (for visual comparison)
for name, vals in profiles.items():
    vals['w_hazard'] = 15 * (1 - vals['risk_tolerance']) * vals['w_hazard_multiplier']


# # ================================
# # 🧭 1. RADAR CHART — Profile Traits
# # ================================
# plt.style.use("default")
# plt.rcParams.update({
#     "font.family": "DejaVu Sans",
#     "axes.edgecolor": palette["Licorice"],
#     "axes.labelcolor": palette["Licorice"],
#     "axes.titleweight": "bold",
#     "axes.grid": True,
#     "grid.alpha": 0.25,
#     "grid.linestyle": "--",
#     "figure.facecolor": "white",
#     "axes.facecolor": "white",
#     "xtick.color": palette["Licorice"],
#     "ytick.color": palette["Licorice"],
# })

# params = ['speed_multiplier', 'risk_tolerance', 'w_hazard_multiplier',
#           'lookahead_multiplier', 'profile_multiplier']
# num_vars = len(params)

# angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
# angles += angles[:1]  # close the loop

# fig = plt.figure(figsize=(8, 8))
# ax = plt.subplot(111, polar=True)

# colors = [palette["Blood Red"], palette["Ultra Violet"], palette["Orange Web"]]

# for (name, vals), color in zip(profiles.items(), colors):
#     values = [vals[p] for p in params]
#     values += values[:1]
#     ax.plot(angles, values, color=color, linewidth=2.5, label=name.capitalize())
#     ax.fill(angles, values, color=color, alpha=0.25)

# # Style radar
# ax.set_xticks(angles[:-1])
# ax.set_xticklabels(
#     [p.replace("_", " ").title() for p in params],
#     fontsize=11, color=palette["Licorice"], weight="bold"
# )
# ax.set_yticklabels([])
# ax.spines["polar"].set_color(palette["Licorice"])
# ax.spines["polar"].set_linewidth(1.2)

# ax.set_title("Protester Profile Traits", size=16, pad=20,
#              color=palette["Licorice"], weight='bold')
# ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), frameon=False)
# plt.tight_layout()
# plt.show()


# ================================
# 🚀 2. BAR CHART — Speed vs Hazard Sensitivity
# ================================
names = list(profiles.keys())
speed = [profiles[n]['speed_multiplier'] for n in names]
hazard = [profiles[n]['w_hazard'] for n in names]

x = np.arange(len(names))
width = 0.38

fig, ax = plt.subplots(figsize=(8, 6))
bars1 = ax.bar(x - width/2, speed, width, label='Speed Multiplier',
               color=palette["Ultra Violet"], edgecolor=palette["Licorice"], alpha=0.9)
bars2 = ax.bar(x + width/2, hazard, width, label='Hazard Weight',
               color=palette["Syracuse Red Orange"], edgecolor=palette["Licorice"], alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels([n.capitalize() for n in names], fontsize=11, weight='bold', color=palette["Licorice"])
ax.set_ylabel("Value", fontsize=12, color=palette["Licorice"])
ax.set_title("Speed vs Hazard Sensitivity by Profile",
             color=palette["Licorice"], weight='bold', fontsize=14, pad=15)
ax.legend(frameon=False)
ax.grid(axis='y', alpha=0.3)

# Annotate bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color=palette["Licorice"])

plt.tight_layout()
plt.show()


# ================================
# 🥧 3. PIE CHART — Agent Ratios
# ================================
fig, ax = plt.subplots(figsize=(6, 6))
colors = [palette["Blood Red"], palette["Ultra Violet"], palette["Orange Web"]]
wedges, texts, autotexts = ax.pie(
    ratios.values(),
    labels=[k.capitalize() for k in ratios.keys()],
    autopct='%1.0f%%',
    startangle=140,
    colors=colors,
    textprops={'weight': 'bold', 'fontsize': 12, 'color': 'white'},
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.2}
)

# Enhance text contrast
for text in texts:
    text.set_color(palette["Licorice"])
    text.set_fontweight("bold")

ax.set_title("Protester Population Ratios", color=palette["Licorice"],
             weight='bold', fontsize=14, pad=15)
plt.tight_layout()
plt.show()
