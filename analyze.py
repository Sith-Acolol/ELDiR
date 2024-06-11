# %% [markdown]
# **Imports**

# %%
import pickle, numpy as np, os, seaborn as sns, sys
from matplotlib import pyplot as plt
from tqdm import tqdm
import moviepy.video.io.ImageSequenceClip
from IPython.display import Video
sys.path.append(os.path.dirname(os.path.abspath('')))
from geno_pheno import get_invalid_robots

# %% [markdown]
# **Fill in data directory (`run.py --outdir`)**

# %%
outdir = "./run-out"

# %% [markdown]
# **Compute number of generations**

# %%
subdirs = os.listdir(outdir)
generations = [s for s in subdirs if s.isdigit()]
generations = sorted(generations, key=int)[1:]

# %% [markdown]
# **Collect fitness statistics across generations**

# %%
generational_means = []
generational_stds = []
for g in tqdm(generations):
    loss = np.load(os.path.join(outdir, g, "loss.npy"))
    invalid = get_invalid_robots(loss)
    loss = np.delete(loss, invalid, axis=0)
    loss = np.min(loss, axis=1)
    fitness = loss * -1
    generational_means.append(np.mean(fitness))
    generational_stds.append(np.std(fitness))
generational_means = np.array(generational_means)
generational_stds = np.array(generational_stds)

# %% [markdown]
# **Plot fitness stats**

# %%
plt.figure(figsize=(10, 5))
sns.lineplot(x=generations, y=generational_means)
plt.fill_between(generations, generational_means - generational_stds, generational_means + generational_stds, alpha=0.2)
gen_labels = [int(g) for g in generations]
max_gen = max(gen_labels)
while len(gen_labels) > 10:
    gen_labels = gen_labels[::2]
    if 1 not in gen_labels:
        gen_labels = [1] + gen_labels
    if max_gen not in gen_labels:
        gen_labels = gen_labels + [max_gen]
gen_labels = [str(g) for g in gen_labels]
plt.xticks(gen_labels)
plt.xlabel("Generation")
plt.ylabel("Mean population fitness")
plt.title("Population fitness statistic over generations")

# %% [markdown]
# **Select a generation from which to plot the best robot**

# %%
generation = int(generations[-1])

# %% [markdown]
# **Where to save robot and simulation results for dynamic visualization**

# %%
viz_outdir = "./tmp-viz/gen-{}".format(generation)
print("Visualization output directory: ", viz_outdir)
if not os.path.exists(viz_outdir):
    os.makedirs(viz_outdir)

# %% [markdown]
# **Load the robots**

# %%
with open(os.path.join(outdir, str(generation), "robots.pkl"), "rb") as f:
    robots = pickle.load(f)

# %% [markdown]
# **Select the best**

# %%
best_robot_id = robots['id'][0]
best_robot_points = robots['points'][0]
best_robot_springs = robots['springs'][0]
best_robot = {
    "points": [best_robot_points],
    "springs": [best_robot_springs],
    "id": [best_robot_id],
    "weights": []
}

# %% [markdown]
# **Load trained weights**

# %%
gen, idx = best_robot_id.split("-")
with open(os.path.join(outdir, gen, "weights", idx, "best.pkl"), "rb") as f:
    best_robot["weights"].append(pickle.load(f))

# %% [markdown]
# **Save the phenotype and weights together for simulation**

# %%
robot_save_file = os.path.join(viz_outdir, "best_robot.pkl")
print(f"Saving best robot to {robot_save_file}")
with open(robot_save_file, "wb") as f:
    pickle.dump(best_robot, f)

# %% [markdown]
# **(optional) View the static robot phenotype**

# %%
fig = plt.figure()

green = plt.colormaps['Greens'](0.75)
purple = plt.colormaps['Purples'](0.7)
gray = plt.colormaps['Greys'](0.4)

for j in range(len(best_robot_springs)):
    if best_robot_springs[j][-1] == 0:
        c = gray
    else:
        c = green
    plt.plot([best_robot_points[best_robot_springs[j][0]][0], best_robot_points[best_robot_springs[j][1]][0]],
                [best_robot_points[best_robot_springs[j][0]][1], best_robot_points[best_robot_springs[j][1]][1]], color=c)

for i in range(len(best_robot_points)):
    plt.plot(best_robot_points[i][0], best_robot_points[i][1], 'o', color=purple)

plt.gca().set_aspect('equal', adjustable='box')
plt.axis('off')


# %% [markdown]
# **Command to generate state for dynamic visualization (don't forget to run it!!!)**

# %%
print(f"python sim.py --outdir {viz_outdir} --robots_file {robot_save_file} --visualize")

# %% [markdown]
# **RUN THIS ^^^ !!!!**

# %% [markdown]
# **Load state data for dynamic visualization**

# %%
state_dir = os.path.join(viz_outdir, "state")
x = np.load(os.path.join(state_dir, "x.npy"))[0]
v = np.load(os.path.join(state_dir, "v.npy"))[0]
act = np.load(os.path.join(state_dir, "act.npy"))[0]
spring_actuation = np.load(os.path.join(state_dir, "spring_actuation.npy"))[0]
spring_anchor_a = np.load(os.path.join(state_dir, "spring_anchor_a.npy"))[0]
spring_anchor_b = np.load(os.path.join(state_dir, "spring_anchor_b.npy"))[0]
center = np.load(os.path.join(state_dir, "center.npy"))[0]
loss = np.load(os.path.join(state_dir, "loss.npy"))[0]

# %% [markdown]
# **Make directory to save visualization frames**

# %%
os.makedirs(os.path.join(viz_outdir, "frames"), exist_ok=True)

# %% [markdown]
# **Generate frames**

# %%
x_min, x_max = np.min(x[:, :, 0]), np.max(x[:, :, 0])
y_min, y_max = np.min(x[:, :, 1]), np.max(x[:, :, 1])

gray = plt.colormaps['Greys'](0.4)
greens = plt.colormaps['Greens']
purples = plt.colormaps['Purples']

steps = x.shape[0]

offx = []
offy = []
vx = []
vy = []

for t in range(steps):
    offx.append(x[t, :, 0] - center[t, 0])
    offy.append(x[t, :, 1] - center[t, 1])
    vx.append(v[t, :, 0])
    vy.append(v[t, :, 1])

offx = np.array(offx)
offy = np.array(offy)
vx = np.array(vx)
vy = np.array(vy)

offx = 0.0 + (offx - np.min(offx)) * (1.0 / (np.max(offx) - np.min(offx)))
offy = 0.0 + (offy - np.min(offy)) * (1.0 / (np.max(offy) - np.min(offy)))
vx = 0.25 + (vx - np.min(vx)) * (1.0 / (np.max(vx) - np.min(vx)))
vy = 0.25 + (vy - np.min(vy)) * (1.0 / (np.max(vy) - np.min(vy)))

for t in tqdm(range(steps)):
    fig = plt.figure(figsize=(10, 5))

    for j in range(act.shape[1]):
        if spring_actuation[j] == 0:
            c = gray
        else:
            a = act[t, j]
            c = greens((a + 2) / 3)
        plt.plot([x[t, spring_anchor_a[j], 0], x[t, spring_anchor_b[j], 0]],
                 [x[t, spring_anchor_a[j], 1], x[t, spring_anchor_b[j], 1]], color=c)

    for i in range(x.shape[1]):
        c0 = purples(offx[t, i])
        c1 = purples(offy[t, i])
        c2 = purples(vx[t, i])
        c3 = purples(vy[t, i])
        plt.plot(x[t, i, 0], x[t, i, 1], 'o', color=c2, markersize=8)
        plt.plot(x[t, i, 0], x[t, i, 1], 'o', color=c3, markersize=6)
        plt.plot(x[t, i, 0], x[t, i, 1], 'o', color=c0, markersize=4)
        plt.plot(x[t, i, 0], x[t, i, 1], 'o', color=c1, markersize=2)

    plt.hlines(0.0915, x_min, x_max, color='black')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.xlim(x_min - 0.05, x_max + 0.05)
    plt.ylim(y_min - 0.05, y_max + 0.05)
    plt.text(x_min - 0.015, 0.06, "0.0")
    plt.text(x_max - 0.025, 0.06, f"{np.round(x_max - 0.025, 2)}")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_outdir, "frames", f"{t}.png"))
    plt.close()


# %% [markdown]
# **Create and save video from image frames**

# %%
frames = os.listdir(os.path.join(viz_outdir, "frames"))
frames.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
frames = [os.path.join(viz_outdir, "frames", f) for f in frames]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames, fps=100)
clip.write_videofile(os.path.join(viz_outdir, "sim.mp4"))

# %% [markdown]
# **Load the video**

# %%
Video(os.path.join(viz_outdir, "sim.mp4"), width=800, height=400)

# %%



