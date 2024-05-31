import csv
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

deg = int(input("input degree:"))
csv_file_path = f"./data/{deg}.csv"
csv_labels = []
csv_data = []
with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
    csv_reader = csv.reader(csvfile)
    l = [row for row in csv_reader]
    csv_labels = [l[i][0] for i in range(len(l))]
    csv_data = [l[i][1:] for i in range(len(l))]

time_len = len(csv_data[0])

time_diff = 0.05
t_ticks = np.arange(0, 0.45, 0.05)

x_ticks = np.arange(0, 0.30, 0.05)
v_ticks = np.arange(0, 1.6, 0.2)
a_ticks = np.arange(0, 8.0, 1.0)

# x-tグラフ
plt.figure()
x_data = np.array(csv_data, dtype="float32") / 100
x_mean = np.mean(x_data, axis=0)
x_axis = 0 + np.arange(time_len) * time_diff
p2 = np.polyfit(x_axis, x_mean, 2)
quad_func = p2[0] * (x_axis**2) + p2[1] * x_axis + p2[2]
plt.plot(x_axis, x_mean, marker="o", label="計測値の平均")
plt.plot(
    x_axis,
    quad_func,
    label=f"計測値の平均の二次関数近似\n(x = {round(p2[0],2)}t^2 + {round(p2[1],2)}t + {round(p2[2],2)})",
)

for i in range(x_data.shape[0]):
    plt.plot(
        x_axis,
        x_data[i],
        linestyle=":",
        alpha=0.6,
        marker="o",
        label=f"計測値{i+1}回目",
    )
plt.xlabel("時間[s]")
plt.ylabel("進んだ距離[m]")
plt.title(f"θ={deg}°のときの進んだ距離と時間の関係")
plt.legend()
plt.xticks(t_ticks)
plt.yticks(x_ticks)
plt.savefig(f"{deg}-x-t.png")
plt.show()

# v-t グラフ
plt.figure()
v_data = np.diff(x_data, n=1, axis=-1) / 0.05
v_mean = np.diff(x_mean, n=1, axis=-1) / 0.05
v_axis = time_diff / 2 + np.arange(time_len - 1) * time_diff
plt.plot(v_axis, v_mean, marker="o", label="距離の計測値の平均から算出した速度")
p1 = np.polyfit(v_axis, v_mean, 1)
linear_func = p1[0] * v_axis + p1[1]
plt.plot(
    v_axis,
    linear_func,
    label=f"計測値の平均の一次関数近似\n(v = {round(p1[0],2)}t + {round(p1[1],2)})",
)
for i in range(v_data.shape[0]):
    plt.plot(
        v_axis,
        v_data[i],
        linestyle=":",
        alpha=0.6,
        marker="o",
        label=f"計測値{i+1}回目",
    )

plt.xlabel("時間[s]")
plt.ylabel("速度[m/s]")
plt.title(f"θ={deg}°のときの速度と時間の関係")
plt.legend()
plt.xticks(t_ticks)
plt.yticks(v_ticks)
plt.savefig(f"{deg}-v-t.png")
plt.show()

# a-t グラフ
plt.figure()
a_axis = time_diff + np.arange(time_len - 2) * time_diff
a_data = np.diff(v_data, n=1, axis=-1) / 0.05
a_mean = np.diff(v_mean, n=1, axis=-1) / 0.05
plt.plot(a_axis, a_mean, marker="o", label="距離の計測値の平均から算出した加速度")
p0 = np.polyfit(a_axis, a_mean, 0)
zero_func = a_axis * 0 + p0[0]
plt.plot(
    a_axis,
    zero_func,
    label=f"計測値の平均の定数関数近似\n(a = {round(p0[0],2)})",
)
for i in range(a_data.shape[0]):
    plt.plot(
        a_axis,
        a_data[i],
        linestyle=":",
        alpha=0.6,
        marker="o",
        label=f"計測値{i+1}回目",
    )

plt.xlabel("時間[s]")
plt.ylabel("加速度[m/s²]")
plt.title(f"θ={deg}°のときの加速度と時間の関係")
plt.legend()
plt.xticks(t_ticks)
plt.yticks(a_ticks)
plt.savefig(f"{deg}-a-t.png")
plt.show()
