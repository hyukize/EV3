import time
import numpy as np
import tkinter as tk
from PIL import Image
from PIL.ImageTk import PhotoImage
import pickle
import threading
import tkinter.font as tk_font
from typing import List
from sys import exit

LR = 0.1
TIME_STEP = 0.2
GAMMA = 0.9
EPSILON = 0.8
DECAY = 0.999

INITIAL_REWARD = 70
TOTAL_EPISODE = 500

UNIT = 140
HEIGHT = 5
WIDTH = 5

ENDED = False


class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()

        self.time_step = TIME_STEP
        self.learning_rate = LR
        self.gamma = GAMMA
        self.decay = DECAY
        self.epsilon = EPSILON
        self.total_episode = TOTAL_EPISODE
        self.episode = 1
        self.initial_reward = INITIAL_REWARD
        self.total_reward = INITIAL_REWARD
        self.width = WIDTH
        self.height = HEIGHT
        self.unit = UNIT
        self.action_space = ['u', 'd', 'l', 'r']
        self.actions = [0, 1, 2, 3]
        self.action_dict = {0: "up", 1: "down", 2: "left", 3: "right"}
        self.n_actions = len(self.action_space)
        self.texts = []
        self.q_table = {}
        self.start_point = []
        self.end_points = []
        self.obstacles = []

        self._build_initial_canvas()

        self._build_setting_canvas()

        self._make_q_table()

        self.canvas = self._build_canvas()

    def _make_q_table(self):
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) not in self.obstacles:
                    self.q_table[str([i, j])] = [.0] * len(self.actions)

    def _x_button_handler(self):
        self.destroy()
        exit()

    def _destroy(self, event):
        self.destroy()

    def _check_condition(self):
        if self.end_points and len(self.start_point) == 1:
            self.destroy()
        else:
            pass

    def _set_position(self, i, j):
        if self.buttons[i][j]["text"] == "None":
            self.buttons[i][j]["text"] = "obs"
            self.buttons[i][j]["image"] = self.images[1]
            self.obstacles.append((i, j))

        elif self.buttons[i][j]["text"] == "obs":
            self.buttons[i][j]["text"] = "start"
            self.buttons[i][j]["image"] = self.images[4]
            self.obstacles.remove((i, j))
            self.start_point.append((i, j))

        elif self.buttons[i][j]["text"] == "start":
            self.buttons[i][j]["text"] = "end"
            self.buttons[i][j]["image"] = self.images[2]
            self.start_point.remove((i, j))
            self.end_points.append((i, j))

        elif self.buttons[i][j]["text"] == "end":
            self.buttons[i][j]["text"] = "None"
            self.buttons[i][j].config(image="")
            self.end_points.remove((i, j))

    def _set_width(self, event):
        width = self.width_entry.get()
        if width.isdigit() and 3 <= int(width) <= 9:
            self.width = int(width)
            self.width_label1.config(text=f"가로 : {self.width}")
            self.setting_button.focus()
        else:
            print("error")

    def _set_height(self, event):
        height = self.height_entry.get()
        if height.isdigit() and 4 <= int(height) <= 7:
            self.height = int(height)
            self.height_label1.config(text=f"세로 : {self.height}")
            self.width_entry.focus()
        else:
            print("error")

    def _set_episode(self, event):
        episode = self.episode_entry.get()
        if episode.isdigit():
            self.total_episode = int(episode)
            self.episode_label1.config(text=f"에피소드 : {self.total_episode}")
            self.reward_entry.focus()
        else:
            print("error")

    def _set_reward(self, event):
        reward = self.reward_entry.get()
        if reward.isdigit():
            self.initial_reward = int(reward)
            self.total_reward = int(reward)
            self.reward_label1.config(text=f"시작 체력 : {self.initial_reward}")
            self.epsilon_entry.focus()
        else:
            print("error")

    def _set_epsilon(self, event):
        epsilon = self.epsilon_entry.get()
        try:
            if 0 <= float(epsilon) <= 1:
                self.epsilon = float(epsilon)
                self.epsilon_label1.config(text=f"Epsilon : {self.epsilon}")
                self.decay_entry.focus()
            else:
                print("error")
        except:
            print("error")

    def _set_decay(self, event):
        decay = self.decay_entry.get()
        try:
            if 0 <= float(decay) <= 1:
                self.decay = float(decay)
                self.decay_label1.config(text=f"Decay : {self.decay}")
                self.sim_start_button.focus()
            else:
                print("error")
        except:
            print("error")

    def _build_initial_canvas(self):
        self.title('Q Learning Simulator --- made by hyuk')
        self.fontStyle1 = tk_font.Font(family="맑은 고딕", size=30)
        self.fontStyle2 = tk_font.Font(family="맑은 고딕", size=13)
        self.fontStyle3 = tk_font.Font(family="맑은 고딕", size=17)
        self.configure(bg="#4682B4")
        self.protocol('WM_DELETE_WINDOW', self._x_button_handler)  # root is your root window
        self.geometry("600x400")

        self.height_entry = tk.Entry(self, width=5, fg="#4682B4", bg="white", font=self.fontStyle3)
        self.height_entry.bind("<Return>", self._set_height)
        self.height_entry.place(x=130, y=50)

        self.height_label1 = tk.Label(self, text=f"세로 : {self.height}  ", bg="#4682B4", fg="white",
                                      font=self.fontStyle1)
        self.height_label1.place(x=360, y=35)

        self.height_label2 = tk.Label(self, text=f"세로 :", bg="#4682B4", fg="white", font=self.fontStyle3)
        self.height_label2.place(x=60, y=50)

        self.height_label3 = tk.Label(self, text="(4 ~ 7)", bg="#4682B4", fg="white", font=self.fontStyle2)
        self.height_label3.place(x=210, y=53)

        self.width_entry = tk.Entry(self, width=5, fg="#4682B4", bg="white", font=self.fontStyle3)
        self.configure(bg="#4682B4")
        self.width_entry.bind("<Return>", self._set_width)
        self.width_entry.place(x=130, y=150)

        self.width_label1 = tk.Label(self, text=f"가로 : {self.width}  ", bg="#4682B4", fg="white",
                                     font=self.fontStyle1)
        self.width_label1.place(x=360, y=135)

        self.width_label2 = tk.Label(self, text=f"가로 :", bg="#4682B4", fg="white", font=self.fontStyle3)
        self.width_label2.place(x=60, y=150)

        self.width_label3 = tk.Label(self, text="(3 ~ 9)", bg="#4682B4", fg="white", font=self.fontStyle2)
        self.width_label3.place(x=210, y=153)

        self.setting_button = tk.Button(self, text='학습 설정', activebackground="#6495ED", activeforeground="white",
                                        bg="#4682B4", fg="white", command=self.destroy, font=self.fontStyle1)
        self.setting_button.place(x=200, y=280)
        self.setting_button.bind('<Return>', self._destroy)

        self.mainloop()

    def _build_setting_canvas(self):
        super(Env, self).__init__()
        self.fontStyle1 = tk_font.Font(family="맑은 고딕", size=30)
        self.fontStyle2 = tk_font.Font(family="맑은 고딕", size=13)
        self.fontStyle3 = tk_font.Font(family="맑은 고딕", size=17)
        self.title('Q Learning Simulator --- made by hyuk')
        self.configure(bg="#4682B4")
        self.images = self._load_images()
        self.geometry(str((self.width + 4) * UNIT) + "x" + str(max(5, self.height) * UNIT))
        self.resizable(False, False)
        self.buttons: List[List[tk.Button]] = [[] for i in range(self.height)]

        for i in range(self.height):
            for j in range(self.width):
                frame = tk.Frame(self, width=UNIT, height=UNIT)
                button = tk.Button(frame, text="None", image="", activebackground="#6495ED", activeforeground="white",
                                   bg="#4682B4", fg="white", command=lambda a=i, b=j: self._set_position(a, b))
                frame.grid_propagate(False)
                frame.columnconfigure(0, weight=1)
                frame.rowconfigure(0, weight=1)
                frame.grid(row=i, column=j)
                button.grid(sticky="wens")
                self.buttons[i].append(button)

        self.episode_entry = tk.Entry(self, width=5, fg="#4682B4", bg="white", font=self.fontStyle3)
        self.episode_entry.bind("<Return>", self._set_episode)
        self.episode_entry.place(x=int((self.width + 2) * UNIT) - 140, y=50)

        self.episode_label1 = tk.Label(self, text=f"에피소드 : {self.total_episode}  ", bg="#4682B4", fg="white",
                                       font=self.fontStyle1)
        self.episode_label1.place(x=int((self.width + 2) * UNIT) - 30, y=35)

        self.episode_label2 = tk.Label(self, text=f"에피소드 :", bg="#4682B4", fg="white", font=self.fontStyle3)
        self.episode_label2.place(x=int((self.width + 2) * UNIT) - 250, y=50)

        self.reward_entry = tk.Entry(self, width=5, fg="#4682B4", bg="white", font=self.fontStyle3)
        self.reward_entry.bind("<Return>", self._set_reward)
        self.reward_entry.place(x=int((self.width + 2) * UNIT) - 140, y=120)

        self.reward_label1 = tk.Label(self, text=f"시작 체력 : {self.initial_reward}  ", bg="#4682B4", fg="white",
                                      font=self.fontStyle1)
        self.reward_label1.place(x=int((self.width + 2) * UNIT) - 30, y=105)

        self.reward_label2 = tk.Label(self, text=f"시작 체력 :", bg="#4682B4", fg="white", font=self.fontStyle3)
        self.reward_label2.place(x=int((self.width + 2) * UNIT) - 250, y=120)

        self.epsilon_entry = tk.Entry(self, width=5, fg="#4682B4", bg="white", font=self.fontStyle3)
        self.epsilon_entry.bind("<Return>", self._set_epsilon)
        self.epsilon_entry.place(x=int((self.width + 2) * UNIT) - 140, y=190)

        self.epsilon_label1 = tk.Label(self, text=f"Epsilon : {self.epsilon}  ", bg="#4682B4", fg="white",
                                       font=self.fontStyle1)
        self.epsilon_label1.place(x=int((self.width + 2) * UNIT) - 30, y=175)

        self.epsilon_label2 = tk.Label(self, text=f"Epsilon :", bg="#4682B4", fg="white", font=self.fontStyle3)
        self.epsilon_label2.place(x=int((self.width + 2) * UNIT) - 250, y=190)

        self.decay_entry = tk.Entry(self, width=5, fg="#4682B4", bg="white", font=self.fontStyle3)
        self.decay_entry.bind("<Return>", self._set_decay)
        self.decay_entry.place(x=int((self.width + 2) * UNIT) - 140, y=260)

        self.decay_label1 = tk.Label(self, text=f"Decay : {self.decay}  ", bg="#4682B4", fg="white",
                                     font=self.fontStyle1)
        self.decay_label1.place(x=int((self.width + 2) * UNIT) - 30, y=245)

        self.decay_label2 = tk.Label(self, text=f"Decay :", bg="#4682B4", fg="white", font=self.fontStyle3)
        self.decay_label2.place(x=int((self.width + 2) * UNIT) - 250, y=260)

        self.sim_start_button = tk.Button(self, text='시뮬레이션 시작', activebackground="#6495ED",
                                          activeforeground="white",
                                          bg="#4682B4", fg="white", command=self._check_condition, font=self.fontStyle1)
        self.sim_start_button.place(x=(self.width + 0.9) * UNIT, y=480)
        self.mainloop()

    def _build_canvas(self):
        super(Env, self).__init__()
        self.fontStyle1 = tk_font.Font(family="맑은 고딕", size=20)
        self.fontStyle2 = tk_font.Font(family="맑은 고딕", size=13)
        self.fontStyle3 = tk_font.Font(family="맑은 고딕", size=17)
        self.title('Q Learning Simulator --- made by hyuk')
        self.configure(bg="white")
        self.images = self._load_images()
        self.geometry('{0}x{1}'.format((self.width + 4) * UNIT, max(5, self.height) * UNIT))
        self.resizable(False, False)
        canvas = tk.Canvas(self, bg="#4682B4", width=(self.width + 4) * UNIT, height=max(self.height, 5) * UNIT)

        canvas.create_image(0, 0, image=self.images[3])

        base_height = int(1.1 * UNIT)
        self.episode_label = tk.Label(self, bg="#4682B4", fg="white",
                                      text=f"Episode : {self.episode} / {self.total_episode}",
                                      font=self.fontStyle1)
        self.episode_label.place(x=int((self.width + 2) * UNIT) - 110, y=20)

        self.epsilon_label = tk.Label(self, bg="#4682B4", fg="white", text=f"Epsilon : {self.epsilon : .3f}",
                                      font=self.fontStyle3)
        self.epsilon_label.place(x=int((self.width + 2) * UNIT) - 75, y=90)

        self.time_scale = tk.Scale(self, activebackground="#483D8B", troughcolor="white", bg="#4682B4", fg="white",
                                   command=self._change_time_step, orient="horizontal", showvalue=False,
                                   tickinterval=1, to=20, length=int(UNIT * 3.5))
        self.time_scale.set(int(self.time_step * 10))
        self.time_scale.place(x=int((self.width + 0.25) * UNIT), y=base_height)

        self.time_label = tk.Label(self, bg="#4682B4", fg="white", text=f"Time Step : {self.time_step} 초",
                                   font=self.fontStyle2)
        self.time_label.place(x=int((self.width + 2) * UNIT) - 65, y=base_height + 50)

        self.lr_scale = tk.Scale(self, activebackground="#483D8B", troughcolor="white", bg="#4682B4", fg="white",
                                 command=self._change_lr, orient="horizontal", showvalue=False, tickinterval=1,
                                 to=10, length=int(UNIT * 3.5))
        self.lr_scale.set(int(self.learning_rate * 10))
        self.lr_scale.place(x=int((self.width + 0.25) * UNIT), y=base_height + 110)

        self.lr_label = tk.Label(self, bg="#4682B4", fg="white", text=f"Learning Rate : {self.time_step}",
                                 font=self.fontStyle2)
        self.lr_label.place(x=int((self.width + 2) * UNIT) - 70, y=base_height + 160)

        self.gamma_scale = tk.Scale(self, activebackground="#483D8B", troughcolor="white", bg="#4682B4", fg="white",
                                    command=self._change_gamma, orient="horizontal", showvalue=False,
                                    tickinterval=1, to=10, length=int(UNIT * 3.5))
        self.gamma_scale.set(int(self.gamma * 10))
        self.gamma_scale.place(x=int((self.width + 0.25) * UNIT), y=base_height + 220)

        self.gamma_label = tk.Label(self, bg="#4682B4", fg="white", text=f"Gamma : {self.gamma}", font=self.fontStyle2)
        self.gamma_label.place(x=int((self.width + 2) * UNIT) - 40, y=base_height + 270)

        self.reward_label = tk.Label(self, bg="#4682B4", fg="white",
                                     text=f"남은 체력(Total_Reward) : {self.total_reward} / {self.initial_reward}",
                                     font=self.fontStyle3)
        self.reward_label.place(x=int((self.width + 2) * UNIT) - 160, y=base_height + 340)

        for c in range(0, (self.width + 1) * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, self.height * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for r in range(0, (self.height + 1) * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, self.width * UNIT, r
            canvas.create_line(x0, y0, x1, y1)
        self.character_image = canvas.create_image(*self.state_to_cd(self.start_point[0])[::-1], image=self.images[0])
        self.obs_image = [canvas.create_image(*self.state_to_cd(obstacle)[::-1], image=self.images[1]) for obstacle in
                          self.obstacles]
        self.end_image = [canvas.create_image(*self.state_to_cd(end_point)[::-1], image=self.images[2]) for end_point in
                          self.end_points]
        self.start_image = canvas.create_image(*self.state_to_cd(self.start_point[0])[::-1], image=self.images[4])

        canvas.pack()

        return canvas

    def _change_time_step(self, value):
        self.time_step = int(value) / 10.
        value = f"Time Step : {self.time_step} 초"
        self.time_label.config(text=value)

    def _change_lr(self, value):
        self.learning_rate = int(value) / 10.
        value = f"Learning Rate : {self.learning_rate}"
        self.lr_label.config(text=value)

    def _change_gamma(self, value):
        self.gamma = int(value) / 10.
        value = f"Gamma : {self.gamma}"
        self.gamma_label.config(text=value)

    def update_epsilon(self):
        value = f"Epsilon : {self.epsilon : .3f}"
        self.epsilon_label.config(text=value)

    def update_episode(self, episode):
        self.episode = episode
        value = f"Episode : {self.episode} / {self.total_episode}"
        self.episode_label.config(text=value)

    def update_reward(self, total_reward):
        self.total_reward = total_reward
        value = f"남은 체력(Total Reward) : {self.total_reward} / {self.initial_reward}"
        self.reward_label.config(text=value)

    def _load_images(self):
        character_image = PhotoImage(
            Image.open("img/character.png").resize((int(UNIT * 3 / 5), int(UNIT * 3 / 5))))
        obs_image = PhotoImage(
            Image.open("img/obstacle.png").resize((int(UNIT * 4 / 5), int(UNIT * 4 / 5))))
        end_image = PhotoImage(
            Image.open("img/end.png").resize((int(UNIT * 3 / 5), int(UNIT * 3 / 5))))
        bg_image = PhotoImage(
            Image.open("img/bg3.jpg").resize((self.width * UNIT * 2, self.height * UNIT * 2)))
        start_image = PhotoImage(
            Image.open("img/start.png").resize((int(UNIT * 3 / 5), int(UNIT * 3 / 5))))

        return character_image, obs_image, end_image, bg_image, start_image

    def _text_value(self, row, col, contents, action, font='Helvetica', size=10, style='normal', anchor="nw"):
        if action == 0:
            origin_x, origin_y = int(UNIT * 1 / 8 - 5), int(UNIT * 1 / 2 - 10)
        elif action == 1:
            origin_x, origin_y = int(UNIT * 6 / 7 - 5), int(UNIT * 1 / 2 - 10)
        elif action == 2:
            origin_x, origin_y = int(UNIT * 1 / 2 - 5), int(UNIT * 1 / 8 - 10)
        else:
            origin_x, origin_y = int(UNIT * 1 / 2 - 5), int(UNIT * 4 / 5 - 5)

        x, y = origin_y + (UNIT * col), origin_x + (UNIT * row)
        font = (font, str(size), style)
        text = self.canvas.create_text(x, y, fill="black", text=contents, font=font, anchor=anchor)
        return self.texts.append(text)

    def print_value_all(self):
        for i in self.texts:
            self.canvas.delete(i)
        self.texts.clear()
        for i in range(self.height):
            for j in range(self.width):
                for action in range(0, 4):
                    state = str([i, j])
                    if state in self.q_table and (i, j) not in self.end_points:
                        self._text_value(i, j, round(self.q_table[state][action], 2), action)

    @staticmethod
    def cd_to_state(coords):
        x = int((coords[1] - UNIT / 2) / UNIT)
        y = int((coords[0] - UNIT / 2) / UNIT)
        return [y, x]

    @staticmethod
    def state_to_cd(state):
        x = int(state[1] * UNIT + UNIT / 2)
        y = int(state[0] * UNIT + UNIT / 2)
        return [y, x]

    def reset(self):
        start_time = time.time()
        while time.time() - start_time < self.time_step:
            self.render()
        x, y = self.canvas.coords(self.character_image)
        start_coord = self.state_to_cd(self.start_point[0])
        self.canvas.move(self.character_image, start_coord[1] - x, start_coord[0] - y)
        start_time = time.time()
        while time.time() - start_time < self.time_step:
            self.render()
        return self.cd_to_state(self.canvas.coords(self.character_image)[::-1])

    def step(self, action, isend):
        state = self.canvas.coords(self.character_image)[::-1]
        base_action = np.array([0, 0])
        start_time = time.time()
        while time.time() - start_time < self.time_step:
            self.render()

        if action == 0:  # 상
            base_action[0] -= UNIT
        elif action == 1:  # 하
            base_action[0] += UNIT
        elif action == 2:  # 좌
            base_action[1] -= UNIT
        elif action == 3:  # 우
            base_action[1] += UNIT

        next_state = (np.array(state) + base_action).tolist()
        done = False
        collided = False

        if next_state in [self.canvas.coords(end_image)[::-1] for end_image in self.end_image]:
            reward = 50
            done = True

        elif next_state in [self.canvas.coords(obs)[::-1] for obs in self.obs_image]:
            reward = -10
            collided = True

        elif next_state[0] < 0 or next_state[1] < 0 or next_state[0] > UNIT * self.height or next_state[
            1] > UNIT * self.width:
            reward = -5
            collided = True

        else:
            reward = -1
        data = self.action_dict[action]
        if collided:
            data = "obs_" + data
            next_state = state
        else:
            self.canvas.move(self.character_image, base_action[1], base_action[0])
            self.canvas.tag_raise(self.character_image)
        return self.cd_to_state(next_state), reward, done

    def render(self):
        time.sleep(0.03)
        self.update()

    def learn(self, state, action, reward, next_state):
        q_1 = self.q_table[state][action]
        q_2 = reward + self.gamma * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (q_2 - q_1)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions)

        else:
            state_action = self.q_table[state]
            action = self.arg_max(state_action)

        self.epsilon *= self.decay

        if self.epsilon < 0.1:
            self.epsilon = 0
        return action

    def save_data(self):
        with open("data.txt", "wb") as f:
            pickle.dump({"q_table": self.q_table,
                         "learning_rate": self.learning_rate,
                         "gamma": self.gamma,
                         "epsilon": self.epsilon,
                         "decay": self.decay}, f)

    def load_data(self):
        with open("data.txt", "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.learning_rate = data["learning_rate"]
        self.gamma = data["gamma"]
        self.epsilon = data["epsilon"]
        self.decay = data["decay"]

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return np.random.choice(max_index_list)


def main(load_saved_data):
    env = Env()
    if load_saved_data:
        env.load_data()

    for episode in range(TOTAL_EPISODE):
        env.update_episode(episode + 1)
        print(f"episode : {episode + 1}, epsilon : {env.epsilon}")
        if episode == TOTAL_EPISODE - 1:
            is_ended = True
        else:
            is_ended = False
        state = env.reset()
        total_reward = env.initial_reward

        while True:
            env.render()
            action = env.get_action(str(state))
            next_state, reward, done = env.step(action, is_ended)
            total_reward += reward
            env.update_reward(total_reward)
            if total_reward <= 0:
                done = True
            env.update_epsilon()
            env.learn(str(state), action, reward, str(next_state))
            state = next_state
            env.print_value_all()

            if done:
                print(f"Finished!! \ntotal reward : {total_reward}\n\n")
                env.save_data()
                break
    with threading.Lock():
        ENDED = True


if __name__ == "__main__":
    main(load_saved_data=False)
    while True:
        time.sleep(1)
