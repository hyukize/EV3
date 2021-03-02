import time
import numpy as np
import tkinter as tk
from PIL import Image
from PIL.ImageTk import PhotoImage
from collections import deque
import pickle
import threading
from queue import Queue
import tkinter.font as tkFont
import socket



LR = 0.1
TIMESTEP = 0.2
GAMMA = 0.9
EPSILON = 0.8
DECAY = 0.998

INITIAL_REWARD = 50
TOTAL_EPISODE = 100

UNIT = 150  # 픽셀 수
HEIGHT = 5  #  세로
WIDTH = 5  # 가로

START_POINT = (0, 0)
END_POINT = (4, 4)
OBSTACLES = [(2, 1), (2, 2), (1, 3), (3, 4), (4, 2)]

#     GRID INFO
#
#      0 1 2 3 4
#
#  0   S . . . .
#  1   . . . X .
#  2   . X X . .
#  3   . . . . X
#  4   . . X . E

QUEUE1 = Queue()
QUEUE2 = Queue()

ended = False

HOST = '0.0.0.0'
PORT = 8000

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen()
client, addr = server.accept()

print('Connected by', addr)


def handle_socket():
    while True:
        data = QUEUE1.get()
        if data == "episode_ended":
            client.send(b"beep")
        elif data == "return_ended":
            client.send(b"beep")
            if ended:
                print("train ended")
                break
        else:
            client.send(data.encode())
        result = client.recv(1024).decode()
        if result != "done":
            raise ValueError
    
    client.send(b'yes')
    client.recv(1024)
    time.sleep(1)
    while True:
        data = QUEUE2.get()
        if data == "episode_ended":
            client.send(b"beep")
        elif data == "return_ended":
            client.send(b"beep")
        else:
            client.send(data.encode())
        result = client.recv(1024).decode()
        if result != "done":
            raise ValueError


class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()

        self.time_step = TIMESTEP
        self.learning_rate = LR
        self.gamma = GAMMA
        self.decay = DECAY
        self.epsilon = EPSILON
        self.total_episode = TOTAL_EPISODE
        self.episode = 1
        self.total_reward = INITIAL_REWARD
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('Q Learning Simulator --- made by hyuk')
        self.fontStyle1 = tkFont.Font(family="Lucida Grande", size=20)
        self.fontStyle2 = tkFont.Font(family="Lucida Grande", size=13)
        self.fontStyle3 = tkFont.Font(family="Lucida Grande", size=17)
        self.start_image = PhotoImage(Image.open("img/start.png").resize((int(UNIT * 1/2), int(UNIT * 1/2))))
        self.bg_image = PhotoImage(Image.open("img/bg3.jpg").resize((WIDTH * UNIT * 2, HEIGHT * UNIT * 2)))
        self.geometry('{0}x{1}'.format((WIDTH+4) * UNIT, max(5, HEIGHT) * UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.texts = []
        self.action_dict = {0: "up", 1: "down", 2: "left", 3: "right"}
        
    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=max(5, HEIGHT) * UNIT,
                           width=(WIDTH+4) * UNIT)

        canvas.create_image(10, 10, image=self.bg_image)

        base_height = int(1.1 * UNIT)
        self.episode_label = tk.Label(self, text=f"Episode : {self.episode} / {self.total_episode}", font=self.fontStyle1)
        self.episode_label.place(x = int((WIDTH+2)*UNIT) - 100, y = 20)

        self.epsilon_label = tk.Label(self, text=f"Epsilon : {self.epsilon : .3f}", font=self.fontStyle3)
        self.epsilon_label.place(x = int((WIDTH+2)*UNIT) - 75, y = 90)

        self.time_scale=tk.Scale(self, command=self.change_timestep, orient="horizontal", showvalue=False, tickinterval=1, to=20, length=int(UNIT*3.5))
        self.time_scale.set(int(self.time_step * 10))
        self.time_scale.place(x = int((WIDTH+0.25)*UNIT), y = base_height)

        self.time_label = tk.Label(self, text=f"Time Step : {self.time_step} 초", font=self.fontStyle2)
        self.time_label.place(x = int((WIDTH+2)*UNIT) - 65, y = base_height + 50)


        self.lr_scale=tk.Scale(self, command=self.change_lr, orient="horizontal", showvalue=False, tickinterval=1, to=10, length=int(UNIT*3.5))
        self.lr_scale.set(int(self.learning_rate * 10))
        self.lr_scale.place(x = int((WIDTH+0.25)*UNIT), y = base_height + 110)

        self.lr_label = tk.Label(self, text=f"Learning Rate : {self.time_step}", font=self.fontStyle2)
        self.lr_label.place(x = int((WIDTH+2)*UNIT) - 70, y = base_height + 160)


        self.gamma_scale=tk.Scale(self, command=self.change_gamma, orient="horizontal", showvalue=False, tickinterval=1, to=10, length=int(UNIT*3.5))
        self.gamma_scale.set(int(self.gamma * 10))
        self.gamma_scale.place(x = int((WIDTH+0.25)*UNIT), y = base_height + 220)

        self.gamma_label = tk.Label(self, text=f"Gamma : {self.gamma}", font=self.fontStyle2)
        self.gamma_label.place(x = int((WIDTH+2)*UNIT) - 40, y = base_height + 270)

        self.reward_label = tk.Label(self, text=f"남은 체력(Total_Reward) : {self.total_reward}", font=self.fontStyle3)
        self.reward_label.place(x = int((WIDTH+2)*UNIT) - 135, y = base_height + 340)




        for c in range(0, (WIDTH + 1) * UNIT, UNIT): 
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for r in range(0, (HEIGHT + 1) * UNIT, UNIT):  
            x0, y0, x1, y1 = 0, r, WIDTH * UNIT, r
            canvas.create_line(x0, y0, x1, y1)
        self.start = canvas.create_image(*self.state_to_cd(START_POINT)[::-1], image=self.start_image)
        self.rectangle = canvas.create_image(*self.state_to_cd(START_POINT)[::-1], image=self.shapes[0])
        self.triangles = [canvas.create_image(*self.state_to_cd(OBS)[::-1], image=self.shapes[1]) for OBS in OBSTACLES]
        self.circle = canvas.create_image(*self.state_to_cd(END_POINT)[::-1], image=self.shapes[2])
        canvas.pack()

        return canvas
    
    def change_timestep(self, value):
        self.time_step = int(value) / 10.
        value = f"Time Step : {self.time_step} 초"
        self.time_label.config(text=value)

    def change_lr(self, value):
        self.learning_rate = int(value) / 10.
        value = f"Learning Rate : {self.learning_rate}"
        self.lr_label.config(text=value)

    def change_gamma(self, value):
        self.gamma = int(value) / 10.
        value = f"Gamma : {self.gamma}"
        self.gamma_label.config(text=value)

    def update_epsilon(self, epsilon):
        self.epsilon = epsilon
        value = f"Epsilon : {self.epsilon : .3f}"
        self.epsilon_label.config(text=value)

    def update_episode(self, episode):
        self.episode = episode
        value = f"Episode : {self.episode} / {self.total_episode}"
        self.episode_label.config(text=value)

    def update_reward(self, total_reward):
        self.total_reward = total_reward
        value = f"남은 체력(Total Reward) : {total_reward}"
        self.reward_label.config(text=value)

    @staticmethod
    def load_images():
        rectangle = PhotoImage(
            Image.open("img/character.png").resize((int(UNIT * 2/3), int(UNIT * 2/3))))
        triangle = PhotoImage(
            Image.open("img/obstacle.png").resize((int(UNIT * 3 / 4), int(UNIT * 3 / 4))))
        circle = PhotoImage(
            Image.open("img/end.png").resize((int(UNIT * 2/3), int(UNIT * 2/3))))
        return rectangle, triangle, circle

    def text_value(self, row, col, contents, action, font='Helvetica', size=10, style='normal', anchor="nw"):
        if action == 0:
            origin_x, origin_y = int(UNIT * 1/8 - 5), int(UNIT * 1/2 - 10)
        elif action == 1:
            origin_x, origin_y = int(UNIT * 6/7 - 5), int(UNIT * 1/2 - 10)
        elif action == 2:
            origin_x, origin_y = int(UNIT * 1/2 - 5), int(UNIT * 1/8 - 10)
        else:
            origin_x, origin_y = int(UNIT * 1/2 - 5), int(UNIT * 4/5 - 5)

        x, y = origin_y + (UNIT * col), origin_x + (UNIT * row)
        font = (font, str(size), style)
        text = self.canvas.create_text(x, y, fill="black", text=contents, font=font, anchor=anchor)
        return self.texts.append(text)

    def print_value_all(self, q_table):
        for i in self.texts:
            self.canvas.delete(i)
        self.texts.clear()
        for i in range(HEIGHT):
            for j in range(WIDTH):
                for action in range(0, 4):
                    state = str([i, j])
                    if state in q_table:
                        temp = q_table[state][action]
                        self.text_value(i, j, round(temp, 2), action)

    @staticmethod
    def cd_to_state(coords):
        x = int((coords[1] - UNIT/2) / UNIT)
        y = int((coords[0] - UNIT/2) / UNIT)
        return [y, x]

    @staticmethod
    def state_to_cd(state):
        x = int(state[1] * UNIT + UNIT/2)
        y = int(state[0] * UNIT + UNIT/2)
        return [y, x]

    def reset(self):
        start_time = time.time()
        while time.time() - start_time < self.time_step:
            self.render()
        x, y = self.canvas.coords(self.rectangle)
        start_coord = self.state_to_cd(START_POINT)
        self.canvas.move(self.rectangle, start_coord[1] - x, start_coord[0] - y)
        start_time = time.time()
        while time.time() - start_time < self.time_step:
            self.render()
        return self.cd_to_state(self.canvas.coords(self.rectangle)[::-1])

    def step(self, action, isend):
        state = self.canvas.coords(self.rectangle)[::-1]
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

        if next_state == self.canvas.coords(self.circle)[::-1]:
            reward = 50
            done = True

        elif next_state in [self.canvas.coords(triangle)[::-1] for triangle in self.triangles]:
            reward = -10
            collided = True

        elif next_state[0] < 0 or next_state[1] < 0 or next_state[0] > UNIT * HEIGHT or next_state[1] > UNIT * WIDTH:
            reward = -5
            collided = True

        else:
            reward = -1
        data = self.action_dict[action] 
        if collided:
            data = "obs_" + data
            next_state = state
        else:
            self.canvas.move(self.rectangle, base_action[1], base_action[0])
            self.canvas.tag_raise(self.rectangle)

        if isend:
            QUEUE1.put(data)
        else:
            QUEUE2.put(data)

        return self.cd_to_state(next_state), reward, done


    def render(self):
        time.sleep(0.03)
        self.update()


class QLearningAgent:
    def __init__(self):
        self.actions = [0, 1, 2, 3]
        self.learning_rate = LR
        self.discount_factor = GAMMA
        self.epsilon = EPSILON
        self.decay = DECAY

        self.q_table = {}
        self.grid = self.make_grid()

        for i in range(HEIGHT):
            for j in range(WIDTH):
                if (i, j) not in OBSTACLES:
                    self.q_table[str([i, j])] = [.0] * len(self.actions)

    @staticmethod
    def make_grid():
        grid = np.full((HEIGHT, WIDTH), ".")
        grid[START_POINT[0], START_POINT[1]] = "S"
        for OBSTACLE in OBSTACLES:
            grid[OBSTACLE[0], OBSTACLE[1]] = "X"
        grid[END_POINT[0], END_POINT[1]] = "E"
        return grid


    def learn(self, state, action, reward, next_state):
        q_1 = self.q_table[state][action]
        q_2 = reward + self.discount_factor * max(self.q_table[next_state])
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
                         "discount_factor": self.discount_factor,
                         "epsilon": self.epsilon,
                         "decay": self.decay}, f)

    def load_data(self):
        with open("data.txt", "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.learning_rate = data["learning_rate"]
        self.discount_factor = data["discount_factor"]
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

    # BFS(너비 우선 탐색법)를 통한 시작점으로의 최단거리 계산
    def bfs(self, start_point):
        start_point = tuple(start_point)
        queue = deque([[start_point]])
        seen = {start_point}
        while queue:
            path = queue.popleft()
            y, x = path[-1]
            if self.grid[y, x] == "S":
                return path
            for x2, y2 in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= x2 < WIDTH and 0 <= y2 < HEIGHT and self.grid[y2, x2] != "X" and (y2, x2) not in seen:
                    queue.append(path + [(y2, x2)])
                    seen.add((y2, x2))

    # 위에서 계산한 최단거리를 이용하여 시작점으로의 최단거리 복귀
    @staticmethod
    def return_home(path, isend):
        if isend:
            QUEUE2.put("episode_ended")
            dir_dict = {"[-1  0]": "up", "[1 0]": "down", "[ 0 -1]": "left", "[0 1]": "right"}
            direction_list = list(map(lambda x: dir_dict[str(np.array(x[0]) - np.array(x[1]))], zip(path[1:], path[:-1])))
            for action in direction_list:
                QUEUE2.put(action)
            # 비프음 출력을 위한 EV3와의 송수신
            QUEUE2.put("return_ended")
            # 비프음 출력을 위한 EV3와의 송수신
        else:
            QUEUE1.put("episode_ended")
            dir_dict = {"[-1  0]": "up", "[1 0]": "down", "[ 0 -1]": "left", "[0 1]": "right"}
            direction_list = list(map(lambda x: dir_dict[str(np.array(x[0]) - np.array(x[1]))], zip(path[1:], path[:-1])))
            for action in direction_list:
                QUEUE1.put(action)
            # 비프음 출력을 위한 EV3와의 송수신
            QUEUE1.put("return_ended")
            # 비프음 출력을 위한 EV3와의 송수신

def main(load_saved_data):
    global ended
    env = Env()
    agent = QLearningAgent()

    socket_thread = threading.Thread(target=handle_socket) 
    socket_thread.daemon = True 
    socket_thread.start()

    if load_saved_data:
        agent.load_data()

    for episode in range(TOTAL_EPISODE):
        env.update_episode(episode + 1)
        print(f"episode : {episode + 1}, epsilon : {agent.epsilon}")
        if episode == TOTAL_EPISODE - 1:
            isend = True
        else:
            isend = False
        state = env.reset()                     
        total_reward = 50

        while True:
            env.render()
            action = agent.get_action(str(state))
            next_state, reward, done = env.step(action, isend)
            total_reward += reward
            env.update_reward(total_reward)
            if total_reward <= 0:
                done = True
            agent.learning_rate = env.learning_rate
            agent.gamma = env.gamma
            env.update_epsilon(agent.epsilon)
            agent.learn(str(state), action, reward, str(next_state))
            state = next_state
            env.print_value_all(agent.q_table)

            if done:
                print(f"Finished!! \ntotal reward : {total_reward}\n\n")
                agent.save_data()
                return_path = agent.bfs(state)
                agent.return_home(return_path, isend)
                break
    ended = True


if __name__ == "__main__":
    main(load_saved_data=False)
    while True:
        time.sleep(1)