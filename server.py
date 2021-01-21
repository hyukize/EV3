import time
import numpy as np
import tkinter as tk
from PIL import Image
from PIL.ImageTk import PhotoImage
import socket
from collections import deque
import pickle

UNIT = 100  # 픽셀 수
HEIGHT = 4  # 그리드월드 세로
WIDTH = 4  # 그리드월드 가로

START_POINT = (0, 0)
END_POINT = (3, 3)
OBSTACLES = [(2, 0), (1, 2), (1, 3), (3, 2)]

#     GRID INFO
#
#      0 1 2 3
#
#  0   S . . .
#  1   . . X X
#  2   X . . .
#  3   . . X E


# 접속할 서버 주소입니다. 여기에서는 루프백(loopback) 인터페이스 주소 즉 localhost 를 사용합니다.
HOST = '0.0.0.0'

# 클라이언트 접속을 대기하는 포트 번호입니다.
PORT = 8000

# 소켓 객체를 생성합니다.
# 주소 체계(address family)로 IPv4, 소켓 타입으로 TCP 사용합니다.
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 포트 사용중이라 연결할 수 없다는
# WinError 10048 에러 해결를 위해 필요합니다.
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# bind 함수는 소켓을 특정 네트워크 인터페이스와 포트 번호에 연결하는데 사용됩니다.
# HOST 는 hostname, ip address, 빈 문자열 ""이 될 수 있습니다.
# 빈 문자열이면 모든 네트워크 인터페이스로부터의 접속을 허용합니다.
# PORT 는 1-65535 사이의 숫자를 사용할 수 있습니다.
server.bind((HOST, PORT))

# 서버가 클라이언트의 접속을 허용하도록 합니다.
server.listen()

# accept 함수에서 대기하다가 클라이언트가 접속하면 새로운 소켓을 리턴합니다.
client, addr = server.accept()

# 접속한 클라이언트의 주소입니다.
print('Connected by', addr)


# 프로그램 화면 출력을 위한 함수 모음
class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('Q Learning')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.texts = []
        self.action_dict = {0: "up", 1: "down", 2: "left", 3: "right"}

    def _build_canvas(self):

        # 캔버스 생성
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)

        # 그리드 생성
        for c in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for r in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
            canvas.create_line(x0, y0, x1, y1)

        # 캔버스에 이미지 추가
        self.rectangle = canvas.create_image(*self.state_to_cd(START_POINT)[::-1], image=self.shapes[0])
        self.triangles = [canvas.create_image(*self.state_to_cd(OBS)[::-1], image=self.shapes[1]) for OBS in OBSTACLES]
        self.circle = canvas.create_image(*self.state_to_cd(END_POINT)[::-1], image=self.shapes[2])
        canvas.pack()

        return canvas

    # 프로그램 내에 사용될 도형 이미지 불러오기
    @staticmethod
    def load_images():
        rectangle = PhotoImage(
            Image.open("img/rectangle.png").resize((65, 65)))
        triangle = PhotoImage(
            Image.open("img/triangle.png").resize((65, 65)))
        circle = PhotoImage(
            Image.open("img/circle.png").resize((65, 65)))
        return rectangle, triangle, circle

    # 각각의 0-value 가 표시될 격자내에서의 상대적 위치
    def text_value(self, row, col, contents, action, font='Helvetica', size=10,
                   style='normal', anchor="nw"):
        if action == 0:
            origin_x, origin_y = 7, 42
        elif action == 1:
            origin_x, origin_y = 85, 42
        elif action == 2:
            origin_x, origin_y = 42, 5
        else:
            origin_x, origin_y = 42, 77

        x, y = origin_y + (UNIT * col), origin_x + (UNIT * row)
        font = (font, str(size), style)
        text = self.canvas.create_text(x, y, fill="black", text=contents,
                                       font=font, anchor=anchor)
        return self.texts.append(text)

    # 프로그램 화면 상에  Q-value 를 표시
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

    # 좌표 변환
    @staticmethod
    def cd_to_state(coords):
        x = int((coords[1] - 50) / 100)
        y = int((coords[0] - 50) / 100)
        return [y, x]

    # 좌표 변환
    @staticmethod
    def state_to_cd(state):
        x = int(state[1] * 100 + 50)
        y = int(state[0] * 100 + 50)
        return [y, x]

    # 프로그램 화면 초기상태로 돌림
    def reset(self):
        self.update()
        time.sleep(0.5)
        x, y = self.canvas.coords(self.rectangle)
        start_coord = self.state_to_cd(START_POINT)
        self.canvas.move(self.rectangle, start_coord[1] - x, start_coord[0] - y)
        self.render()
        return self.cd_to_state(self.canvas.coords(self.rectangle)[::-1])

    # 선택된 action 에 따라 reward 를 계산하고, EV3에 이동 명령 하달
    def step(self, action):
        state = self.canvas.coords(self.rectangle)[::-1]
        base_action = np.array([0, 0])
        self.render()

        # 선택된 액션에 따라 프로그램 상에 표시되는 도형이 움직일 변위(y, x)를 구함
        if action == 0:  # 상
            base_action[0] -= UNIT
        elif action == 1:  # 하
            base_action[0] += UNIT
        elif action == 2:  # 좌
            base_action[1] -= UNIT
        elif action == 3:  # 우
            base_action[1] += UNIT

        # 선택된 action 을 통해 다음 state 를 추측
        next_state = (np.array(state) + base_action).tolist()
        # 보상 함수
        done = False
        collided = False

        # 도착지에 도달한 경우
        if next_state == self.canvas.coords(self.circle)[::-1]:
            reward = 50
            done = True

        # 장애물과 충돌한 경우
        elif next_state in [self.canvas.coords(triangle)[::-1] for triangle in self.triangles]:
            reward = -10
            collided = True

        # 벽과 충돌한 경우
        elif next_state[0] < 0 or next_state[1] < 0 or next_state[0] > UNIT * HEIGHT or next_state[1] > UNIT * WIDTH:
            reward = -5
            collided = True

        # 나머지(일반)
        else:
            reward = -1

        # action 숫자가 아닌 이름으로 불러오기
        data = self.action_dict[action]  # action_dict = {0: "up", 1: "down", 2: "left", 3: "right"}

        # 다음 state 를 계산한 결과 벽 혹은 장애물과 충돌할 경우에는,
        if collided:

            # 장애물 혹은 벽과 충돌한 경우 해당 정보를 EV3 에 넘겨주기 위한 준비
            data = "obs_" + data

            # 충돌한 경우는 위치가 바뀌지 않음 -> 다음 state 와 현재 state 가 같다
            next_state = state
        else:
            # 에이전트 이동
            self.canvas.move(self.rectangle, base_action[1], base_action[0])

            # 에이전트(빨간 네모)를 가장 상위로 배치
            self.canvas.tag_raise(self.rectangle)

        # return self.cd_to_state(next_state), reward, done
        # 최종 결정된 action 을 EV3에 전송 (여기에는 장애물 혹은 벽과의 충돌 여부도 포함되어 있음)
        client.send(data.encode())
        result = client.recv(1024).decode()
        if result == "done":
            return self.cd_to_state(next_state), reward, done
        else:
            raise ValueError

    def render(self):
        time.sleep(0.03)
        self.update()


class QLearningAgent:
    def __init__(self):
        # 행동 = [0, 1, 2, 3] 순서대로 상, 하, 좌, 우
        self.actions = [0, 1, 2, 3]

        # 학습에 쓰이는 PARAMETER
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.5
        self.decay = 0.99

        self.q_table = {}
        self.grid = self.make_grid()

        # Q-table 을 모두 0 으로 초기화 (장애물에는 도달하지 않으므로 장애물에는 Q-value 를 할당하지 않있음)
        for i in range(HEIGHT):
            for j in range(WIDTH):
                if (i, j) not in OBSTACLES:
                    self.q_table[str([i, j])] = [.0] * len(self.actions)

    # 맵 행렬을 생성 (시작점은 S, 장애물은 X, 도착점은 E, 나머지는 .)
    @staticmethod
    def make_grid():
        grid = np.full((HEIGHT, WIDTH), ".")
        grid[START_POINT[0], START_POINT[1]] = "S"
        for OBSTACLE in OBSTACLES:
            grid[OBSTACLE[0], OBSTACLE[1]] = "X"
        grid[END_POINT[0], END_POINT[1]] = "E"
        return grid

    # <state, action, reward, new_state> 샘플로부터 큐함수(Q-Function = Q-table) 업데이트
    def learn(self, state, action, reward, next_state):
        q_1 = self.q_table[state][action]

        # 벨만 최적 방정식을 사용한 큐함수의 업데이트
        q_2 = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (q_2 - q_1)

    # 큐함수에 의거하여 입실론 탐욕 정책에 따라서 행동을 반환
    def get_action(self, state):

        # 랜덤으로 액션을 선택하는 경우에는
        if np.random.rand() < self.epsilon:

            # 무작위 행동 반환
            action = np.random.choice(self.actions)

        # 그렇지 않은 경우는 Q-Value 에 따른 행동 반환
        else:
            state_action = self.q_table[state]
            action = self.arg_max(state_action)

        # decay 에 의한 입실론 감소
        self.epsilon *= self.decay

        # 입실론이 0.1, 즉 10% 미만으로 떨어지면 입실론 탐욕 정책 종료 (epsilon -> 0 으로 설정)
        if self.epsilon < 0.1:
            self.epsilon = 0
        return action

    # 딕셔너리 형태의 데이터를 pickle 라이브러리를 이용해 저장
    def save_data(self):
        with open("data.txt", "wb") as f:
            pickle.dump({"q_table": self.q_table,
                         "learning_rate": self.learning_rate,
                         "discount_factor": self.discount_factor,
                         "epsilon": self.epsilon,
                         "decay": self.decay}, f)

    # 딕셔너리 형태의 데이터를 pickle 라이브러리를 이용해 불러오기
    def load_data(self):
        with open("data.txt", "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.learning_rate = data["learning_rate"]
        self.discount_factor = data["discount_factor"]
        self.epsilon = data["epsilon"]
        self.decay = data["decay"]

    # 액션이 가지고 있는 q-value 중 가장 큰 값을 가지는 액션을 선택하는 함수(가장 큰 값이 여러개일 경우 그 중 랜덤으로 선택)
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
    def return_home(path):
        dir_dict = {"[-1  0]": "up", "[1 0]": "down", "[ 0 -1]": "left", "[0 1]": "right"}
        direction_list = list(map(lambda x: dir_dict[str(np.array(x[0]) - np.array(x[1]))], zip(path[1:], path[:-1])))
        for action in direction_list:
            client.send(action.encode())
            result = client.recv(1024).decode()
            if result != "done":
                raise ValueError


# 메인 함수
def main(load_saved_data):
    env = Env()
    agent = QLearningAgent()

    # 저장된 데이터를 불러올지 선택
    if load_saved_data:
        agent.load_data()

    # 에피소드 루프 시작
    for episode in range(100):
        print(f"episode : {episode + 1}, epsilon : {agent.epsilon}")

        # 출력 화면 초기화
        state = env.reset()

        # 초기 체력 100 으로 시작
        total_reward = 100

        # Time Step 루프 시작
        while True:
            # 프로그램 출력용
            env.render()

            # 현재 상태에 대한 행동 선택
            action = agent.get_action(str(state))

            # 행동을 취한 후 다음 상태, 보상 에피소드의 종료여부를 받아옴 (도착점에 다다를 경우 종료)
            next_state, reward, done = env.step(action)

            # 보상의 총합을 계산 (체력)
            total_reward += reward

            # 플레이어의 체력이 0 이하로 떨어져도 게임 종료
            if total_reward <= 0:
                done = True

            # <s,a,r,s'>로 큐함수를 업데이트
            agent.learn(str(state), action, reward, str(next_state))
            state = next_state

            # 모든 큐함수를 화면에 표시
            env.print_value_all(agent.q_table)

            # 게임 종료 조건을 만족할 경우 비프음을 한번 울리고 최단거리로 출발점으로 복귀
            if done:
                print(f"Finished!! \ntotal reward : {total_reward}\n\n")

                # 데이터 저장
                agent.save_data()

                # 비프음 출력을 위한 EV3와의 송수신
                client.send(b"end")
                data = client.recv(1024).decode()
                if data != "received":
                    raise ValueError

                # BFS(너비 우선 탐색법)을 통한 출발점으로의 최단경로 복귀
                return_path = agent.bfs(state)
                agent.return_home(return_path)

                # 비프음 출력을 위한 EV3와의 송수신
                client.send(b"end")
                data = client.recv(1024).decode()
                if data != "received":
                    raise ValueError

                break


if __name__ == "__main__":
    main(load_saved_data=False)
