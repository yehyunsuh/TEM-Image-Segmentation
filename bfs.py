from collections import deque
import numpy as np

def bfs(visited: list, n: int, m: int, grid: list) -> int: 
    # row: 시작 행, col: 시작 열, visited: 빈 이미지, n: 이미지 행, m: 이미지 열, grid: 원본이미지
    # visited -> pore 나타내는 이미지 (앞서 mask 추출할 때 사용) pore O: 255, pore X: 0 
    # 살펴볼 방향들
    dx = [0, 0, 1, -1] # x축 방향
    dy = [1, -1, 0, 0] # y축 방향
    q = deque()
    count = [] # pore 안의 pixel 개수들(cnt) 저장할 list
    # 여기에 전체 좌표 살피는 반복문 + 그게 pore이면 아래 진행
    for row in range(len(grid[0])):
        for col in range(len(grid[1])):
            cnt = 0 # pore 안의 pixel 수 나타내는 변
            if grid[row][col] == 255 and visited[row][col] != 255:
                q.append([row, col]) # 시작 좌표
                visited[row][col] = 255 # 시작 좌표 방문한 것으로 처리
                
                while q:
                    x, y = q.popleft()

                    for i in range(4):
                        # 현좌표에서 상하좌우 하나씩 가봄
                        nx = x + dx[i] 
                        ny = y + dy[i]

                        # 좌표가 이미지 벗어나면 반복문 탈출
                        if not (0 <= nx < n and 0 <= ny < m):
                            continue
                        # pore(255)이고 방문 안 했으면(0) visited 255로 바꿈
                        if grid[nx][ny] == 255 and visited[nx][ny] == 0:
                            visited[nx][ny] = 255
                            cnt += 1
                            q.append([nx, ny])
                count.append(cnt)
    return visited, count


def padding_pore(args, pore_list):
    pore_list.sort(reverse=True)
    n_pore = len(pore_list)
    
    # when the number of pores in predictions exceeds the maximum numbers of pore
    if n_pore > args.max_pore:
        pore_list = pore_list[:args.max_pore]
    
    # when the number of pores in predictions lacks the maximum numbers of pore
    while len(pore_list) < args.max_pore:
        pore_list.append(0)
        
    pore_list = np.array(pore_list)
    
    return n_pore, pore_list


def padding_np(args, np_list):
    np_list.sort(reverse=True)
    n_np = len(np_list)
    
    if n_np <= 2:
        while len(np_list) < 2:
            np_list.append(0)
    else:
        np_list = np_list[:2]
    
    return n_np, np_list