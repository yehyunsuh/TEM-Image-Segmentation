from collections import deque
import numpy as np

def bfs(visited: list, n: int, m: int, grid: list): 
    # visited: empty image, n: row of original image, m: column of original image, grid: original image
    
    dx = [0, 0, 1, -1] 
    dy = [1, -1, 0, 0] 
    q = deque()
    count = [] # list that contatins the number of pixels in each pore.
    
    for row in range(len(grid[0])):
        for col in range(len(grid[1])):
            cnt = 0 # saves the number of pixels in a pore.
            if grid[row][col] == 255 and visited[row][col] != 255:
                q.append([row, col]) # starting coordinate 
                visited[row][col] = 255 
                
                while q:
                    x, y = q.popleft()

                    for i in range(4):
                        nx = x + dx[i] 
                        ny = y + dy[i]

                        if not (0 <= nx < n and 0 <= ny < m):
                            continue
                            
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