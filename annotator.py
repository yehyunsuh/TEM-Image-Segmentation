"""
reference: https://gaussian37.github.io/vision-opencv-coordinate_extraction/
"""


import os
from datetime import datetime
import cv2
import argparse


dir_del = None
clicked_points = []
clone = None


def MouseLeftClick(event, x, y, flags, param):
    # 왼쪽 마우스가 클릭되면 (x, y) 좌표를 저장한다.
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((y, x))

        # 원본 파일을 가져 와서 clicked_points에 있는 점들을 그린다.
        image = clone.copy()
        for point in clicked_points:
            cv2.circle(image, (point[1], point[0]),
                       5, (0, 255, 255), thickness=-1)
        cv2.imshow("image", image)


def GetArgument():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default='./todo_re_img',
                    help="Enter the image files path")
    args = ap.parse_args(args=[])
    path = args.path
    return path


def main():
    global clone, clicked_points

    # 이미지 디렉토리 경로를 입력 받는다.
    path = GetArgument()
    # path의 이미지명을 받는다.
    image_names = sorted(os.listdir(path))

    # path를 구분하는 delimiter를 구한다.
    if len(path.split('\\')) > 1:
        dir_del = '\\'
    else:
        dir_del = '/'

    # path에 입력된 마지막 폴더 명을 구한다.
    folder_name = path.split(dir_del)[-1]

    # 결과 파일을 저장하기 위하여 현재 시각을 입력 받는다.
    now = datetime.now()
    # now_str = "%s%02d%02d_%02d%02d%02d" % (
    #     now.year - 2000, now.month, now.day, now.hour, now.minute, now.second)

    # 새 윈도우 창을 만들고 그 윈도우 창에 click_and_crop 함수를 세팅해 줍니다.
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", MouseLeftClick)

    for idx, image_name in enumerate(image_names):
        print(image_name)
        image_path = path + dir_del + image_name
        image = cv2.imread(image_path)
        clone = image.copy()
        flag = False

        while True:
            cv2.imshow("image", image)
            key = cv2.waitKey(0)

            if key == ord('n'):
                # 텍스트 파일을 출력 하기 위한 stream을 open 합니다.
                # 중간에 프로그램이 꺼졌을 경우 작업한 것을 저장하기 위해 쓸 때 마다 파일을 연다.
                # file_write = open('./' + now_str + '_' +
                #                   folder_name + '.txt', 'a+')
                file_write = open(
                    './coord.txt', 'a+')

                text_output = image_name
                text_output += "," + str(len(clicked_points))
                for points in clicked_points:
                    text_output += "," + str(points[0]) + "," + str(points[1])
                text_output += '\n'
                file_write.write(text_output)

                # 클릭한 점 초기화
                clicked_points = []

                # 파일 쓰기를 종료한다.
                file_write.close()

                break

            if key == ord('b'):
                if len(clicked_points) > 0:
                    clicked_points.pop()
                    image = clone.copy()
                    for point in clicked_points:
                        cv2.circle(
                            image, (point[1], point[0]), 5, (0, 255, 255), thickness=-1)
                    cv2.imshow("image", image)

            if key == ord('q'):
                # 프로그램 종료
                flag = True
                break

        if flag:
            break

    # 모든 window를 종료합니다.
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()