import cv2
import numpy as np

MaxPoints = 2
nb_points = 0

MapCurveimage512 = None
image = None
mapped_result_img = None
SrcPtInt = [(0, 0) for _ in range(MaxPoints)]


def on_mouse(event, x, y, flags, param):
    global nb_points, MapCurveimage512, mapped_result_img

    if event == cv2.EVENT_LBUTTONDOWN:
        if nb_points < MaxPoints:
            print("test")
            SrcPtInt[nb_points] = (x, y)

            if nb_points:
                # second point in SrcPtInt
                MapCurveimage512[:] = 0

                # read the two extrema points
                x1 = SrcPtInt[0][0]
                x2 = SrcPtInt[1][0]
                y1 = SrcPtInt[0][1]
                y2 = SrcPtInt[1][1]

                # C++
                # double        dx = x1 - x2;	double		  dy = y1 - y2;
                # float x0 = (float)(x1 + x2) / 2;
                # float a = (float)(-2.0*dy / pow(dx, 3.0));
                # float b = (float)(-3.0 / 2.0*dy / dx);
                # float c = (float)((y1 + y2) / 2.0 + b*x0);

                # Python
                dx = x1 - x2
                dy = y1 - y2
                x0 = (x1 + x2) / 2
                a = -2.0 * dy / pow(dx, 3.0)
                b = -3.0 / 2.0 * dy / dx
                # b = 3 * (dy / pow(dx, 3.0)) * (x1 + x2)
                c = (y1 + y2) / 2.0 + b * x0

                # Define the mapping function
                MapCurveimage512 = np.zeros((512, 512), dtype=np.uint8)
                for i in range(512):
                    y = a * (i - x0) * (i - x0) * (i - x0) - b * i + c
                    if y < 0:
                        y = 0
                    if y > 511:
                        y = 511
                    MapCurveimage512[int(y), i] = 255

                # Example line: this should be replaced with actual polynomial and color transform code
                cv2.line(MapCurveimage512, (x1, y1), (x2, y2), (255, 255, 255), 1)

                # Show non-linear mapping curve
                cv2.imshow("GreyCurve", MapCurveimage512)

                # Apply the mapping function to each pixel in the image in a loop
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        mapped_result_img[i, j] = MapCurveimage512[50, 50]  # image[int(MapCurveimage512[i, j]), j] / 2

                # Show the result
                cv2.imshow("result image", mapped_result_img)
                nb_points = 0
            else:
                nb_points += 1


def main():
    global image, mapped_result_img, MapCurveimage512

    filename = "../assets/image.png"
    image = cv2.imread(filename, 1)
    mapped_result_img = image.copy()

    cv2.namedWindow("GreyCurve")
    cv2.namedWindow("Fruits!")
    cv2.imshow("Fruits!", mapped_result_img)

    MapCurveimage512 = np.zeros((512, 512), dtype=np.uint8)
    cv2.imshow("GreyCurve", MapCurveimage512)

    cv2.setMouseCallback("GreyCurve", on_mouse)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
