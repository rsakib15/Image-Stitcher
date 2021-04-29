import cv2
import numpy as np


def save_points(image, x, y, newimageName):
    cv2.imwrite('results/' + newimageName, image)


def get_correspondences(image):
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            a.append(x)
            b.append(y)
            font = cv2.FONT_HERSHEY_SIMPLEX
            strXY = str(x) + ', ' + str(y)
            cv2.putText(image, strXY, (x, y), font, .5, (255, 255, 0), 2)
            cv2.imshow('image', image)

    a = []
    b = []
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return a, b


def compute_homography(src_points,dest_points):
    if src_points.shape != dest_points.shape:
        raise Exception('Source and Destination dimensions must be same')

    n_points = src_points.shape[0]
    coefficient_mat = np.zeros((2 * n_points, 8),dtype=np.float32)
    b = np.zeros((2 * n_points, 1),dtype=np.float32)

    for i in range(n_points):
        coefficient_mat[2 * i] = [src_points[i][0], src_points[i][1], 1, 0, 0, 0, -src_points[i][0] * dest_points[i][0],-src_points[i][1] * dest_points[i][0]]
        coefficient_mat[2 * i + 1] = [0, 0, 0, src_points[i][0], src_points[i][1], 1, -src_points[i][0] * dest_points[i][1], -src_points[i][1] * dest_points[i][1]]
        b[2 * i] = dest_points[i][0]
        b[2*i+1] = dest_points[i][1]

    perspective_mat = np.linalg.lstsq(coefficient_mat, b, rcond=None)[0]
    perspective_mat = np.insert(perspective_mat, 8, values=np.array([1]), axis=0)
    H = perspective_mat.reshape((3, 3))
    return H


def draw_correspondences(image1, image2, H, a, b):
    colors = [(0,0,255), (255,0,0), (0,255,0), (255,255,255)]
    height, width = image1.shape[:2]

    for i in range(len(a)):
        cv2.circle(image1, (a[i], b[i]), radius=3, color=colors[i%4])
        tp = np.mat(H) * np.transpose(np.mat([a[i], b[i], 1]))
        cv2.circle(image2, (int(tp[0, 0] / tp[-1, 0]), int(tp[1, 0] / tp[-1, 0])), radius=3, color=colors[i])

    merged_image = np.hstack([image2, image1])

    for i in range(len(a)):
        tp = np.mat(H) * np.transpose(np.mat([a[i], b[i], 1]))
        cv2.line(merged_image, (a[i]+width, b[i]), (int(tp[0, 0] / tp[-1, 0]), int(tp[1, 0] / tp[-1, 0])), thickness=1, color=colors[i])

    cv2.imshow('Correspondences', merged_image)
    cv2.imwrite('results/additional_correspondences.png', merged_image)
    cv2.waitKey(0)


def warp_and_blend(image1, image2, H):
    p1 = np.dot(H, np.array((0, 0, 1)))
    p2 = np.dot(H, np.array((0, image1.shape[0] - 1, 1)))
    p3 = np.dot(H, np.array((image1.shape[1] - 1, image1.shape[0] - 1, 1)))
    p4 = np.dot(H, np.array((image1.shape[1] - 1, 0, 1)))

    p1 = p1/p1[2]
    p2 = p2/p2[2]
    p3 = p3/p3[2]
    p4 = p4/p4[2]

    minY = int(min(p1[0], p2[0], p3[0], p4[0]))
    minX = int(min(p1[1], p2[1], p3[1], p4[1]))
    maxY = int(max(p1[0], p2[0], p3[0], p4[0]))
    maxX = int(max(p1[1], p2[1], p3[1], p4[1]))

    x, y, z = image1.shape

    blend_image = np.zeros((np.maximum(x, maxX) - np.minimum(0, minX) + 1, np.maximum(y, maxY) - np.minimum(0, minY) + 1, z), dtype=np.float32)
    warp_image = np.zeros((np.maximum(x, maxX) - np.minimum(0, minX) + 1, np.maximum(2 * y, maxY) - np.minimum(0, minY) + 1, z), dtype=np.float32)

    blend_image[np.maximum(0, -minX):np.maximum(0, -minX) + x, :y, :] = image2

    for i in range(x):
        for j in range(y):
            trans = np.mat(H) * np.transpose(np.mat([j, i, 1]))
            rh = int(trans[0, 0] / trans[-1, 0])
            rv = int(trans[1, 0] / trans[-1, 0])
            s = np.minimum(0, minX)

            if rv - s - 1 > 0 and rv - s + 1 < blend_image.shape[0] and rh + 1 < blend_image.shape[1]:
                warp_image[rv - s, rh] = image1[i, j]
                warp_image[rv - s+1, rh] = image1[i, j]
                warp_image[rv - s-1, rh] = image1[i, j]
                warp_image[rv - s, rh + 1] = image1[i, j]
                warp_image[rv - s, rh - 1] = image1[i, j]
                warp_image[rv - s + 1, rh + 1] = image1[i, j]
                warp_image[rv - s - 1, rh - 1] = image1[i, j]
                warp_image[rv - s - 1, rh + 1] = image1[i, j]

                blend_image[rv - s, rh] = image1[i, j]
                blend_image[rv - s + 1, rh] = image1[i, j]
                blend_image[rv - s, rh + 1] = image1[i, j]
                blend_image[rv - s, rh - 1] = image1[i, j]
                blend_image[rv - s + 1, rh + 1] = image1[i, j]
                blend_image[rv - s + 1, rh - 1] = image1[i, j]
                blend_image[rv - s - 1, rh + 1] = image1[i, j]
                blend_image[rv - s - 1, rh - 1] = image1[i, j]

    cv2.imwrite('results/wrap.png', warp_image)
    cv2.imwrite('results/warp_and_blend.png', blend_image)


def generate_billboard(billboard_image, new_billboard_image, content_image):
    a, b = get_correspondences(billboard_image)
    x, y, z = content_image.shape

    c1 = np.array((0, 0, 1))
    c2 = np.array((0, content_image.shape[0] - 1, 1))
    c3 = np.array((content_image.shape[1] - 1, content_image.shape[0] - 1, 1))
    c4 = np.array((content_image.shape[1] - 1, 0, 1))

    src_points = np.array([[c1[0], c1[1]], [c2[0], c2[1]],[c3[0], c3[1]], [c4[0], c4[1]]])
    dest_points = np.array([[a[0], b[0]], [a[1], b[1]], [a[2], b[2]], [a[3], b[3]]])

    H = compute_homography(src_points, dest_points)

    for i in range(x):
        for j in range(y):
            trans = np.mat(H) * np.transpose(np.mat([j, i, 1]))
            rh = int(trans[0, 0] / trans[-1, 0])
            rv = int(trans[1, 0] / trans[-1, 0])
            new_billboard_image[rv, rh] = content_image[i, j, :]

    cv2.imwrite('results/billboard_overwrite.png', new_billboard_image)


if __name__ == '__main__':
    image1 = cv2.imread('data/uttower1.JPG')
    image2 = cv2.imread('data/uttower2.JPG')

    #getting correspondences
    x1, y1 = get_correspondences(image1)
    save_points(image1, x1, y1, 'uttower1_points.jpg')
    x2, y2 = get_correspondences(image2)
    save_points(image2, x1, y1, 'uttower2_points.jpg')

    # computing homography parameter
    if len(x1) < 4 or len(y1) < 4 or len(x2) < 4 or len(y2) < 4:
        raise Exception('At least 4 set of points is required to compute homography')

    src_points = np.array([[x1[0], y1[0]], [x1[1], y1[1]], [x1[2], y1[2]], [x1[3], y1[3]]],dtype=np.float32)
    dest_points = np.array([[x2[0], y2[0]], [x2[1], y2[1]], [x2[2], y2[2]], [x2[3], y2[3]]],dtype=np.float32)

    H = compute_homography(src_points,dest_points)
    print(H)
    draw_correspondences(image1, image2, H, x1, y1)

    #Wraping between image planes
    warp_and_blend(image1,image2,H)

    #billboard_image_overwrite
    print("Generating Billboard image....")
    billboard_image = cv2.imread('data/billboard.jpg')
    new_billboard_image = cv2.imread('data/billboard.jpg')
    content_image = cv2.imread('data/ads.jpg')
    generate_billboard(billboard_image,new_billboard_image,content_image)
    print("Billboard overwrite completed")