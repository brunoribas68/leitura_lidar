import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

def lidar_to_cartesian(ranges, angles, origin):
  """
  Converte dados LiDAR em coordenadas cartesianas.

  Args:
    ranges: Uma matriz de distâncias LiDAR.
    angles: Uma matriz de ângulos LiDAR.
    origin: Uma matriz de coordenadas cartesianas do ponto de origem do LiDAR.

  Returns:
    Uma matriz de coordenadas cartesianas dos pontos LiDAR.
  """

  # Calcula as coordenadas x e y dos pontos LiDAR.

  x = ranges * np.cos(angles)
  y = ranges * np.sin(angles)

  # Adiciona as coordenadas do ponto de origem do LiDAR.

  coordinates = np.column_stack((x, y)) + 0

  return coordinates


# Exemplo de uso.
def create_dataFrame():
    dataFrame = pd.read_csv('dados_lidar.txt', sep=" ", header=None)
    dataFrame.replace([np.inf, -np.inf], np.nan, inplace=True)
    return dataFrame


def fix_dataFrame(dataFrame):
    ranges = dataFrame[6:]
    ranges.columns = ["id", "y"]
    ranges["x"] = np.arange(dataFrame[1][0], dataFrame[1][1], dataFrame[1][2])
    newDataFrame = pd.DataFrame({"ranges": ranges["y"], "angles": ranges["x"]})
    return newDataFrame

def gerar_grafico_sem_axis(coordinates):
    plt.xlim([np.nanmax(coordinates[:, 1]) + 2, min(coordinates[:, 1]) - 2])
    plt.grid(False)
    plt.axis('off')
    # Mostra o gráfico.
    plt.savefig("grafico_original.png", bbox_inches=0)
    plt.show()

def gerar_grafico(coordinates):
    plt.scatter(coordinates[:, 1], coordinates[:, 0], color='red')

    # Define os limites dos eixos x.
    plt.xlim([np.nanmax(coordinates[:, 1]) + 2, min(coordinates[:, 1]) - 2])
    plt.savefig("ex4.png", bbox_inches=0)

    # Mostra o gráfico.
    plt.show()

def identify_corners():
    image = cv2.imread('grafico_original.png')

    # convert the input image into
    # grayscale color space
    operatedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # modify the data type
    # setting to 32-bit floating point
    operatedImage = np.float32(operatedImage)

    # apply the cv2.cornerHarris method
    # to detect the corners with appropriate
    # values as input parameters
    dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)

    # Results are marked through the dilated corners
    dest = cv2.dilate(dest, None)

    # Reverting back to the original image,
    # with optimal threshold value
    image[dest > 0.01 * dest.max()] = [0, 0, 255]

    # the window showing output image with corners
    cv2.imshow('Image with Borders', image)

    # De-allocate any associated memory usage
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

    cv2.imwrite('ex4.jpg', image)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataFrame = create_dataFrame()
    newDataFrame = fix_dataFrame(dataFrame)
    angles = newDataFrame.angles
    ranges = newDataFrame.ranges

    coordinates = lidar_to_cartesian(ranges, angles, 0)
    coordinates[np.where(np.logical_and(coordinates[:, 1] >= 1.74533, coordinates[:, 1] <= 1.74533))]

    print(coordinates[:, 1])
    # Cria o gráfico de dispersão.
    plt.scatter(coordinates[:, 1], coordinates[:, 0])
    #100 = 1,74533 rad
    # 80 = 1,39626
    # Gera a imagem do grafico.
    gerar_grafico_sem_axis(coordinates)

    identify_corners()


