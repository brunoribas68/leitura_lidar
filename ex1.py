import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    plt.savefig("testplot.png", bbox_inches=0)
    plt.show()

def gerar_grafico(coordinates):
    plt.scatter(coordinates[:, 1], coordinates[:, 0], color='red')

    # Define os limites dos eixos x.
    plt.xlim([np.nanmax(coordinates[:, 1]) + 2, min(coordinates[:, 1]) - 2])

    # Mostra o gráfico.
    plt.show()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataFrame = create_dataFrame()
    newDataFrame = fix_dataFrame(dataFrame)
    angles = newDataFrame.angles
    ranges = newDataFrame.ranges

    coordinates = lidar_to_cartesian(ranges, angles, 0)

    # Cria o gráfico de dispersão.
    plt.scatter(coordinates[:, 1],coordinates[:, 0])

    # Define os limites dos eixos x.
    gerar_grafico(coordinates)


