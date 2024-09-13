
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visor_3d(dx, dy, dz, ancho, largo, width=8, height=6):
    """
    Crea una visualización 3D de vectores desde una cuadrícula base usando flechas.
    
    Parámetros:
        dx (list or array): Componentes X de los vectores.
        dy (list or array): Componentes Y de los vectores.
        dz (list or array): Componentes Z de los vectores.
        ancho (int): Número de columnas en la cuadrícula.
        largo (int): Número de filas en la cuadrícula.
        width (int, optional): Ancho de la figura en pulgadas. Por defecto es 8.
        height (int, optional): Alto de la figura en pulgadas. Por defecto es 6.
    """
    # Verificar que los vectores tengan la longitud correcta
    if len(dx) != ancho * largo or len(dy) != ancho * largo or len(dz) != ancho * largo:
        raise ValueError("Los vectores dx, dy y dz deben tener una longitud igual a ancho * largo.")
    
    # Crear la figura y los ejes 3D
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generar las posiciones base para la cuadrícula
    x0, y0 = np.meshgrid(np.arange(ancho), np.arange(largo))
    
    # Aplanar las matrices de posiciones
    x0 = x0.flatten()
    y0 = y0.flatten()
    z0 = np.zeros_like(x0)  # El plano base es z=0
    
    # Dibujar las flechas desde cada píxel (x0, y0, z0) hacia (x0+dx, y0+dy, z0+dz)
    ax.quiver(x0, y0, z0, dx, dy, dz, color='b', arrow_length_ratio=0.1)
    
    # Configurar las etiquetas de los ejes
    ax.set_xlabel('Eje X')
    ax.set_ylabel('Eje Y')
    ax.set_zlabel('Eje Z')
    
    # Configurar el título del gráfico
    ax.set_title('Visualización 3D de Vectores desde una Cuadrícula (Flechas)')
    
    # Configurar los límites de los ejes
    ax.set_xlim([0, ancho])
    ax.set_ylim([0, largo])
    ax.set_zlim([0, max(z0 + dz)])  # El plano base es z=0
    
    # Mostrar la cuadrícula
    ax.grid(True)
    
    # Mostrar el gráfico
    plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    # Definir las dimensiones de la cuadrícula
    ancho = 5
    largo = 5
    total_puntos = ancho * largo

    # Generar datos de ejemplo
    dx = np.random.rand(total_puntos) - 0.5  # Valores aleatorios para dx
    dy = np.random.rand(total_puntos) - 0.5  # Valores aleatorios para dy
    dz = np.random.rand(total_puntos)  # Valores aleatorios para dz

    # Llamar a la función visor_3d con los datos generados
    visor_3d(dx, dy, dz, ancho, largo, width=10, height=8)
