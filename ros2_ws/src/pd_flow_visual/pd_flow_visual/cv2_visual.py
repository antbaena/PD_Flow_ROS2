import rclpy
from rclpy.node import Node
from pd_flow_msgs.msg import FlowField
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Visor3D:
    def __init__(self, ancho, largo, width=8, height=6):
        """
        Inicializa el visor 3D con una cuadrícula base.
        
        Parámetros:
            ancho (int): Número de columnas en la cuadrícula.
            largo (int): Número de filas en la cuadrícula.
            width (int, optional): Ancho de la figura en pulgadas. Por defecto es 8.
            height (int, optional): Alto de la figura en pulgadas. Por defecto es 6.
        """
        self.ancho = ancho
        self.largo = largo

        # Crear la figura y los ejes 3D
        self.fig = plt.figure(figsize=(width, height))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Generar las posiciones base para la cuadrícula
        x0, y0 = np.meshgrid(np.arange(ancho), np.arange(largo))
        self.x0 = x0.flatten()
        self.y0 = y0.flatten()
        self.z0 = np.zeros_like(self.x0)  # El plano base es z=0

        # Configurar los límites de los ejes
        self.ax.set_xlim([0, ancho])
        self.ax.set_ylim([0, largo])
        self.ax.set_zlim([0, 1])  # Se actualizará dinámicamente según los datos

        # Configurar las etiquetas de los ejes
        self.ax.set_xlabel('Eje X')
        self.ax.set_ylabel('Eje Y')
        self.ax.set_zlabel('Eje Z')

        # Configurar el título del gráfico
        self.ax.set_title('Visualización 3D de Vectores desde una Cuadrícula (Flechas)')
        
        # Mostrar la cuadrícula
        self.ax.grid(True)

        # Inicializar el contenedor de flechas
        self.quiver = None

    def actualizar(self, dx, dy, dz):
        """
        Actualiza la visualización 3D con nuevos vectores.
        
        Parámetros:
            dx (list or array): Componentes X de los vectores.
            dy (list or array): Componentes Y de los vectores.
            dz (list or array): Componentes Z de los vectores.
        """
        # Verificar que los vectores tengan la longitud correcta
        if len(dx) != self.ancho * self.largo or len(dy) != self.ancho * self.largo or len(dz) != self.ancho * self.largo:
            raise ValueError("Los vectores dx, dy y dz deben tener una longitud igual a ancho * largo.")
        
        # Si ya existen flechas dibujadas, eliminarlas antes de actualizar
        if self.quiver:
            self.quiver.remove()
        
        # Dibujar nuevas flechas
        self.quiver = self.ax.quiver(self.x0, self.y0, self.z0, dx, dy, dz, color='b', arrow_length_ratio=0.1)

        # Actualizar el límite del eje Z si es necesario
        self.ax.set_zlim([0, max(self.z0 + dz)])

        # Redibujar la figura
        plt.draw()
        plt.pause(0.001)

class CV2Visual(Node):
    def __init__(self):
        super().__init__('cv2_visual')
        self.subscription = self.create_subscription(FlowField, 'flow_field', self.listener_callback, 10)
        self.image_shape = (480, 640)  # Ajustar según las dimensiones reales de tu imagen
        self.bridge = CvBridge()
        self.visor = Visor3D(self.image_shape[1], self.image_shape[0], width=10, height=8)
        self.get_logger().info("CV2 Visual inicializado correctamente")

    def listener_callback(self, msg):
        dx = np.array(msg.dx)
        dy = np.array(msg.dy)
        dz = np.array(msg.dz)
        self.visor.actualizar(dx, dy, dz)

def main(args=None):
    rclpy.init(args=args)

    cv2_visual = CV2Visual()
    rclpy.spin(cv2_visual)

    cv2_visual.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()