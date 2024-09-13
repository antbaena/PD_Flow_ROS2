import rclpy
from rclpy.node import Node
from pd_flow_msgs.msg import FlowField
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class CV2Visual(Node):
    def __init__(self):
        super().__init__('cv2_visual')
        self.subscription = self.create_subscription(FlowField, 'flow_field', self.listener_callback, 10)
        self.image_shape = (480, 640)  # Ajustar según las dimensiones reales de tu imagen
        self.bridge = CvBridge()
        self.get_logger().info("CV2 Visual inicializado correctamente")

    def listener_callback(self, msg, scale_x = 10, scale_y = 1):
        dx = np.array(msg.dx)
        dy = np.array(msg.dy)
        dz = np.array(msg.dz)
        
        num_elements = self.image_shape[0] * self.image_shape[1]
        num_elements = 409500
        # num_elements = 307200
        # len(dx) = 409500
        # if len(dx) != num_elements or len(dy) != num_elements or len(dz) != num_elements:
        #     self.get_logger().error(f"El tamaño de los datos no coincide con las dimensiones esperadas: {self.image_shape}")
        #     return
        
        magnitudes = np.sqrt(dx**2 + dy**2 + dz**2)
        max_magnitude = np.max(magnitudes)
        normalized_magnitudes = magnitudes / max_magnitude if max_magnitude > 0 else magnitudes

        color_map = np.zeros((len(dx), 3), dtype=np.uint8)
        color_map[:, 0] = (normalized_magnitudes * 255).astype(np.uint8)  # Blue channel
        color_map[:, 2] = (255 - normalized_magnitudes * 255).astype(np.uint8)  # Red channel

        # COmo el image_shape está mal esto de aqui se ve doble
        image = np.zeros((self.image_shape[0], self.image_shape[1], 3), dtype=np.uint8)
        
        
        for i in range(num_elements):
            y = (i // self.image_shape[1])
            x = (i % self.image_shape[1])
            start_point = (x, y)
            end_point = (x + int(scale_x * dx[i]), y + int(scale_y * dy[i]))
            color = (int(color_map[i, 2]), 0, int(color_map[i, 0]))
            cv2.arrowedLine(image, start_point, end_point, color, 1)

       
        
        # Mostrar el Flow Field
        # self.get_logger().info(f"Mostrando flujo con CV2. Suma de los vectores: ({sum(dx)}, {sum(dy)}, {sum(dz)})")
        cv2.imshow('CV2 Flow Field Visualization', image)
        cv2.waitKey(1)
        
    
        

    def convied_image_visualizer(self, msg):
         # Convertir los mensajes a imágenes OpenCV
        rgb_image = self.bridge.imgmsg_to_cv2(msg.image.rgb_image, desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(msg.image.depth_image, desired_encoding='16UC1')

        # Normalizar la imagen de profundidad a un rango de 0 a 255 y convertir a tipo np.uint8
        depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_image_uint8 = np.uint8(depth_image_normalized)

        # Convertir la imagen de profundidad a un formato RGB
        depth_image_rgb = cv2.cvtColor(depth_image_uint8, cv2.COLOR_GRAY2BGR)

        # Combinar las imágenes RGB y Depth
        combined_image = np.hstack((rgb_image, depth_image_rgb))

        # Mostrar la imagen combinada RGBD
        cv2.imshow('RGBD Image', combined_image)

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

def main(args=None):
    rclpy.init(args=args)

    cv2_visual = CV2Visual()
    rclpy.spin(cv2_visual)

    cv2_visual.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
