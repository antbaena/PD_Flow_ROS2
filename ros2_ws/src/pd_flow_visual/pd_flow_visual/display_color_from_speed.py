import rclpy
from rclpy.node import Node
from pd_flow_msgs.msg import FlowField
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge


class ColorFromSpeed(Node):
    def __init__(self):
        super().__init__('display_color_from_speed')
        self.bridge = CvBridge()

        # Reducir el tamaño de la cola a 1 para procesar solo los frames más recientes
        self.subscription = self.create_subscription(
            FlowField,
            'flow_field',
            self.listener_callback,
            1  # Tamaño de la cola reducido para evitar acumulación de mensajes
        )

        # Variables para almacenar la imagen y el color map generados
        self.image_height = None
        self.image_width = None
        self.color_map = None
        self.depth_image = None

        # Timer para controlar la frecuencia de visualización
        self.create_timer(0.3, self.display_image)  # Aproximadamente 30 FPS

    def listener_callback(self, msg):
        # Convertir los datos de velocidad a arrays de NumPy
        dx, dy, dz = np.array(msg.dx), np.array(msg.dy), np.array(msg.dz)

        # Convertir los mensajes a imágenes OpenCV
        rgb_image = self.bridge.imgmsg_to_cv2(msg.image.rgb_image, desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(msg.image.depth_image, desired_encoding='16UC1')

        # Obtener las dimensiones de la imagen
        self.image_height, self.image_width = rgb_image.shape[:2]

        # Validar dimensiones
        if len(dx) != self.image_height * self.image_width:
            self.get_logger().error(f"El tamaño de los datos no coincide con las dimensiones esperadas: {self.image_height}x{self.image_width}")
            return

        # Calcular magnitudes y normalizar
        magnitudes = np.sqrt(dx**2 + dy**2 + dz**2)
        max_magnitude = np.max(magnitudes)
        normalized_magnitudes = magnitudes / max_magnitude if max_magnitude > 0 else magnitudes

        # Crear el mapa de color
        color_map = np.zeros((len(dx), 3), dtype=np.uint8)
        color_map[:, 2] = (normalized_magnitudes * 255).astype(np.uint8)  # Canal azul
        color_map[:, 0] = (255 - normalized_magnitudes * 255).astype(np.uint8)  # Canal rojo

        # Almacenar el mapa de color y la imagen de profundidad para visualización
        self.color_map = color_map
        self.depth_image = depth_image

    def display_image(self):
        # Mostrar la imagen solo si los datos están disponibles
        if self.color_map is not None and self.depth_image is not None:
            # Crear una imagen vacía
            image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

            # Usar indexación avanzada para aplicar el color de manera vectorizada
            z_scaled = (self.depth_image * 0.02).flatten()  # Aplicar escala de profundidad
            valid = z_scaled > 0  # Filtrar valores válidos de profundidad

            # Aplicar colores solo a los píxeles válidos
            positions = np.column_stack(np.where(self.depth_image > 0))
            image[positions[:, 0], positions[:, 1]] = self.color_map[valid]

            # Mostrar la imagen
            cv2.imshow('Color from speed', image)
            cv2.waitKey(1)  # Permitir que OpenCV procese eventos de ventana correctamente

def main(args=None):
    rclpy.init(args=args)
    color_from_speed_node = ColorFromSpeed()
  
    # Ejecutar el nodo hasta que se cierre manualmente
    rclpy.spin(color_from_speed_node)

    # Destruir el nodo de manera segura
    color_from_speed_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
