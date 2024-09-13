import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from pd_flow_msgs.msg import FlowField
import cv2
import numpy as np
from cv_bridge import CvBridge


class CombinedImage(Node):
    def __init__(self):
        super().__init__('display_combined_image')
        self.bridge = CvBridge()
        
        # Suscripción al tópico con un tamaño de cola reducido para evitar acumulación de frames
        self.subscription = self.create_subscription(
            FlowField,
            'flow_field',
            self.listener_callback,
            1  # Reducir la cola para evitar desbordamientos
        )

        # Variable para almacenar la última imagen combinada
        self.combined_image = None

        # Timer para controlar la frecuencia de visualización
        self.create_timer(0.3, self.display_image)  # Aproximadamente 30 FPS

    def listener_callback(self, msg):
        # Convertir los mensajes a imágenes OpenCV
        rgb_image = self.bridge.imgmsg_to_cv2(msg.image.rgb_image, desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(msg.image.depth_image, desired_encoding='16UC1')

        # Normalizar la imagen de profundidad a un rango de 0 a 255 y convertir a tipo np.uint8
        depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_image_uint8 = np.uint8(depth_image_normalized)

        # Convertir la imagen de profundidad a un formato RGB
        depth_image_rgb = cv2.cvtColor(depth_image_uint8, cv2.COLOR_GRAY2BGR)

        # Combinar las imágenes RGB y Depth
        self.combined_image = np.hstack((rgb_image, depth_image_rgb))

    def display_image(self):
        # Mostrar la imagen combinada solo si existe
        if self.combined_image is not None:
            cv2.imshow('RGBD Image', self.combined_image)
            cv2.waitKey(1)  # Esto permite que OpenCV procese eventos de ventana correctamente

def main(args=None):
    rclpy.init(args=args)
    combined_image_node = CombinedImage()

    rclpy.spin(combined_image_node)

    # Destruir el nodo de manera segura
    combined_image_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
