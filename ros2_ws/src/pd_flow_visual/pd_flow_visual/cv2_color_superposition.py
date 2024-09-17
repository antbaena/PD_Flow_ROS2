import rclpy
from rclpy.node import Node
from pd_flow_msgs.msg import FlowField
import cv2
import numpy as np
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ColorSuperposition(Node):
    def __init__(self):
        super().__init__('display_color_from_speed')
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(FlowField, 'flow_field', self.listener_callback, 1)

        self.image_height = None
        self.image_width = None
        self.color_map = None
        self.depth_image = None
        self.f = 525.2
        self.u0 = 339.65
        self.v0 = 241.54
        self.scale = 0.02  # ~5m/256

        self.K = np.array([
            [self.f, 0, self.u0],
            [0, self.f, self.v0],
            [0, 0, 1]
        ])

        self.create_timer(0.01, self.display_image)  # Aproximadamente 30 FPS

    def listener_callback(self, msg):
        # Convertir los datos de velocidad a arrays de NumPy
        dx, dy, dz = np.array(msg.dx), np.array(msg.dy), np.array(msg.dz)

        # Convertir los mensajes a imágenes OpenCV
        rgb_image = self.bridge.imgmsg_to_cv2(msg.image.rgb_image, desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(msg.image.depth_image, desired_encoding='16UC1')
        flatten_depth_image = depth_image.flatten()

        max_depth = 2500  # Por ejemplo, 1500 mm (1.5 metros)

        # Crear una máscara donde los píxeles estén por debajo o igual a la profundidad máxima
        mask = cv2.inRange(depth_image, 0, max_depth)

        # Convertimos la máscara de 1 canal a 3 para poder aplicarla a la imagen RGB
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Aplicar la máscara a la imagen RGB
        rgb_image = cv2.bitwise_and(rgb_image, mask_rgb)

        # Pasar a escala de grises
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        gray_image[depth_image <= 0] = 0
        
        # Obtener las dimensiones de la imagen
        self.image_height, self.image_width = gray_image.shape[:2]

        # Validar dimensiones
        if len(dx) != self.image_height * self.image_width:
            self.get_logger().error(f"El tamaño de los datos no coincide con las dimensiones esperadas: {self.image_height}x{self.image_width}")
            return

        # Calcular magnitudes y normalizar
        magnitudes = np.sqrt(dx**2 + dy**2 + dz**2)
        magnitudes[flatten_depth_image >= 2500] = 0

        max_magnitude = np.max(magnitudes)
        normalized_magnitudes = magnitudes / max_magnitude if max_magnitude > 0 else magnitudes
        

        # Crear el mapa de color con gradiente blanco-rojo
        color_map = np.zeros((len(dx), 3), dtype=np.uint8)
        # Definir el umbral para transicionar de negro a azul
        threshold = 0.2 # Ajusta este valor según sea necesario

        # Calcular la parte azul (gradiente de negro a azul)
        low_magnitude_mask = normalized_magnitudes < threshold
        color_map[low_magnitude_mask, 0] = (normalized_magnitudes[low_magnitude_mask] / threshold * 255).astype(np.uint8)

        # Calcular la parte roja (gradiente de azul a rojo)
        high_magnitude_mask = normalized_magnitudes >= threshold
        normalized_high_magnitudes = (normalized_magnitudes[high_magnitude_mask] - threshold) / (1 - threshold)
        color_map[high_magnitude_mask, 2] = (normalized_high_magnitudes * 255).astype(np.uint8)
        color_map[high_magnitude_mask, 0] = (255 - color_map[high_magnitude_mask, 2]).astype(np.uint8)


        # Almacenar el mapa de color y la imagen de profundidad para visualización
        self.color_map = color_map
        self.depth_image = depth_image
        self.gray_image = gray_image

    def display_image(self):
        # Mostrar la imagen solo si los datos están disponibles
        if self.color_map is not None and self.depth_image is not None:
            # Crear una imagen vacía
            image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

            # Usar indexación avanzada para aplicar el color de manera vectorizada
            z_scaled = (self.depth_image * 0.02).flatten()  # Aplicar escala de profundidad
            valid = self.depth_image.flatten() > 0  # Filtrar valores válidos de profundidad

            # Aplicar colores solo a los píxeles válidos
            positions = np.column_stack(np.where(self.depth_image > 0))
            color_map_valid = self.color_map[valid]

            # Combinar con la imagen en escala de grises
            combined_image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)  # Convertir escala de grises a BGR

            # Donde el color es completamente blanco (255,255,255), mantener la imagen gris
            mask_non_white = np.any(color_map_valid != [255, 255, 255], axis=1)

            # Aplicar un suavizado entre la imagen gris y el color rojo en las áreas correspondientes
            alpha = 0.7  # Factor de mezcla (0: solo gris, 1: solo color)
            color_blended = (alpha * combined_image[positions[:, 0], positions[:, 1]] + (1 - alpha) * color_map_valid).astype(np.uint8)

            # Actualizar solo las posiciones donde no hay blanco puro
            combined_image[positions[mask_non_white, 0], positions[mask_non_white, 1]] = color_blended[mask_non_white]

            # Mostrar la imagen combinada
            cv2.imshow('Color Superposition', combined_image)
            cv2.waitKey(1)  # Permitir que OpenCV procese eventos de ventana correctamente

    def depth_to_3d(self, depth_threshold=2500):
        """
        Convierte una imagen de profundidad en una nube de puntos 3D, 
        eliminando aquellos puntos donde la profundidad sea mayor a un umbral.
        
        :param depth_threshold: Valor umbral de profundidad. Los puntos con profundidad mayor a este valor serán eliminados.
        """
        if self.color_map is not None and self.depth_image is not None:
            # Obtener las dimensiones de la imagen de profundidad
            height, width = self.depth_image.shape
            
            # Crear una rejilla de coordenadas u, v
            u, v = np.meshgrid(np.arange(width), np.arange(height))
            
            # Extraer los parámetros de la matriz intrínseca
            fx, fy = self.K[0, 0], self.K[1, 1]  # Distancias focales
            cx, cy = self.K[0, 2], self.K[1, 2]  # Punto principal
            
            # Proyección inversa: calcular X, Y, Z
            Z = self.depth_image  # La profundidad es el valor Z
            
            # Aplicar el filtro de umbral de profundidad, si se especifica
            if depth_threshold is not None:
                mask = Z <= depth_threshold  # Máscara para valores Z que estén por debajo del umbral
            else:
                mask = np.ones_like(Z, dtype=bool)  # Si no hay umbral, todos los puntos son válidos
            
            # Aplicar la máscara a las coordenadas
            Z = Z[mask]
            u = u[mask]
            v = v[mask]
            
            # Calcular las coordenadas X e Y en base a la profundidad filtrada
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            
            # Apilar las coordenadas para obtener la nube de puntos 3D
            points_3d = np.vstack((X, Y, Z)).T
            
            # Crear la visualización de la nube de puntos
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            # Crear la nube de puntos
            ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=points_3d[:, 2], cmap='jet', marker='.')
            
            # Etiquetas de los ejes
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            plt.show()

def main(args=None):
    rclpy.init(args=args)
    color_superposition = ColorSuperposition()
  
    # Ejecutar el nodo hasta que se cierre manualmente
    rclpy.spin(color_superposition)

    # Destruir el nodo de manera segura
    color_superposition.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
