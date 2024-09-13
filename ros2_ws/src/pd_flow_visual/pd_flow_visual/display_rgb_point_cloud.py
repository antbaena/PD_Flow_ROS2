import rclpy
from rclpy.node import Node
from pd_flow_msgs.msg import FlowField
from cv_bridge import CvBridge


class ColorFromSpeed(Node):
    def __init__(self):
        super().__init__('display_color_from_speed')
        self.subscription = self.create_subscription(FlowField, 'flow_field', self.listener_callback, 1000)
        self.bridge = CvBridge()

        def display_rgb_point_cloud(self, image_height, image_width, rgb_image, depth_image, dx, dy, dz, color_map):
            f = 525
            u0 = 319.5
            v0 = 239.5
            scale = 0.02  # ~5m/256

            K = np.array([
                [f, 0, u0],
                [0, f, v0],
                [0, 0, 1]
            ])

            # Crear coordenadas del sensor
            h_sensor = np.zeros((3, image_height * image_width))
            point = 0
            for v in range(rgb_image.shape[0]): # rows
                for u in range(rgb_image.shape[1]): # columns
                    h_sensor[0, point] = (u - u0) / f
                    h_sensor[1, point] = (v - v0) / f
                    h_sensor[2, point] = 1  # Coordenadas homogéneas
                    point += 1

            image_d_fila = np.reshape(depth_image, (1, depth_image.shape[0] * depth_image.shape[1]))
            image_d_fila = image_d_fila * scale
            map_3d = h_sensor * image_d_fila # TENEMOS LOS PUNTOS EN EL SENSOR EN 3D
            
            # Filtrar puntos, seleccionar 1 de cada N puntos
            indices = image_d_fila[0] < 25.0
            x = map_3d[0, indices]
            y = map_3d[1, indices]
            z = map_3d[2, indices]
            rgb_image_filtered = rgb_image.reshape(image_height * image_width, 3)[indices]
            
            # Para visualizar los puntos en 3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            # Normalizar colores
            colors = rgb_image_filtered / 255.0

            # Plot de los puntos 3D con colores
            ax.scatter(x, y, z, c=colors, marker='o')

            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            
            plt.show()







def main(args=None):
    rclpy.init(args=args)

    cv2_visual = ColorFromSpeed()
    rclpy.spin(cv2_visual)

    cv2_visual.destroy_node()
    rclpy.shutdown()



if __name__ == '__main__':
    main()
