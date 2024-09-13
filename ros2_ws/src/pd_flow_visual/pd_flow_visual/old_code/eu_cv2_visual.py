import rclpy
from rclpy.node import Node
from pd_flow_msgs.msg import FlowField
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class CV2Visual(Node):
    def __init__(self):
        super().__init__('cv2_visual')
        self.subscription = self.create_subscription(FlowField, 'flow_field', self.listener_callback, 1000)
        self.bridge = CvBridge()
        self.get_logger().info("CV2 Visual inicializado correctamente")

    def listener_callback(self, msg):
        dx = np.array(msg.dx)
        dy = np.array(msg.dy)
        dz = np.array(msg.dz)

        # Convertir los mensajes a imágenes OpenCV
        rgb_image = self.bridge.imgmsg_to_cv2(msg.image.rgb_image, desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(msg.image.depth_image, desired_encoding='16UC1')

        # Obtener las dimensiones de la imagen
        image_width, image_height = rgb_image.shape[:2]

        # Normalizar la imagen de profundidad a un rango de 0 a 255 y convertir a tipo np.uint8
        depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_image_uint8 = np.uint8(depth_image_normalized)

        # Convertir la imagen de profundidad a un formato RGB
        depth_image_rgb = cv2.cvtColor(depth_image_uint8, cv2.COLOR_GRAY2BGR)

        # Combinar las imágenes RGB y Depth
        combined_image = np.hstack((rgb_image, depth_image_rgb))

        num_elements = image_height * image_width

        if len(dx) != num_elements or len(dy) != num_elements or len(dz) != num_elements:
            self.get_logger().error(f"El tamaño de los datos no coincide con las dimensiones esperadas: {image_height}x{image_width}")
            return


        magnitudes = np.sqrt(dx**2 + dy**2 + dz**2)
        max_magnitude = np.max(magnitudes)
        normalized_magnitudes = magnitudes / max_magnitude if max_magnitude > 0 else magnitudes

        color_map = np.zeros((len(dx), 3), dtype=np.uint8)
        color_map[:, 2] = (normalized_magnitudes * 255).astype(np.uint8)  # Canal azul
        color_map[:, 0] = (255 - normalized_magnitudes * 255).astype(np.uint8)  # Canal rojo
        
        #color_img=normalized_magnitudes.reshape((image_height,image_width,1))
        #color_img=color_map.reshape((image_width,image_height,3))
        #cv2.imshow('Color MAP',color_img)
        #cv2.waitKey(1)

        # Mostrar la imagen combinada RGBD
        #cv2.imshow('RGBD Image', combined_image)

        self.display_color_from_speed(image_height, image_width, rgb_image, depth_image, dx, dy, dz, color_map)
        #self.display_rgb_point_cloud(image_height, image_width, rgb_image, depth_image, dx, dy, dz, color_map)
        #self.display_flow_field(image_height, image_width, rgb_image, depth_image, dx, dy, dz, color_map)
        #self.display_flow_field_old(image_height, image_width, rgb_image, depth_image, dx, dy, dz, color_map)
        #self.display_vectors_with_color_map(image_height, image_width, rgb_image, depth_image, dx, dy, dz, color_map)
        #self.display_vectors_with_pixel_color(image_height, image_width, rgb_image, depth_image, dx, dy, dz, color_map)
        #self.display_vectors_with_color_map(image_height, image_width, rgb_image, depth_image, dx, dy, dz, color_map)


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


    def display_motion_field(self, image_height, image_width, rgb_image, depth_image, dx, dy, dz, color_map):
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
        
        # Filtrar puntos por profundidad
        depth_threshold = 25.0
        valid_points = image_d_fila[0] < depth_threshold
        map_3d = map_3d[:, valid_points]
        dx = dx[valid_points]
        dy = dy[valid_points]
        dz = dz[valid_points]

        # Filtrar puntos, seleccionar 1 de cada N puntos
        N = 10
        indices = np.arange(0, map_3d.shape[1], N)
        x = map_3d[0, indices]
        y = map_3d[1, indices]
        z = map_3d[2, indices]
        dx = dx[indices]
        dy = dy[indices]
        dz = dz[indices]

        # Para visualizar los puntos en 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Colorear todos los puntos de rojo
        colors = 'r'

        # Dibujar puntos originales en rojo
        ax.scatter(x, y, z, c=colors, marker='o')

        # Calcular puntos finales
        final_x = x + dx
        final_y = y + dy
        final_z = z + dz

        # Dibujar líneas de puntos originales a puntos finales en azul oscuro
        for i in range(len(x)):
            ax.plot([x[i], final_x[i]], [y[i], final_y[i]], [z[i], final_z[i]], color='navy')

        # Dibujar puntos finales en azul celeste
        ax.scatter(final_x, final_y, final_z, c='skyblue', marker='o')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        
        # Calcular estadísticas
        dx_stats = {
            'min': np.min(dx),
            'max': np.max(dx),
            'mean': np.mean(dx),
            'median': np.median(dx)
        }
        dy_stats = {
            'min': np.min(dy),
            'max': np.max(dy),
            'mean': np.mean(dy),
            'median': np.median(dy)
        }
        dz_stats = {
            'min': np.min(dz),
            'max': np.max(dz),
            'mean': np.mean(dz),
            'median': np.median(dz)
        }
        depth_stats = {
            'min': np.min(image_d_fila),
            'max': np.max(image_d_fila),
            'mean': np.mean(image_d_fila),
            'median': np.median(image_d_fila)
        }
        
        # Mostrar estadísticas
        print("dx stats:", dx_stats)
        print("dy stats:", dy_stats)
        print("dz stats:", dz_stats)
        print("depth stats:", depth_stats)

        plt.show()

    def display_color_from_speed(self, image_height, image_width, rgb_image, depth_image, dx, dy, dz, color_map):
        image = np.zeros((image_width, image_height, 3), dtype=np.uint8)
        i = 0
        for v in range(image_width):  # rows
            for u in range(image_height):  # columns                
                z = depth_image[v, u] * 0.02  # Aplicar escala
                if z > 0:
                    color = (int(color_map[i, 0]), 0, int(color_map[i, 2]))
                    cv2.circle(image, (u, v), 1, color, -1)  # Dibujar el punto
                i = i + 1
                    

        cv2.imshow('Color from speed', image)
        cv2.waitKey(1)


    def display_flow_field(self, image_height, image_width, rgb_image, depth_image, dx, dy, dz, color_map):
        f = 525
        u0 = 319.5
        v0 = 239.5
        scale = 0.02  # ~5m/256

        # Convertir la imagen RGB a un formato adecuado para OpenCV
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # Crear una imagen en negro para dibujar el campo de flujo
        image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

        for v in range(image_height):  # rows
            for u in range(image_width):  # columns
                i = v * image_width + u
                z = depth_image[v, u] * scale  # Aplicar escala

                # Punto 3D original
                original_point = np.array([
                    (u - u0) * z / f,
                    (v - v0) * z / f,
                    z
                ])

                # Punto 3D actualizado con velocidad
                new_point = original_point + np.array([dx[i], dy[i], dz[i]])

                # Proyección del punto 3D actualizado a la imagen 2D
                if new_point[2] != 0:
                    new_x = int((new_point[0] * f / new_point[2]) + u0)
                    new_y = int((new_point[1] * f / new_point[2]) + v0)

                    # Asegurarse de que el punto esté dentro de los límites de la imagen
                    if 0 <= new_x < image_width and 0 <= new_y < image_height:
                        # Obtener el color del pixel original
                        color = tuple(map(int, rgb_image[v, u]))
                        cv2.circle(image, (new_x, new_y), 1, color, -1)  # Dibujar el punto

        # Mostrar la imagen con OpenCV
        cv2.imshow('CV2 Flow Field', image)
        cv2.waitKey(1)

    def rotate_points(self, image_height, image_width, rgb_image, depth_image, dx, dy, dz, color_map):
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

        h_map3D = np.append(map_3d, np.ones((1, map_3d.shape[1])), axis=0) # En homogeneas


        # Para rotarla:
        angle = np.radians(30)
        R = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])  # Rotación

        T = np.zeros((3, 4))
        T[0:3, 0:3] = R
        T[0:3, 3] = [0, 0, 0.5]  # Translación

        h_new_image = K @ T @ h_map3D # Rotamos



        # Transformar a coordenadas cartesianas
        proj = np.divide(h_new_image[:2, :], h_new_image[2, :], where=h_new_image[2, :] != 0)
        proj[np.isnan(proj)] = 0  # Corregir división por 0
        proj = proj.astype(int)

        # Crear la nueva imagen
        new_image = np.zeros_like(rgb_image)
        new_depth = np.full(depth_image.shape,np.inf)

        image_vector = rgb_image.reshape(rgb_image.shape[0]*rgb_image.shape[1],3) # this is a Nx3 ve

        # Iterar sobre los puntos proyectados y verificar si deben aparecer en la imagen
        for p in range(proj.shape[1]):
            u, v = proj[:, p]
            z = h_map3D[2, p]  # Coordenada z
            # Verificar si el píxel debe aparecer en la imagen
            if (0 <= u < image_width) and (0 <= v < image_height):  # Verificar límites
                if new_depth[v, u] > z:  # Verificar si el píxel es más cercano
                    new_depth[v, u] = z
                    new_image[v,u,:] = image_vector[p,:] # Get the color

        # Mostrar la imagen original y la nueva imagen
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 13))
        ax1.imshow(rgb_image)  
        ax1.set_title("Imagen RGB Original")
        ax2.imshow(new_image)
        ax2.set_title("Nueva Imagen")
        plt.show()
                
    def display_flow_field_old(self, image_height, image_width, rgb_image, depth_image, dx, dy, dz, color_map):
        image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        for v in range(rgb_image.shape[0]): # rows
            for u in range(rgb_image.shape[1]): # columns
                i = v * image_width + u
                z = depth_image[v, u] * 0.02

                # Punto 3D original
                original_point = np.array([u * z, v * z, z])

                # Punto 3D actualizado con velocidad
                new_point = original_point + np.array([dx[i], dy[i], dz[i]])

                # HACE FALTA CALIBRAR PARA HACERLO BIEN
                new_x = int(new_point[0]) if new_point[2] == 0 else int(new_point[0] / new_point[2])
                new_y = int(new_point[1]) if new_point[2] == 0 else int(new_point[1] / new_point[2])

                if 0 <= new_x < image_width and 0 <= new_y < image_height:
                    #color = (int(color_map[i, 2]), 0, int(color_map[i, 0])) # Color del mapa
                    color = tuple(map(int, rgb_image[v, u])) # Color pixel
                    
                    cv2.circle(image, (new_x, new_y), 1, color, -1) # Dibujar el punto
                    #cv2.arrowedLine(image, start_point, end_point, color, 1) # Flecha

        self.get_logger().info(f"Mostrando flujo con CV2. Shape de la imagen ({rgb_image.shape[0]}, {rgb_image.shape[1]})")
        cv2.imshow('CV2 Flow Field Old', image)
        cv2.waitKey(1)


    def display_vectors_with_pixel_color(self, image_height, image_width, rgb_image, depth_image, dx, dy, dz, color_map):
        image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        
        for y in range(image_height):
            for x in range(image_width):
                i = y * image_width + x

                start_point = (x, y)
                end_point = (x + int(dx[i]), y + int(dy[i]))
                
                #color = (int(color_map[i, 2]), 0, int(color_map[i, 0])) # Color flecha
                color = tuple(map(int, rgb_image[y, x])) # Color pixel

                cv2.arrowedLine(image, start_point, end_point, color, 1) # Flecha
                #cv2.circle(image, start_point, 1, color, -1) # Punto

        # Mostrar el Flow Field
        self.get_logger().info(f"Mostrando vectores con el color de la imagen RGB")
        cv2.imshow('CV2 Flow Field Vectors RGB', image)
        cv2.waitKey(1)


    def display_vectors_with_color_map(self, image_height, image_width, rgb_image, depth_image, dx, dy, dz, color_map):
        image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        
        scale = 10
        for y in range(image_height):
            for x in range(image_width):
                i = y * image_width + x
                z = depth_image[y, x] * 0.02  # Aplicar escala

                start_point = (x, y)
                end_point = (x + int(scale * dx[i]), y + int(scale * dy[i]))
                
                if z > 0:
                    color = (int(color_map[i, 2]), 0, int(color_map[i, 0])) # Color flecha
                    #color = tuple(map(int, rgb_image[y, x])) # Color pixel

                    cv2.arrowedLine(image, start_point, end_point, color, 1) # Flecha
                    #cv2.circle(image, start_point, 1, color, -1) # Punto

        # Mostrar el Flow Field
        self.get_logger().info(f"Mostrando vectores con un mapa de color")
        cv2.imshow('CV2 Flow Field Vectors Color Map', image)
        cv2.waitKey(1)
        
def main(args=None):
    rclpy.init(args=args)

    cv2_visual = CV2Visual()
    rclpy.spin(cv2_visual)

    cv2_visual.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
