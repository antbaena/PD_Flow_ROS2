import cv2
import numpy as np
from queue import Queue

import rclpy
from rclpy.node import Node

from pd_flow_msgs.msg import FlowField

from cv_bridge import CvBridge

class ColorFromSpeedNode(Node):
    def __init__(self):
        super().__init__('cv2_color_from_speed')
        self.queue = Queue()
        self.subscription = self.create_subscription(FlowField, 'flow_field', self.listener_callback, 100)

    def listener_callback(self, msg):
        self.queue.put(msg)

class ColorFromSpeed():
    def __init__(self):
        self.node = ColorFromSpeedNode()
        self.bridge = CvBridge()
    
    def spin(self):
        while rclpy.ok():
            if not self.node.queue.empty():
                msg = self.node.queue.get()

                image = self.create_color_from_speed_image(msg)
                
                cv2.imshow('Color from speed', image)
                cv2.waitKey(1)

            rclpy.spin_once(self.node)

    def create_color_from_speed_image(self, msg):
        dx, dy, dz = np.array(msg.dx), np.array(msg.dy), np.array(msg.dz)

        rgb_image = self.bridge.imgmsg_to_cv2(msg.image.rgb_image, desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(msg.image.depth_image, desired_encoding='16UC1')
        flatten_depth_image = depth_image.flatten()

        image_height, image_width = rgb_image.shape[:2]

        if len(dx) != image_height * image_width:
            self.node.get_logger().error(f"El tamaño de los datos no coincide con las dimensiones esperadas: {image_height}x{image_width}")
            return

        magnitudes = np.sqrt(dx**2 + dy**2 + dz**2)
        # Filtramos las profundidades lejanas, pues la camara capta mal los colores lejos
        # y eso causa ruido y sale movimiento donde no lo hay
        magnitudes[flatten_depth_image >= 2500] = 0

        max_magnitude = np.max(magnitudes)
        normalized_magnitudes = magnitudes / max_magnitude if max_magnitude > 0 else magnitudes
        
        color_map = np.zeros((len(dx), 3), dtype=np.uint8)
        color_map[:, 2] = (normalized_magnitudes * 255).astype(np.uint8)  # Canal azul
        color_map[:, 0] = (255 - normalized_magnitudes * 255).astype(np.uint8)  # Canal rojo

        image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

        z_scaled = flatten_depth_image * 0.02
        valid = z_scaled > 0

        positions = np.column_stack(np.where(depth_image > 0))
        image[positions[:, 0], positions[:, 1]] = color_map[valid]
        
        # Descomentar para visualizar solo a las personas
        # positions2 = np.column_stack(np.where(depth_image >= 2500))
        # image[positions2[:, 0], positions2[:, 1]] = [0,0,0]

        return image

def main(args=None):
    rclpy.init(args=args)

    color_from_speed = ColorFromSpeed()
    color_from_speed.spin()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
