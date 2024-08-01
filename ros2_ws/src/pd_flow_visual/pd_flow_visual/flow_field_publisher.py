import rclpy
from rclpy.node import Node
from pd_flow_msgs.msg import FlowField
import numpy as np

class FlowFieldPublisher(Node):
    def __init__(self):
        super().__init__('flow_field_publisher')
        
        self.publisher_ = self.create_publisher(FlowField, 'flow_field', 10)
        self.get_logger().info("Flow Field Publisher inicializado correctamente")
        self.timer = self.create_timer(1, self.vector_publisher)
        self.i = 0

    def vector_publisher(self):
        msg = FlowField()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'test_frame'
        
        rows, cols = 480, 640
        num_vectors = rows * cols

        # Simulate a more realistic distribution of flow vectors
        msg.dx = np.random.uniform(-1, 1, num_vectors).tolist()
        msg.dy = np.random.uniform(-1, 1, num_vectors).tolist()
        msg.dz = np.random.uniform(-1, 1, num_vectors).tolist()
        
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing FlowField message with %d vectors' % num_vectors)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)

    flow_field_publisher = FlowFieldPublisher()
    rclpy.spin(flow_field_publisher)

    flow_field_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
