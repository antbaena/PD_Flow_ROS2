#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2/LinearMath/Quaternion.h>

class VectorFieldPublisher : public rclcpp::Node
{
public:
    VectorFieldPublisher() : Node("vector_field_publisher")
    {
        publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("vector_field", 10);

        // Timer to publish data periodically
        timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            [this]() { publish_vector_field(); }
        );
    }

private:
    void publish_vector_field()
    {
        visualization_msgs::msg::MarkerArray marker_array;
        int id = 0;

        // Example positions and vectors
        std::vector<std::tuple<float, float, float>> positions = {
            {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0},
            {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}, {2.0, 1.0, 0.0},
            {0.0, 2.0, 0.0}, {1.0, 2.0, 0.0}, {2.0, 2.0, 0.0}
        };

        std::vector<std::tuple<float, float, float>> vectors = {
            {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0},
            {1.0, 1.0, 0.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 1.0},
            {1.0, 1.0, 1.0}, {1.0, -1.0, 0.0}, {-1.0, 1.0, 0.0}
        };

        for (size_t i = 0; i < positions.size(); ++i)
        {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "map";
            marker.header.stamp = this->now();
            marker.ns = "vector_field";
            marker.id = id++;
            marker.type = visualization_msgs::msg::Marker::ARROW;
            marker.action = visualization_msgs::msg::Marker::ADD;
            
            auto [px, py, pz] = positions[i];
            marker.pose.position.x = px;
            marker.pose.position.y = py;
            marker.pose.position.z = pz;

            auto [vx, vy, vz] = vectors[i];

            // Compute quaternion for orientation
            tf2::Quaternion quat;
            quat.setRPY(0.0, 0.0, atan2(vy, vx));
            marker.pose.orientation.x = quat.x();
            marker.pose.orientation.y = quat.y();
            marker.pose.orientation.z = quat.z();
            marker.pose.orientation.w = quat.w();

            // Set the scale of the arrow
            marker.scale.x = sqrt(vx * vx + vy * vy + vz * vz); // Arrow length
            marker.scale.y = 0.1; // Arrow shaft diameter
            marker.scale.z = 0.1; // Arrow head diameter

            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            marker.color.a = 1.0; // Fully opaque

            marker_array.markers.push_back(marker);
        }

        // Publish the marker array
        publisher_->publish(marker_array);
    }

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VectorFieldPublisher>());
    rclcpp::shutdown();
    return 0;
}
