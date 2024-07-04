#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/io.h>
#include <pcl/point_types.h>
#include <memory>

class PD_flow : public rclcpp::Node
{
public:
    PointCloudViewer() : Node("point_cloud_viewer")
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/camera/depth/points", 10,
            std::bind(&PointCloudViewer::topic_callback, this, std::placeholders::_1));

    }

private:
    void topic_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "I heard: ");

    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PointCloudViewer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
