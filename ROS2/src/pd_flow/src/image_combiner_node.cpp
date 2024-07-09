#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <pd_flow_msgs/msg/combined_image.hpp>

class ImageCombiner : public rclcpp::Node
{
public:
  ImageCombiner() : Node("image_combiner")
  {
    rgb_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera/color/image_raw", 1,
        std::bind(&ImageCombiner::rgb_callback, this, std::placeholders::_1));

    depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera/depth/image_raw", 1,
        std::bind(&ImageCombiner::depth_callback, this, std::placeholders::_1));

    combined_pub_ = this->create_publisher<pd_flow_msgs::msg::CombinedImage>("combined_image", 10);
  }

private:
  void rgb_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    rgb_image_ = msg;
    combine_images();
  }

  void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    depth_image_ = msg;
    combine_images();
  }

  void combine_images()
  {
    if (!rgb_image_ || !depth_image_)
    {
      return;
    }

    auto combined_msg = pd_flow_msgs::msg::CombinedImage();
    combined_msg.rgb_image = *rgb_image_;
    combined_msg.depth_image = *depth_image_;
 
    combined_pub_->publish(combined_msg);
    RCLCPP_INFO(this->get_logger(), "Publishing combined Image");

    // Reiniciar las variables a nullptr después de publicar para que no se publique varias veces el mismo par
    rgb_image_ = nullptr;
    depth_image_ = nullptr;
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Publisher<pd_flow_msgs::msg::CombinedImage>::SharedPtr combined_pub_;

  sensor_msgs::msg::Image::SharedPtr rgb_image_;
  sensor_msgs::msg::Image::SharedPtr depth_image_;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImageCombiner>());
  rclcpp::shutdown();
  return 0;
}
