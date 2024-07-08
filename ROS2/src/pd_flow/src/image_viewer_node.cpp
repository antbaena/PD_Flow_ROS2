#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <pd_flow_msgs/msg/combined_image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class ImageViewerNode : public rclcpp::Node {
public:
    ImageViewerNode() : Node("image_viewer_node") {
        subscription_ = this->create_subscription<pd_flow_msgs::msg::CombinedImage>(
            "combined_image", 10, std::bind(&ImageViewerNode::imageCallback, this, std::placeholders::_1));
    }

private:
    void imageCallback(const pd_flow_msgs::msg::CombinedImage::SharedPtr msg) {
        try {
            // Convertir el mensaje de ROS a imágenes OpenCV
            cv::Mat rgb_image = cv_bridge::toCvCopy(msg->rgb_image, sensor_msgs::image_encodings::RGB8)->image;
            cv::Mat depth_image = cv_bridge::toCvCopy(msg->depth_image, sensor_msgs::image_encodings::TYPE_16UC1)->image;

            // Convertir de RGB a BGR para OpenCV
            cv::Mat bgr_image;
            cv::cvtColor(rgb_image, bgr_image, cv::COLOR_RGB2BGR);

            // Mostrar las imágenes en ventanas
            cv::imshow("RGB Image", bgr_image);
            cv::imshow("Depth Image", depth_image);
            cv::waitKey(1);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    rclcpp::Subscription<pd_flow_msgs::msg::CombinedImage>::SharedPtr subscription_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageViewerNode>());
    rclcpp::shutdown();
    return 0;
}
