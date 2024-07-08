#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <pd_flow_msgs/msg/combined_image.hpp>
#include "scene_flow_impair.h"

using namespace std;

class SceneFlowNode : public rclcpp::Node
{
public:
    SceneFlowNode() : Node("scene_flow_node")
    {
        subscription_ = this->create_subscription<pd_flow_msgs::msg::CombinedImage>(
            "combined_image", 10,
            std::bind(&SceneFlowNode::image_callback, this, std::placeholders::_1));
    }

private:
    void image_callback(const pd_flow_msgs::msg::CombinedImage::SharedPtr msg)
    {
        // Buffer the received images
        rgb_image_ = msg->rgb_image;
        depth_image_ = msg->depth_image;

        // Check if both images are available
        if (rgb_image_ && depth_image_)
        {
            // Convert ROS image messages to OpenCV images
            cv_bridge::CvImagePtr cv_ptr_rgb;
            try
            {
                cv_ptr_rgb = cv_bridge::toCvCopy(rgb_image_, sensor_msgs::image_encodings::BGR8);
                cv::Mat intensity_image_1 = cv_ptr_rgb->image;
            }
            catch (cv_bridge::Exception& e)
            {
                RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                return;
            }

            cv_bridge::CvImagePtr cv_ptr_depth;
            try
            {
                cv_ptr_depth = cv_bridge::toCvCopy(depth_image_, depth_image_->encoding);
                cv::Mat depth_image_1 = cv_ptr_depth->image;
            }
            catch (cv_bridge::Exception& e)
            {
                RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                return;
            }

            // Here, simulate a second pair of images. In a real-world scenario, you might receive these as well
            cv::Mat intensity_image_2 = intensity_image_1.clone();  // Example: use the same image
            cv::Mat depth_image_2 = depth_image_1.clone();          // Example: use the same image

            // Save images to temporary files (if necessary)
            cv::imwrite("i1.png", intensity_image_1);
            cv::imwrite("i2.png", intensity_image_2);
            cv::imwrite("z1.png", depth_image_1);
            cv::imwrite("z2.png", depth_image_2);

            // Initialize scene flow algorithm with the received images
            PD_flow_opencv sceneflow(240, "i1.png", "i2.png", "z1.png", "z2.png", "pdflow");

            // Initialize CUDA and set some internal variables
            sceneflow.initializeCUDA();

            if (sceneflow.loadRGBDFrames())
            {
                sceneflow.solveSceneFlowGPU();

                cv::Mat image = sceneflow.createImage();
                sceneflow.saveResults(image);
                sceneflow.showAndSaveResults();

                sceneflow.freeGPUMemory();
            }

            // Clear the buffers
            rgb_image_ = nullptr;
            depth_image_ = nullptr;
        }
    }

    rclcpp::Subscription<pd_flow_msgs::msg::CombinedImage>::SharedPtr subscription_;
    sensor_msgs::msg::Image::SharedPtr rgb_image_;
    sensor_msgs::msg::Image::SharedPtr depth_image_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SceneFlowNode>());
    rclcpp::shutdown();
    return 0;
}
