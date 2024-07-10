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

        // Initialize OpenCV windows for displaying images
        cv::namedWindow("Intensity Image 1", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Depth Image 1", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Intensity Image 2", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Depth Image 2", cv::WINDOW_AUTOSIZE);
    }

    ~SceneFlowNode()
    {
        // Close OpenCV windows
        cv::destroyAllWindows();
    }

private:
    void image_callback(const pd_flow_msgs::msg::CombinedImage::SharedPtr msg)
    {
        // Buffer the received images
        rgb_image_ = std::make_shared<sensor_msgs::msg::Image>(msg->rgb_image);
        depth_image_ = std::make_shared<sensor_msgs::msg::Image>(msg->depth_image);

        // Check if both images are available
        if (rgb_image_ && depth_image_)
        {
            // Convert ROS image messages to OpenCV images
            cv_bridge::CvImagePtr cv_ptr_rgb;
            cv::Mat intensity_image_1, depth_image_1;
            try
            {
                cv_ptr_rgb = cv_bridge::toCvCopy(rgb_image_, sensor_msgs::image_encodings::BGR8);
                intensity_image_1 = cv_ptr_rgb->image;

                // Corrected declaration and assignment of cv_ptr_depth
                cv_bridge::CvImagePtr cv_ptr_depth;
                cv_ptr_depth = cv_bridge::toCvCopy(depth_image_, depth_image_->encoding);
                depth_image_1 = cv_ptr_depth->image;

                // Verificar si los datos de la imagen de profundidad no son todos ceros
                double min_val, max_val;
                cv::minMaxIdx(depth_image_1, &min_val, &max_val);

                // Escalar los valores de la imagen de profundidad para visualización
                cv::Mat depth_image_1_scaled;
                depth_image_1.convertTo(depth_image_1_scaled, CV_8UC1, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val));
                depth_image_1 = depth_image_1_scaled;
            }
            catch (cv_bridge::Exception& e)
            {
                RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                return;
            }

            // Display intensity_image_1 and depth_image_1
            cv::imshow("Intensity Image 1", intensity_image_1);
            cv::imshow("Depth Image 1", depth_image_1);
            cv::waitKey(1); // Wait for a key press to update windows

            // Simulate a second pair of images by cloning the first pair
            cv::Mat intensity_image_2 = intensity_image_1.clone();
            cv::Mat depth_image_2 = depth_image_1.clone();

            // Display intensity_image_2 and depth_image_2
            cv::imshow("Intensity Image 2", intensity_image_2);
            cv::imshow("Depth Image 2", depth_image_2);
            cv::waitKey(1); // Wait for a key press to update windows

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
                //sceneflow.saveResults(image);
                sceneflow.showAndSaveResults();

                sceneflow.freeGPUMemory();
            }

            RCLCPP_INFO(this->get_logger(), "Iteración correcta del algoritmo");

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
