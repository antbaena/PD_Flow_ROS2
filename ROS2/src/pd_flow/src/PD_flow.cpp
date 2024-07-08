#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <memory>
#include "pdflow_cudalib.h"
#include <Eigen/Core>
#include <pd_flow_msgs/msg/combined_image.hpp>
#include <cv_bridge/cv_bridge.h>

using namespace std;
using Eigen::MatrixXf;
CSF_cuda csf_host, *csf_device;
MatrixXf rgb_matrix;
MatrixXf depth_matrix;
unsigned int cam_mode;
unsigned int rows;
unsigned int cols;
unsigned int ctf_levels;
unsigned int pyr_levels;

class PD_flow : public rclcpp::Node
{
public:
    PD_flow() : Node("PD_flow")
    {
        subscription_ = this->create_subscription<pd_flow_msgs::msg::CombinedImage>("combined_image", 10, std::bind(&PD_flow::topic_callback, this, std::placeholders::_1));
    }

private:
    void topic_callback(const pd_flow_msgs::msg::CombinedImage msg)
    {
        // Convertir imágenes ROS a OpenCV
        cv_bridge::CvImagePtr rgb_cv_ptr;
        cv_bridge::CvImagePtr depth_cv_ptr;

        try
        {
            rgb_cv_ptr = cv_bridge::toCvCopy(msg.rgb_image, sensor_msgs::image_encodings::BGR8);
            // depth_cv_ptr = cv_bridge::toCvCopy(msg.depth_image, sensor_msgs::image_encodings::TYPE_32FC1);
        }
        catch (cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // cvToEigen(rgb_cv_ptr->image, rgb_matrix);
        // cvToEigen(depth_cv_ptr->image, depth_matrix);

        // Aquí puedes trabajar con las matrices Eigen
        // Por ejemplo, imprimir sus dimensiones
    }

    rclcpp::Subscription<pd_flow_msgs::msg::CombinedImage>::SharedPtr subscription_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PD_flow>();

    while (rclcpp::ok())
    {
        rclcpp::spin(node);
        // imagenRGB, imnagenDepth = sceneflow.CaptureFrame();
        // sceneflow.createImagePyramidGPU(rgb, depth);
        // sceneflow.solveSceneFlowGPU();
    }
    rclcpp::shutdown();
    return 0;
}

// void createImagePyramidGPU(MatrixXf colour_wf, MatrixXf depth_wf)
// {
//     // Copy new frames to the scene flow object
//     csf_host.copyNewFrames(colour_wf.data(), depth_wf.data());

//     // Copy scene flow object to device
//     csf_device = ObjectToDevice(&csf_host);
//     // Cam mode es sobre la resolucion de la camara
//     unsigned int cam_mode = 1;
//     unsigned int rows = colour_wf.rows();
//     unsigned int cols = colour_wf.cols();
//     unsigned int ctf_levels = round(log2(rows / 15)) + 1;
//     unsigned int pyr_levels = round(log2(640 / (cam_mode * cols))) + ctf_levels;

//     GaussianPyramidBridge(csf_device, pyr_levels, cam_mode);

//     // Copy scene flow object back to host
//     BridgeBack(&csf_host, csf_device);
// }

// void solveSceneFlowGPU()
// {
//     // Define variables

//     unsigned int s;
//     unsigned int cols_i, rows_i;
//     unsigned int level_image;
//     unsigned int num_iter;

//     // For every level (coarse-to-fine)
//     for (unsigned int i = 0; i < ctf_levels; i++)
//     {
//         const unsigned int width = colour_wf.cols();
//         s = pow(2.f, int(ctf_levels - (i + 1)));
//         cols_i = cols / s;
//         rows_i = rows / s;
//         level_image = ctf_levels - i + round(log2(width / cols)) - 1;

//         //=========================================================================
//         //                              Cuda - Begin
//         //=========================================================================

//         // Cuda allocate memory
//         csf_host.allocateMemoryNewLevel(rows_i, cols_i, i, level_image);

//         // Cuda copy object to device
//         csf_device = ObjectToDevice(&csf_host);

//         // Assign zeros to the corresponding variables
//         AssignZerosBridge(csf_device);

//         // Upsample previous solution
//         if (i > 0)
//             UpsampleBridge(csf_device);

//         // Compute connectivity (Rij)
//         RijBridge(csf_device);

//         // Compute colour and depth derivatives
//         ImageGradientsBridge(csf_device);
//         WarpingBridge(csf_device);

//         // Compute mu_uv and step sizes for the primal-dual algorithm
//         MuAndStepSizesBridge(csf_device);

//         // Primal-Dual solver
//         for (num_iter = 0; num_iter < num_max_iter[i]; num_iter++)
//         {
//             GradientBridge(csf_device);
//             DualVariablesBridge(csf_device);
//             DivergenceBridge(csf_device);
//             PrimalVariablesBridge(csf_device);
//         }

//         // Filter solution
//         FilterBridge(csf_device);

//         // Compute the motion field
//         MotionFieldBridge(csf_device);

//         // BridgeBack
//         BridgeBack(&csf_host, csf_device);

//         // Free variables of variables associated to this level
//         csf_host.freeLevelVariables();

//         // Copy motion field and images to CPU
//         csf_host.copyAllSolutions(dx[ctf_levels - i - 1].data(), dy[ctf_levels - i - 1].data(), dz[ctf_levels - i - 1].data(),
//                                   depth[level_image].data(), depth_old[level_image].data(), colour[level_image].data(), colour_old[level_image].data(),
//                                   xx[level_image].data(), xx_old[level_image].data(), yy[level_image].data(), yy_old[level_image].data());

//         //=========================================================================
//         //                              Cuda - end
//         //=========================================================================
//     }
// }
