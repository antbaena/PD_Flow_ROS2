
/*****************************************************************************
**                Primal-Dual Scene Flow for RGB-D cameras                  **
**                ----------------------------------------                  **
**                                                                          **
**  Copyright(c) 2015, Mariano Jaimez Tarifa, University of Malaga          **
**  Copyright(c) 2015, Mohamed Souiai, Technical University of Munich       **
**  Copyright(c) 2015, MAPIR group, University of Malaga                    **
**  Copyright(c) 2015, Computer Vision group, Tech. University of Munich    **
**                                                                          **
**  This program is free software: you can redistribute it and/or modify    **
**  it under the terms of the GNU General Public License (version 3) as     **
**  published by the Free Software Foundation.                              **
**                                                                          **
**  This program is distributed in the hope that it will be useful, but     **
**  WITHOUT ANY WARRANTY; without even the implied warranty of              **
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the            **
**  GNU General Public License for more details.                            **
**                                                                          **
**  You should have received a copy of the GNU General Public License       **
**  along with this program.  If not, see <http://www.gnu.org/licenses/>.   **
**                                                                          **
*****************************************************************************/

#include "scene_flow_visualization.h"

PD_flow::PD_flow(unsigned int cam_mode_config, unsigned int fps_config, unsigned int rows_config)
{

    rows = rows_config; // Maximum size of the coarse-to-fine scheme - Default 240 (QVGA)
    cols = rows * 320 / 240;
    cam_mode = cam_mode_config; // (1 - 640 x 480, 2 - 320 x 240), Default - 1
    ctf_levels = round(log2(rows / 15)) + 1;
    fovh = M_PI * 62.5f / 180.f;
    fovv = M_PI * 45.f / 180.f;
    fps = fps_config; // In Hz, Default - 30
    cout << "Initializing PD_flow" << endl;
    cout << "Rows: " << rows << endl;
    cout << "Cols: " << cols << endl;
    cout << "Cam mode: " << cam_mode << endl;
    cout << "CTF levels: " << ctf_levels << endl;
    cout << "FPS: " << fps << endl;

    // Iterations of the primal-dual solver at each pyramid level.
    // Maximum value set to 100 at the finest level
    for (int i = 5; i >= 0; i--)
    {
        if (i >= ctf_levels - 1)
            num_max_iter[i] = 100;
        else
            num_max_iter[i] = num_max_iter[i + 1] - 15;
    }

    // num_max_iter[ctf_levels-1] = 0.f;

    // Compute gaussian mask
    float v_mask[5] = {1.f, 4.f, 6.f, 4.f, 1.f};
    for (unsigned int i = 0; i < 5; i++)
        for (unsigned int j = 0; j < 5; j++)
            g_mask[i + 5 * j] = v_mask[i] * v_mask[j] / 256.f;

    // Matrices that store the original and filtered images with the image resolution
    colour_wf.resize(480 / cam_mode, 640 / cam_mode);
    depth_wf.resize(480 / cam_mode, 640 / cam_mode);

    // Resize vectors according to levels
    dx.resize(ctf_levels);
    dy.resize(ctf_levels);
    dz.resize(ctf_levels);

    const unsigned int width = colour_wf.cols();
    const unsigned int height = colour_wf.rows();
    unsigned int s, cols_i, rows_i;

    for (unsigned int i = 0; i < ctf_levels; i++)
    {
        s = pow(2.f, int(ctf_levels - (i + 1)));
        cols_i = cols / s;
        rows_i = rows / s;
        dx[ctf_levels - i - 1].resize(rows_i, cols_i);
        dy[ctf_levels - i - 1].resize(rows_i, cols_i);
        dz[ctf_levels - i - 1].resize(rows_i, cols_i);
    }

    // Resize pyramid
    const unsigned int pyr_levels = round(log2(width / cols)) + ctf_levels;
    colour.resize(pyr_levels);
    colour_old.resize(pyr_levels);
    depth.resize(pyr_levels);
    depth_old.resize(pyr_levels);
    xx.resize(pyr_levels);
    xx_old.resize(pyr_levels);
    yy.resize(pyr_levels);
    yy_old.resize(pyr_levels);

    for (unsigned int i = 0; i < pyr_levels; i++)
    {
        s = pow(2.f, int(i));
        colour[i].resize(height / s, width / s);
        colour_old[i].resize(height / s, width / s);
        colour[i].setZero();
        colour_old[i].setZero();
        depth[i].resize(height / s, width / s);
        depth_old[i].resize(height / s, width / s);
        depth[i].setZero();
        depth_old[i].setZero();
        xx[i].resize(height / s, width / s);
        xx_old[i].resize(height / s, width / s);
        xx[i].setZero();
        xx_old[i].setZero();
        yy[i].resize(height / s, width / s);
        yy_old[i].resize(height / s, width / s);
        yy[i].setZero();
        yy_old[i].setZero();
    }

    // Parameters of the variational method
    lambda_i = 0.04f;
    lambda_d = 0.35f;
    mu = 75.f;

    //-------------------------------------------------------------------------
    //                          Cuda - Begin - Initialize
    //-------------------------------------------------------------------------

    // Read parameters
    csf_host.readParameters(rows, cols, lambda_i, lambda_d, mu, g_mask, ctf_levels, cam_mode, fovh, fovv);

    // Allocate memory
    csf_host.allocateDevMemory();

    cout << "PD_flow initialized" << endl;
}

void PD_flow::createImagePyramidGPU()
{
    // Copy new frames to the scene flow object
    csf_host.copyNewFrames(colour_wf.data(), depth_wf.data());
    cout << "Creating image pyramid" << endl;
    // Copy scene flow object to device
    csf_device = ObjectToDevice(&csf_host);
    cout << "Object copied to device" << endl;
    unsigned int pyr_levels = round(log2(640 / (cam_mode * cols))) + ctf_levels;
    GaussianPyramidBridge(csf_device, pyr_levels, cam_mode);

    // Copy scene flow object back to host
    BridgeBack(&csf_host, csf_device);
}

void PD_flow::solveSceneFlowGPU()
{
    // Define variables

    unsigned int s;
    unsigned int cols_i, rows_i;
    unsigned int level_image;
    unsigned int num_iter;

    // For every level (coarse-to-fine)
    for (unsigned int i = 0; i < ctf_levels; i++)
    {
        const unsigned int width = colour_wf.cols();
        s = pow(2.f, int(ctf_levels - (i + 1)));
        cols_i = cols / s;
        rows_i = rows / s;
        level_image = ctf_levels - i + round(log2(width / cols)) - 1;

        //=========================================================================
        //                              Cuda - Begin
        //=========================================================================

        // Cuda allocate memory
        csf_host.allocateMemoryNewLevel(rows_i, cols_i, i, level_image);

        // Cuda copy object to device
        csf_device = ObjectToDevice(&csf_host);

        // Assign zeros to the corresponding variables
        AssignZerosBridge(csf_device);

        // Upsample previous solution
        if (i > 0)
            UpsampleBridge(csf_device);

        // Compute connectivity (Rij)
        RijBridge(csf_device);

        // Compute colour and depth derivatives
        ImageGradientsBridge(csf_device);
        WarpingBridge(csf_device);

        // Compute mu_uv and step sizes for the primal-dual algorithm
        MuAndStepSizesBridge(csf_device);

        // Primal-Dual solver
        for (num_iter = 0; num_iter < num_max_iter[i]; num_iter++)
        {
            GradientBridge(csf_device);
            DualVariablesBridge(csf_device);
            DivergenceBridge(csf_device);
            PrimalVariablesBridge(csf_device);
        }

        // Filter solution
        FilterBridge(csf_device);

        // Compute the motion field
        MotionFieldBridge(csf_device);

        // BridgeBack
        BridgeBack(&csf_host, csf_device);

        // Free variables of variables associated to this level
        csf_host.freeLevelVariables();

        // Copy motion field and images to CPU
        csf_host.copyAllSolutions(dx[ctf_levels - i - 1].data(), dy[ctf_levels - i - 1].data(), dz[ctf_levels - i - 1].data(),
                                  depth[level_image].data(), depth_old[level_image].data(), colour[level_image].data(), colour_old[level_image].data(),
                                  xx[level_image].data(), xx_old[level_image].data(), yy[level_image].data(), yy_old[level_image].data());

        //=========================================================================
        //                              Cuda - end
        //=========================================================================
    }
}
bool PD_flow::GetFromRGBDImages(cv::Mat &rgb_img, cv::Mat &depth_img)
{
    if (rgb_img.empty() || depth_img.empty())
    {
        std::cerr << "Invalid input images." << std::endl;
        return false;
    }

    // Resize images
    cv::resize(rgb_img, rgb_img, cv::Size(640 / cam_mode, 480 / cam_mode));
    cv::resize(depth_img, depth_img, cv::Size(640 / cam_mode, 480 / cam_mode));

    // Convert RGB image to grayscale
    cv::cvtColor(rgb_img, rgb_img, cv::COLOR_BGR2GRAY);

    // Ensure depth image has the correct type
    if (depth_img.type() != CV_16UC1)
    {
        std::cerr << "Depth image must be of type CV_16UC1." << std::endl;
        return false;
    }

    // Map cv::Mat data to Eigen matrices and convert to float
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> colour_wf_temp =
        Eigen::Map<Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(rgb_img.data, rgb_img.rows, rgb_img.cols);
    colour_wf = colour_wf_temp.cast<float>();

    Eigen::Matrix<unsigned short, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> depth_wf_temp =
        Eigen::Map<Eigen::Matrix<unsigned short, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            reinterpret_cast<unsigned short *>(depth_img.data), depth_img.rows, depth_img.cols);
    depth_wf = depth_wf_temp.cast<float>();

    return true;
}