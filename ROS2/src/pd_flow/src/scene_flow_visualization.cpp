
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
}

void PD_flow::createImagePyramidGPU()
{
    // Copy new frames to the scene flow object
    csf_host.copyNewFrames(colour_wf.data(), depth_wf.data());

    // Copy scene flow object to device
    csf_device = ObjectToDevice(&csf_host);

    unsigned int pyr_levels = round(log2(640 / (cam_mode * cols))) + ctf_levels;
    GaussianPyramidBridge(csf_device, pyr_levels, cam_mode);

    // Copy scene flow object back to host
    BridgeBack(&csf_host, csf_device);
}

void PD_flow::process_frame(cv::Mat &rgb_image, cv::Mat &depth_image)
{
    if (rgb_image.size() != depth_image.size() || rgb_image.empty() || depth_image.empty())
    {
        cout << "The RGB and the depth images don't have the same size or are empty." << endl;
        return;
    }

    int width = rgb_image.cols;
    int height = rgb_image.rows;

    // Resize the colour_wf and depth_wf matrices if necessary
    if (colour_wf.rows() != width || colour_wf.rows() != height)
    {
        colour_wf.resize(height, width);
        depth_wf.resize(height, width);
    }

    // Read new frame
    for (int yc = height - 1; yc >= 0; --yc)
    {
        for (int xc = width - 1; xc >= 0; --xc)
        {
            cv::Vec3b pRgb = rgb_image.at<cv::Vec3b>(yc, xc);
            float pDepth = depth_image.at<float>(yc, xc);

            colour_wf(yc, xc) = 0.299 * pRgb[2] + 0.587 * pRgb[1] + 0.114 * pRgb[0];
            depth_wf(yc, xc) = 0.001f * pDepth;
        }
    }
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

void PD_flow::initializeCUDA()
{
    // Read parameters
    csf_host.readParameters(rows, cols, lambda_i, lambda_d, mu, g_mask, ctf_levels, cam_mode, fovh, fovv);

    // Allocate memory
    csf_host.allocateDevMemory();
}

void PD_flow::freeGPUMemory()
{
    csf_host.freeDeviceMemory();
}

void PD_flow::initializePDFlow()
{
    initializeCUDA();
}

void PD_flow::updateScene()
{
    // Crear imágenes para mostrar el campo de movimiento, color y profundidad
    cv::Mat motion_field = cv::Mat::zeros(rows, cols, CV_8UC3);
    cv::Mat color_image(rows, cols, CV_8UC3);
    cv::Mat depth_image(rows, cols, CV_8UC1);

    const unsigned int repr_level = round(log2(colour_wf.cols() / cols));

    for (unsigned int v = 0; v < rows; v++)
    {
        for (unsigned int u = 0; u < cols; u++)
        {
            // Obtener la profundidad y los desplazamientos de cada punto
            float depth_value = depth[repr_level](v, u);
            if (depth_value > 0.1f)
            {
                // Escalar los valores de desplazamiento para visualizarlos mejor
                float dx_scaled = dx[0](v, u) * 10;
                float dy_scaled = dy[0](v, u) * 10;

                // Dibujar la línea que representa el vector de movimiento
                cv::Point2f start_point(u, v);
                cv::Point2f end_point(u + dx_scaled, v + dy_scaled);

                // Dibujar el vector en la imagen (usar color azul)
                cv::arrowedLine(motion_field, start_point, end_point, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
            }

            // Convertir valores de color y profundidad a formatos adecuados para visualización
            color_image.at<cv::Vec3b>(v, u) = cv::Vec3b(colour_wf(v, u), colour_wf(v, u), colour_wf(v, u));

            depth_image.at<uint8_t>(v, u) = static_cast<uint8_t>(depth_value * 255); // Normalizar la profundidad para visualización
        }
    }

    // Normalizar la imagen de profundidad para visualización
    cv::Mat depth_image_display;
    cv::normalize(depth_image, depth_image_display, 0, 255, cv::NORM_MINMAX);
    depth_image_display.convertTo(depth_image_display, CV_8UC1);


    cv::Mat motion_field_display;
    cv::normalize(motion_field, motion_field_display, 0, 255, cv::NORM_MINMAX);
    motion_field_display.convertTo(motion_field_display, CV_8UC1);

    // Mostrar la imagen del campo de movimiento
    cv::imshow("Motion Field", motion_field_display);

    // Mostrar la imagen de color
    cv::imshow("Color Image", color_image);

    // Mostrar la imagen de profundidad
    cv::imshow("Depth Image", depth_image_display);

    cv::waitKey(1); // Espera breve para actualizar las ventanas
}