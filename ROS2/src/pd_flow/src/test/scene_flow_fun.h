/*****************************************************************************
**				Primal-Dual Scene Flow for RGB-D cameras					**
**				----------------------------------------					**
**																			**
**	Copyright(c) 2015, Mariano Jaimez Tarifa, University of Malaga			**
**	Copyright(c) 2015, Mohamed Souiai, Technical University of Munich		**
**	Copyright(c) 2015, MAPIR group, University of Malaga					**
**	Copyright(c) 2015, Computer Vision group, Tech. University of Munich	**
**																			**
**  This program is free software: you can redistribute it and/or modify	**
**  it under the terms of the GNU General Public License (version 3) as		**
**	published by the Free Software Foundation.								**
**																			**
**  This program is distributed in the hope that it will be useful, but		**
**	WITHOUT ANY WARRANTY; without even the implied warranty of				**
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the			**
**  GNU General Public License for more details.							**
**																			**
**  You should have received a copy of the GNU General Public License		**
**  along with this program.  If not, see <http://www.gnu.org/licenses/>.	**
**																			**
*****************************************************************************/

#include <Eigen/Core>
#include "pdflow_cudalib.h"
#include <stdio.h>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

inline int stoi(char *c) { return int(std::strtol(c, NULL, 10)); }

using namespace std;
using Eigen::MatrixXf;

// Para astra: height: 480, width: 640

class PD_flow_fun
{
public:
    float fps;                    // In Hz
    unsigned int cam_mode;        // (1 - 640 x 480, 2 - 320 x 240)
    unsigned int ctf_levels;      // Number of levels used in the coarse-to-fine scheme (always dividing by two)
    unsigned int num_max_iter[6]; // Max number of iterations distributed homogeneously between all levels
    float g_mask[25];

    // Matrices that store the original images with the image resolution
    MatrixXf colour_wf;
    MatrixXf depth_wf;

    // Matrices that store the images downsampled
    vector<MatrixXf> colour;
    vector<MatrixXf> colour_old;
    vector<MatrixXf> depth;
    vector<MatrixXf> depth_old;
    vector<MatrixXf> xx;
    vector<MatrixXf> xx_old;
    vector<MatrixXf> yy;
    vector<MatrixXf> yy_old;

    // Motion field
    vector<MatrixXf> dx;
    vector<MatrixXf> dy;
    vector<MatrixXf> dz;

    // Camera properties
    float fovh; // In radians
    float fovv; // In radians

    // Max resolution of the coarse-to-fine scheme.
    unsigned int rows;
    unsigned int cols;

    // Optimization Parameters
    float mu, lambda_i, lambda_d;

    // Cuda
    CSF_cuda csf_host, *csf_device;

    // Methods
    void createImagePyramidGPU(MatrixXf colour_wf, MatrixXf depth_wf);
    void solveSceneFlowGPU();
    void freeGPUMemory();
    void initializeCUDA();
    cv::Mat createImage() const;
    void showAndSaveResults();

    // void initializePDFlow();

    PD_flow_fun(unsigned int cam_mode_config, unsigned int fps_config, unsigned int rows_config);
};
