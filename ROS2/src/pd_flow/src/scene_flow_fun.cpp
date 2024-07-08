#include "scene_flow_fun.h"

PD_flow_fun::PD_flow_fun(unsigned int cam_mode_config, unsigned int fps_config, unsigned int rows_config)
{
    rows = rows_config;      //Maximum size of the coarse-to-fine scheme - Default 240 (QVGA)
    cols = rows*320/240;
    cam_mode = cam_mode_config;   // (1 - 640 x 480, 2 - 320 x 240), Default - 1
    ctf_levels = round(log2(rows/15)) + 1;
    fovh = M_PI*62.5f/180.f;
    fovv = M_PI*45.f/180.f;
    fps = fps_config;       //In Hz, Default - 30

    //Iterations of the primal-dual solver at each pyramid level.
    //Maximum value set to 100 at the finest level
    for (int i=5; i>=0; i--)
    {
        if (i >= static_cast<int>(ctf_levels - 1))
            num_max_iter[i] = 100;
        else
            num_max_iter[i] = num_max_iter[i+1]-15;
    }

    //num_max_iter[ctf_levels-1] = 0.f;

    //Compute gaussian mask
    float v_mask[5] = {1.f,4.f,6.f,4.f,1.f};
    for (unsigned int i=0; i<5; i++)
        for (unsigned int j=0; j<5; j++)
            g_mask[i+5*j] = v_mask[i]*v_mask[j]/256.f;

    //Matrices that store the original and filtered images with the image resolution
    colour_wf.resize(480/cam_mode,640/cam_mode);
    depth_wf.resize(480/cam_mode,640/cam_mode);

    //Resize vectors according to levels
    dx.resize(ctf_levels); dy.resize(ctf_levels); dz.resize(ctf_levels);

    const unsigned int width = colour_wf.cols();
    const unsigned int height = colour_wf.rows();
    unsigned int s, cols_i, rows_i;

    for (unsigned int i = 0; i<ctf_levels; i++)
    {
        s = pow(2.f,int(ctf_levels-(i+1)));
        cols_i = cols/s; rows_i = rows/s;
        dx[ctf_levels-i-1].resize(rows_i,cols_i);
        dy[ctf_levels-i-1].resize(rows_i,cols_i);
        dz[ctf_levels-i-1].resize(rows_i,cols_i);
    }

    //Resize pyramid
    const unsigned int pyr_levels = round(log2(width/cols)) + ctf_levels;
    colour.resize(pyr_levels);
    colour_old.resize(pyr_levels);
    depth.resize(pyr_levels);
    depth_old.resize(pyr_levels);
    xx.resize(pyr_levels);
    xx_old.resize(pyr_levels);
    yy.resize(pyr_levels);
    yy_old.resize(pyr_levels);

    for (unsigned int i = 0; i<pyr_levels; i++)
    {
        s = pow(2.f,int(i));
        colour[i].resize(height/s, width/s);
        colour_old[i].resize(height/s, width/s);
        colour[i].setZero();
        colour_old[i].setZero();
        depth[i].resize(height/s, width/s);
        depth_old[i].resize(height/s, width/s);
        depth[i].setZero();
        depth_old[i].setZero();
        xx[i].resize(height/s, width/s);
        xx_old[i].resize(height/s, width/s);
        xx[i].setZero();
        xx_old[i].setZero();
        yy[i].resize(height/s, width/s);
        yy_old[i].resize(height/s, width/s);
        yy[i].setZero();
        yy_old[i].setZero();
    }

    //Parameters of the variational method
    lambda_i = 0.04f;
    lambda_d = 0.35f;
    mu = 75.f;
}

void PD_flow_fun::createImagePyramidGPU(MatrixXf colour_wf, MatrixXf depth_wf)
{
    //Copy new frames to the scene flow object
    csf_host.copyNewFrames(colour_wf.data(), depth_wf.data());

    //Copy scene flow object to device
    csf_device = ObjectToDevice(&csf_host);

    unsigned int pyr_levels = round(log2(640/(cam_mode*cols))) + ctf_levels;
    GaussianPyramidBridge(csf_device, pyr_levels, cam_mode);

    //Copy scene flow object back to host
    BridgeBack(&csf_host, csf_device);
}

void PD_flow_fun::solveSceneFlowGPU()
{
    //Define variables
    unsigned int s;
    unsigned int cols_i, rows_i;
    unsigned int level_image;
    unsigned int num_iter;

    //For every level (coarse-to-fine)
    for (unsigned int i=0; i<ctf_levels; i++)
    {
        const unsigned int width = colour_wf.cols();
        s = pow(2.f,int(ctf_levels-(i+1)));
        cols_i = cols/s;
        rows_i = rows/s;
        level_image = ctf_levels - i + round(log2(width/cols)) - 1;

        //=========================================================================
        //                              Cuda - Begin
        //=========================================================================

        //Cuda allocate memory
        csf_host.allocateMemoryNewLevel(rows_i, cols_i, i, level_image);

        //Cuda copy object to device
        csf_device = ObjectToDevice(&csf_host);

        //Assign zeros to the corresponding variables
        AssignZerosBridge(csf_device);

        //Upsample previous solution
        if (i>0)
            UpsampleBridge(csf_device);

        //Compute connectivity (Rij)
        RijBridge(csf_device);

        //Compute colour and depth derivatives
        ImageGradientsBridge(csf_device);
        WarpingBridge(csf_device);

        //Compute mu_uv and step sizes for the primal-dual algorithm
        MuAndStepSizesBridge(csf_device);

        //Primal-Dual solver
        for (num_iter = 0; num_iter < num_max_iter[i]; num_iter++)
        {
            GradientBridge(csf_device);
            DualVariablesBridge(csf_device);
            DivergenceBridge(csf_device);
            PrimalVariablesBridge(csf_device);
        }

        //Filter solution
        FilterBridge(csf_device);

        //Compute the motion field
        MotionFieldBridge(csf_device);

        //BridgeBack
        BridgeBack(&csf_host, csf_device);

        //Free variables of variables associated to this level
        csf_host.freeLevelVariables();

        //Copy motion field and images to CPU
        csf_host.copyAllSolutions(dx[ctf_levels-i-1].data(), dy[ctf_levels-i-1].data(), dz[ctf_levels-i-1].data(),
                        depth[level_image].data(), depth_old[level_image].data(), colour[level_image].data(), colour_old[level_image].data(),
                        xx[level_image].data(), xx_old[level_image].data(), yy[level_image].data(), yy_old[level_image].data());

        //For debugging
        //DebugBridge(csf_device);

        //=========================================================================
        //                              Cuda - end
        //=========================================================================
    }
}

void PD_flow_fun::freeGPUMemory()
{
    csf_host.freeDeviceMemory();
}

void PD_flow_fun::initializeCUDA()
{
    //Read parameters
    csf_host.readParameters(rows, cols, lambda_i, lambda_d, mu, g_mask, ctf_levels, cam_mode, fovh, fovv);

    //Allocate memory
    csf_host.allocateDevMemory();
}

void PD_flow_fun::initializePDFlow()
{
    initializeCUDA();
    CaptureFrame();
    createImagePyramidGPU(colour_wf, depth_wf);
    CaptureFrame();
    createImagePyramidGPU(colour_wf, depth_wf);
    solveSceneFlowGPU();
}
