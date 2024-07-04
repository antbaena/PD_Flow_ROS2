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

#include "scene_flow_visualization.h"


// ------------------------------------------------------
//						MAIN
// ------------------------------------------------------

int main(int num_arg, char *argv[])
{
	//==============================================================================
	//						Read function arguments
	//==============================================================================
	unsigned int cam_mode = 1, fps = 30, rows = 240;	//Default values

	//Initialize the scene flow object and visualization
	PD_flow sceneflow(cam_mode, fps, rows);
	sceneflow.initializePDFlow();

    //==============================================================================
    //									Main operation
    //==============================================================================

	int pushed_key = 0;
    int stop = 0;
    bool working = false;
	CTicTac	clock;

    while (!stop)
    {

        if (sceneflow.window.keyHit())
            pushed_key = sceneflow.window.getPushedKey();

        else
            pushed_key = 0;

        switch (pushed_key) {

        //Capture new frame
        case  'n':
            sceneflow.CaptureFrame();
            clock.Tic();
            sceneflow.createImagePyramidGPU();
            sceneflow.solveSceneFlowGPU();
            cout << endl << "PD-Flow runtime: " << 1000.f*clock.Tac();

			sceneflow.updateScene();
            break;

        //Start/Stop continuous estimation
        case 's':
            working = !working;
			clock.Tic();
            break;

        //Close the program
        case 'e':
            stop = 1;
            break;
        }

        if (working == 1)
        {
            while(clock.Tac() < 1.f/sceneflow.fps);
			const float exec_time = clock.Tac();
			clock.Tic();
            if (exec_time > 1.05f/sceneflow.fps)
				printf("\n Not enough time to compute the scene flow at %d Hz", int(sceneflow.fps));

            sceneflow.CaptureFrame();
            clock.Tic();
            sceneflow.createImagePyramidGPU();
            sceneflow.solveSceneFlowGPU();
			const float total_time = 1000.f*clock.Tac();
            cout << endl << "PD-Flow runtime (ms): " << total_time;

			sceneflow.updateScene();
        }
    }

    sceneflow.freeGPUMemory();
    sceneflow.CloseCamera();
	return 0;

}