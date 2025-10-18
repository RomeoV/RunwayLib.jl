#include "libposeest.h"
#include <stdio.h>
#include <time.h>

int main() {
    WorldPointF64 world_points[] = {
        {0, 50, 0},
        {3000, 50, 0},
        {3000, -50, 0},
        {0, -50, 0},
    };
    ProjectionPointF64 projections[] = {
        {1908.3869731285338, 1413.7171688399328},
        {1982.0042057862263, 1086.0904145429126},
        {2127.0552446444053, 1082.2921194158434},
        {2269.5962158465413, 1404.2585719400536},
    };
    int num_points = 4;

    PoseEstimateF64 pose_result;
    CameraMatrixF64 cammat = {
        // camera matrix row-by-row, will be transposed when loading into Julia,
        // which is column-major.
        {-7246.4, 0.0, 2048.0, 0.0, -7246.4, 1500.0, 0.0, 0.0, 1.0},
        4096.0,
        3000.0};
    WorldPointF64 initial_guess_pos = {-1000, 0, 150};
    RotationF64 initial_guess_rot = {0., 0., 0.};

    int result = estimate_pose_6dof(world_points, projections, num_points, NULL,
                                    COV_DEFAULT, &cammat, &initial_guess_pos,
                                    &initial_guess_rot, &pose_result);

    if (result != POSEEST_SUCCESS) {
        printf("Error: %s\n", get_error_message(result));
        printf("Details: %s\n", get_last_error_detail());
        return 1;
    }

    printf("Pose estimated successfully\n");
    printf("Position: (%.2f, %.2f, %.2f) m\n", pose_result.position.x,
           pose_result.position.y, pose_result.position.z);
    printf("Rotation (yaw, pitch, roll): (%.4f, %.4f, %.4f) rad\n",
           pose_result.rotation.yaw, pose_result.rotation.pitch,
           pose_result.rotation.roll);
    printf("Residual norm: %.6f\n", pose_result.residual_norm);
    printf("Converged: %s\n", pose_result.converged ? "yes" : "no");

    return 0;
}
