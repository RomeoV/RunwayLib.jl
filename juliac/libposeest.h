#ifndef LIBPOSEEST_H_
#define LIBPOSEEST_H_

#ifdef __cplusplus
extern "C" {
#endif

// Data structures for C API
typedef struct {
    double x, y, z;
} WorldPointF64;

typedef struct {
    double x, y;
} ProjectionPointF64;

typedef struct {
    double yaw, pitch, roll;
} Rotation_C;

typedef struct {
    WorldPointF64 position;
    Rotation_C rotation;
    double residual_norm;
    int converged;
} PoseEstimate_C;

typedef enum {
    CAMERA_CONFIG_CENTERED = 0,
    CAMERA_CONFIG_OFFSET = 1
} CameraConfigType;

// Error codes
#define POSEEST_SUCCESS 0
#define POSEEST_ERROR_INVALID_INPUT -1
#define POSEEST_ERROR_BEHIND_CAMERA -2
#define POSEEST_ERROR_NO_CONVERGENCE -3
#define POSEEST_ERROR_INSUFFICIENT_POINTS -4

// Function declarations

// Simple test function (existing)
int test_estimators();

// Enhanced API functions
int estimate_pose_6dof(const WorldPointF64 *runway_corners,
                       const ProjectionPointF64 *projections, int num_points,
                       CameraConfigType camera_config, PoseEstimate_C *result);

int estimate_pose_3dof(const WorldPointF64 *runway_corners,
                       const ProjectionPointF64 *projections, int num_points,
                       const Rotation_C *known_rotation,
                       CameraConfigType camera_config, PoseEstimate_C *result);

// Utility functions
int project_point(const WorldPointF64 *camera_position,
                  const Rotation_C *camera_rotation,
                  const WorldPointF64 *world_point,
                  CameraConfigType camera_config, ProjectionPointF64 *result);

// Get error message for error code
const char *get_error_message(int error_code);

#ifdef __cplusplus
}
#endif

#endif // LIBPOSEEST_H_
