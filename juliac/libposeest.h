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

/**
 * Camera matrix for pose estimation.
 *
 * The matrix is a 3x3 camera intrinsics matrix stored in ROW-MAJOR order:
 *   [fx,  0, cx,
 *     0, fy, cy,
 *     0,  0,  1]
 *
 * IMPORTANT: The matrix array uses C's row-major layout. It will be transposed
 * when loaded by Julia (which uses column-major ordering).
 *
 * For offset coordinate systems (coordinate_system_tag = 1), focal lengths
 * fx and fy should be negative.
 *
 * Fields:
 *   - matrix: 9 elements in row-major order (row1, row2, row3)
 *   - image_width: Width of the image in pixels
 *   - image_height: Height of the image in pixels
 *   - coordinate_system_tag: 0 for centered, 1 for offset (only offset supported)
 */
typedef struct {
    double matrix[9];
    double image_width;
    double image_height;
    CameraConfigType coordinate_system_tag; // 0 for centered, 1 for offset
} CameraMatrixF64;
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
