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
} RotationF64;

typedef struct {
    WorldPointF64 position;
    RotationF64 rotation;
    double residual_norm;
    int converged;
} PoseEstimateF64;

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

typedef enum {
    COV_DEFAULT = 0,        // Use default noise model (pointer can be null)
    COV_SCALAR = 1,         // Single noise value for all keypoints/directions
    COV_DIAGONAL_FULL = 2,  // Diagonal matrix (length = 2*n_keypoints)
    COV_BLOCK_DIAGONAL = 3, // 2x2 matrix per keypoint (length = 4*n_keypoints)
    COV_FULL_MATRIX = 4     // Full covariance matrix (length = 4*n_keypoints^2)
} CovarianceType;

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
                       const double *covariance_data,
                       CovarianceType covariance_type,
                       const CameraMatrixF64 *cammat,
                       const WorldPointF64 *initial_guess_pos,
                       const RotationF64 *initial_guess_rot,
                       PoseEstimateF64 *result);

// Get error message for error code
const char *get_error_message(int error_code);

// Get detailed error message from last exception (useful for debugging)
const char *get_last_error_detail();

#ifdef __cplusplus
}
#endif

#endif // LIBPOSEEST_H_
