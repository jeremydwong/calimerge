/*
 * test_enumerate.c - Quick test of camera enumeration
 *
 * Compile: clang -o test_enumerate test_enumerate.c -L. -lcalimerge -Wl,-rpath,.
 * Run: ./test_enumerate
 */

#include <stdio.h>
#include "calimerge_platform.h"

int main(void) {
    printf("Calimerge Camera Enumeration Test\n");
    printf("==================================\n\n");

    int result = cm_init();
    if (result != CM_OK) {
        printf("Failed to initialize: %d\n", result);
        return 1;
    }

    CM_Camera cameras[CM_MAX_CAMERAS];
    int count = cm_enumerate_cameras(cameras, CM_MAX_CAMERAS);

    if (count < 0) {
        printf("Enumeration failed: %d\n", count);
        return 1;
    }

    printf("Found %d camera(s):\n\n", count);

    for (int i = 0; i < count; i++) {
        CM_Camera *cam = &cameras[i];
        printf("Camera %d:\n", i);
        printf("  Serial:     %s\n", cam->serial_number);
        printf("  Name:       %s\n", cam->display_name);
        printf("  Resolution: %dx%d\n", cam->width, cam->height);
        printf("  FPS:        %d\n", cam->fps);
        printf("  Supported resolutions:\n");
        for (int r = 0; r < cam->supported_resolution_count; r++) {
            printf("    - %dx%d\n",
                   cam->supported_resolutions[r].width,
                   cam->supported_resolutions[r].height);
        }
        printf("\n");
    }

    cm_shutdown();
    return 0;
}
