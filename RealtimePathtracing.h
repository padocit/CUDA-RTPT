#pragma once
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

#include <glm/glm.hpp>

void UploadScene();
void LaunchRaytracingKernel(cudaSurfaceObject_t      dstSurfMipMap0,
                            unsigned int             imageWidth,
                            unsigned int             imageHeight,
                            const glm::vec3&         lookfrom,
                            const glm::vec3&         lookat,
                            float                    fov,
                            cudaStream_t             streamToRun);
