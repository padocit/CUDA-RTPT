/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 /* This example demonstrates how to use the CUDA-D3D11 External Resource Interoperability APIs
  *  to update D3D11 buffers from CUDA and synchronize between D3D11 and CUDA with Keyed Mutexes.
  */

/*
특징: 
1. 그래픽스 인터롭 사용중이므로 Keyed Mutexes 사용하지않음
2. CUDA Compute로 (CUDA surface <=> D3D11 UAV Texture) 렌더 후 곧바로 Backbuffer에 복사하므로,
    D3D11 렌더링 파이프라인을 거치지 않음. (bypass)
    따라서 IA, VS, RS, PS 세팅하는 과정이 모두 필요없음.
    단 Swapchian, Viewport, Backbuffer 등 기본적인 것은 필요.
*/

#pragma warning(disable : 4312)

  // includes for Windows
#include <windows.h>

// includes for multimedia
#include <mmsystem.h>

// This header inclues all the necessary D3D11 and CUDA includes
#include <cuda_d3d11_interop.h>
#include <cuda_runtime_api.h>
#include <d3dcompiler.h>
#include <dxgi1_2.h>
#include <dynlink_d3d11.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // includes cuda.h and cuda_runtime_api.h
#include <rendercheck_d3d11.h>

#include <glm/glm.hpp>

#include "RealtimePathtracing.h"
//#include "Vertex.h"

#define MAX_EPSILON 10

static char* SDK_name = "simpleD3D11";

//-----------------------------------------------------------------------------
// Global variables
//-----------------------------------------------------------------------------
IDXGIAdapter1* g_pCudaCapableAdapter = NULL; // Adapter to use
ID3D11Device* g_pd3dDevice = NULL; // Our rendering device
ID3D11DeviceContext* g_pd3dDeviceContext = NULL;
IDXGISwapChain* g_pSwapChain = NULL; // The swap chain of the window
ID3D11RenderTargetView* g_pSwapChainRTV = NULL; // The Render target view on the swap chain ( used for clear)

// Cuda Surf
ID3D11Texture2D* myTex = nullptr;
ID3D11UnorderedAccessView* myUAV = nullptr;
cudaGraphicsResource* cudaRes = nullptr;
cudaSurfaceObject_t dstSurfMipMap0;

// testing/tracing function used pervasively in tests.  if the condition is unsatisfied
// then spew and fail the function immediately (doing no cleanup)
#define AssertOrQuit(x)                                                                           \
    if (!(x)) {                                                                                   \
        fprintf(stdout, "Assert unsatisfied in %s at %s:%d\n", __FUNCTION__, __FILE__, __LINE__); \
        return 1;                                                                                 \
    }

bool g_bDone = false;
bool g_bPassed = true;

int* pArgc = NULL;
char** pArgv = NULL;

const unsigned int g_WindowWidth = 1920;
const unsigned int g_WindowHeight = 1080;

int g_iFrameToCompare = 10;

cudaStream_t cuda_stream;

// Camera Rotation
bool    g_isDragging = false;
POINT   g_prevMousePos = {};
int dx = 0, dy = 0;
glm::vec3 lookat = glm::vec3(0.0f);
glm::vec3 lookfrom = glm::vec3{ 13.0, 2.0, -3.0 };
float   g_yaw = 0.0f;
float   g_pitch = 0.0f;
float   g_fov = 20.0f;
bool    dirtyflagRot = false;

// Camera Panning
bool      g_isPanning = false;
bool      dirtyflagPan = false;

//-----------------------------------------------------------------------------
// Forward declarations
//-----------------------------------------------------------------------------
HRESULT InitD3D(HWND hWnd);

bool DrawScene();
void Cleanup();
void Render();
void CreateCudaSurf();
void UpdateRot();
void UpdatePan();

LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

#define NAME_LEN 512

bool findCUDADevice()
{
    int deviceCount = 0;
    // This function call returns 0 if there are no CUDA capable devices.
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("> There are no device(s) supporting CUDA\n");
        return false;
    }
    else {
        printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
    }

    return true;
}

bool findDXDevice(char* dev_name)
{
    HRESULT   hr = S_OK;
    cudaError cuStatus;
    int       cuda_dev = -1;

    // Iterate through the candidate adapters
    IDXGIFactory1* pFactory;
    hr = sFnPtr_CreateDXGIFactory(__uuidof(IDXGIFactory1), (void**)(&pFactory));

    if (!SUCCEEDED(hr)) {
        printf("> No DXGI Factory created.\n");
        return false;
    }

    UINT adapter = 0;

    for (; !g_pCudaCapableAdapter; ++adapter) {
        // Get a candidate DXGI adapter
        IDXGIAdapter1* pAdapter = NULL;

        hr = pFactory->EnumAdapters1(adapter, &pAdapter);

        if (FAILED(hr)) {
            break; // no compatible adapters found
        }

        // Query to see if there exists a corresponding compute device
        int cuDevice;
        cuStatus = cudaD3D11GetDevice(&cuDevice, pAdapter);
        printLastCudaError("cudaD3D11GetDevice failed"); // This prints and resets the cudaError to cudaSuccess

        if (cudaSuccess == cuStatus) {
            // If so, mark it as the one against which to create our d3d11 device
            g_pCudaCapableAdapter = pAdapter;
            g_pCudaCapableAdapter->AddRef();
            cuda_dev = cuDevice;
            printf("\ncuda device id selected = %d\n", cuda_dev);
        }

        pAdapter->Release();
    }

    printf("> Found %d D3D11 Adapater(s).\n", (int)adapter);

    pFactory->Release();

    if (!g_pCudaCapableAdapter) {
        printf("> Found 0 D3D11 Adapater(s) /w Compute capability.\n");
        return false;
    }

    DXGI_ADAPTER_DESC adapterDesc;
    g_pCudaCapableAdapter->GetDesc(&adapterDesc);
    wcstombs(dev_name, adapterDesc.Description, 128);

    checkCudaErrors(cudaSetDevice(cuda_dev));
    checkCudaErrors(cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking));

    printf("> Found 1 D3D11 Adapater(s) /w Compute capability.\n");
    printf("> %s\n", dev_name);

    return true;
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    char  device_name[256];
    char* ref_file = NULL;

    pArgc = &argc;
    pArgv = argv;

    printf("[%s] - Starting...\n", SDK_name);

    if (!findCUDADevice()) // Search for CUDA GPU
    {
        printf("> CUDA Device NOT found on \"%s\".. Exiting.\n", device_name);
        exit(EXIT_SUCCESS);
    }

    if (!dynlinkLoadD3D11API()) // Search for D3D API (locate drivers, does not mean device is found)
    {
        printf("> D3D11 API libraries NOT found on.. Exiting.\n");
        dynlinkUnloadD3D11API();
        exit(EXIT_SUCCESS);
    }

    if (!findDXDevice(device_name)) // Search for D3D Hardware Device
    {
        printf("> D3D11 Graphics Device NOT found.. Exiting.\n");
        dynlinkUnloadD3D11API();
        exit(EXIT_SUCCESS);
    }

    // command line options
    if (argc > 1) {
        // automatied build testing harness
        if (checkCmdLineFlag(argc, (const char**)argv, "file"))
            getCmdLineArgumentString(argc, (const char**)argv, "file", &ref_file);
    }

    //
    // create window
    //
    // Register the window class
    WNDCLASSEX wc = { sizeof(WNDCLASSEX),
                     CS_CLASSDC,
                     MsgProc,
                     0L,
                     0L,
                     GetModuleHandle(NULL),
                     NULL,
                     NULL,
                     NULL,
                     NULL,
                     "CUDA SDK",
                     NULL };
    RegisterClassEx(&wc);

    // Create the application's window
    int  xBorder = ::GetSystemMetrics(SM_CXSIZEFRAME);
    int  yMenu = ::GetSystemMetrics(SM_CYMENU);
    int  yBorder = ::GetSystemMetrics(SM_CYSIZEFRAME);
    HWND hWnd = CreateWindow(wc.lpszClassName,
        "CUDA/D3D11 InterOP",
        WS_OVERLAPPEDWINDOW,
        0,
        0,
        g_WindowWidth + 2 * xBorder,
        g_WindowHeight + 2 * yBorder + yMenu,
        NULL,
        NULL,
        wc.hInstance,
        NULL);

    ShowWindow(hWnd, SW_SHOWDEFAULT);
    UpdateWindow(hWnd);

    // Initialize Direct3D
    if (!SUCCEEDED(InitD3D(hWnd))) {
        printf("InitD3D Failed.. Exiting..\n");
        exit(EXIT_FAILURE);
    }

    // Creat Cuda Surf
    CreateCudaSurf();

    // Copy Host to Device
    UploadScene();

    //
    // the main loop
    //
    while (false == g_bDone) {
        Render();

        //
        // handle I/O
        //
        MSG msg;
        ZeroMemory(&msg, sizeof(msg));

        while (msg.message != WM_QUIT) {
            if (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
            else {

                if (dirtyflagRot)
                   UpdateRot();

                if (dirtyflagPan)
                   UpdatePan();

                Render();

                if (ref_file) {
                    for (int count = 0; count < g_iFrameToCompare; count++) {
                        Render();
                    }

                    const char* cur_image_path = "simpleD3D11.ppm";

                    // Save a reference of our current test run image
                    CheckRenderD3D11::ActiveRenderTargetToPPM(g_pd3dDevice, cur_image_path);

                    // compare to offical reference image, printing PASS or FAIL.
                    g_bPassed = CheckRenderD3D11::PPMvsPPM(cur_image_path, ref_file, argv[0], MAX_EPSILON, 0.15f);

                    g_bDone = true;

                    Cleanup();

                    PostQuitMessage(0);
                }
                else {
                    g_bPassed = true;
                }
            }
        }
    };

    // Release D3D Library (after message loop)
    dynlinkUnloadD3D11API();

    // Unregister windows class
    UnregisterClass(wc.lpszClassName, wc.hInstance);

    //
    // and exit
    //
    printf("> %s running on %s exiting...\n", SDK_name, device_name);

    exit(g_bPassed ? EXIT_SUCCESS : EXIT_FAILURE);
}


//-----------------------------------------------------------------------------
// Name: InitD3D()
// Desc: Initializes Direct3D
//-----------------------------------------------------------------------------
HRESULT InitD3D(HWND hWnd)
{
    HRESULT   hr = S_OK;
    cudaError cuStatus;

    // Set up the structure used to create the device and swapchain
    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory(&sd, sizeof(sd));
    sd.BufferCount = 1;
    sd.BufferDesc.Width = g_WindowWidth;
    sd.BufferDesc.Height = g_WindowHeight;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;

    D3D_FEATURE_LEVEL tour_fl[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
    D3D_FEATURE_LEVEL flRes;
    // Create device and swapchain
    hr = sFnPtr_D3D11CreateDeviceAndSwapChain(g_pCudaCapableAdapter,
        D3D_DRIVER_TYPE_UNKNOWN, // D3D_DRIVER_TYPE_HARDWARE,
        NULL,                    // HMODULE Software
        0,                       // UINT Flags
        tour_fl,                 // D3D_FEATURE_LEVEL* pFeatureLevels
        2,                       // FeatureLevels
        D3D11_SDK_VERSION,       // UINT SDKVersion
        &sd,                     // DXGI_SWAP_CHAIN_DESC* pSwapChainDesc
        &g_pSwapChain,           // IDXGISwapChain** ppSwapChain
        &g_pd3dDevice,           // ID3D11Device** ppDevice
        &flRes,                  // D3D_FEATURE_LEVEL* pFeatureLevel
        &g_pd3dDeviceContext     // ID3D11DeviceContext** ppImmediateContext
    );

    AssertOrQuit(SUCCEEDED(hr));

    g_pCudaCapableAdapter->Release();

    // Get the immediate DeviceContext
    g_pd3dDevice->GetImmediateContext(&g_pd3dDeviceContext);

    // Create a render target view of the swapchain
    ID3D11Texture2D* pBuffer;
    hr = g_pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&pBuffer);
    AssertOrQuit(SUCCEEDED(hr));

    hr = g_pd3dDevice->CreateRenderTargetView(pBuffer, NULL, &g_pSwapChainRTV);
    AssertOrQuit(SUCCEEDED(hr));
    pBuffer->Release();

    g_pd3dDeviceContext->OMSetRenderTargets(1, &g_pSwapChainRTV, NULL);

    // Setup the viewport
    D3D11_VIEWPORT vp;
    vp.Width = g_WindowWidth;
    vp.Height = g_WindowHeight;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;
    g_pd3dDeviceContext->RSSetViewports(1, &vp);

    // 1) 텍스처 생성
    D3D11_TEXTURE2D_DESC texDesc = {};
    texDesc.Width = g_WindowWidth;                          // 화면/버퍼 가로 크기
    texDesc.Height = g_WindowHeight;                         // 화면/버퍼 세로 크기
    texDesc.MipLevels = 1;                              // 단일 레벨만 쓰는 경우
    texDesc.ArraySize = 1;
    texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;     // uchar4 surf2Dwrite 예시
    texDesc.SampleDesc.Count = 1;
    texDesc.SampleDesc.Quality = 0;
    texDesc.Usage = D3D11_USAGE_DEFAULT;
    texDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS    // CUDA surf 읽기/쓰기
        | D3D11_BIND_SHADER_RESOURCE;   // (선택) HLSL 샘플링
    texDesc.CPUAccessFlags = 0;
    // --- 외부 메모리 Import(API2)를 쓸 거면 아래 추가:
    // texDesc.MiscFlags        = D3D11_RESOURCE_MISC_SHARED     // 공유 핸들
    //                          | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;

    hr = g_pd3dDevice->CreateTexture2D(&texDesc, nullptr, &myTex);
    if (FAILED(hr)) {
        // 에러 처리
    }

    // 2) UAV 뷰 생성
    D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = texDesc.Format;
    uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    uavDesc.Texture2D.MipSlice = 0;

    hr = g_pd3dDevice->CreateUnorderedAccessView(myTex, &uavDesc, &myUAV);
    if (FAILED(hr)) {
        // 에러 처리
    }

    return S_OK;
}

//----------------------------------------------
// CreateCudaSurf & Graphics Interlop
//----------------------------------------------
void CreateCudaSurf() {
    // 0) D3D11 텍스처에 CUDA용 플래그로 바인딩(UAV)  
    //    ID3D11Texture2D* myTex;   // 이미 생성된 텍스처

    // 1) 리소스 등록
    checkCudaErrors(cudaGraphicsD3D11RegisterResource(
        &cudaRes,
        myTex,
        cudaGraphicsRegisterFlagsSurfaceLoadStore  // surf2Dread/surf2Dwrite 용
    ));

    // 2) 매핑
    checkCudaErrors(cudaGraphicsMapResources(1, &cudaRes, cuda_stream));
    //cudaError_t err = cudaGraphicsMapResources(1, &cudaRes, cuda_stream);
    //if (err != cudaSuccess) {
    //    std::cerr << "MapResources failed: "
    //        << cudaGetErrorString(err) << "\n";
    //    return;
    //}

    // 3) Mip-level 0
    cudaArray_t cuArray = nullptr;
    unsigned mipLevel = 0; // 단일 레벨만 쓴다면 0
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(
        &cuArray, cudaRes, /*arrayIndex=*/0, mipLevel
    ));

    // 4) Surface Object 생성
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    checkCudaErrors(cudaCreateSurfaceObject(&dstSurfMipMap0, &resDesc));

    // 5) Unmap
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaRes, cuda_stream));
}

////////////////////////////////////////////////////////////////////////////////
//! Draw the final result on the screen
////////////////////////////////////////////////////////////////////////////////
bool DrawScene()
{

    HRESULT hr = S_OK;

    // 1) 백버퍼 얻기
    ID3D11Texture2D* pBackBuffer = nullptr;
    hr = g_pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&pBackBuffer);
    AssertOrQuit(SUCCEEDED(hr));
 
    // 2) Compute 결과 텍스처를 백버퍼로 복사
    //    (myTex 는 cudaGraphicsD3D11RegisterResource 로 등록한 텍스처)
    g_pd3dDeviceContext->CopyResource(pBackBuffer, myTex);

    // 3) Present
    pBackBuffer->Release();
    g_pSwapChain->Present(0, 0);

    return true;
}

//-----------------------------------------------------------------------------
// Name: Cleanup()
// Desc: Releases all previously initialized objects
//-----------------------------------------------------------------------------
void Cleanup()
{
    //
    // clean up Direct3D
    //

    if (g_pSwapChainRTV != NULL) {
        g_pSwapChainRTV->Release();
    }

    if (g_pSwapChain != NULL) {
        g_pSwapChain->Release();
    }

    if (g_pd3dDevice != NULL) {
        g_pd3dDevice->Release();
    }

    checkCudaErrors(cudaDestroySurfaceObject(dstSurfMipMap0));
    checkCudaErrors(cudaGraphicsUnregisterResource(cudaRes));
}

//-----------------------------------------------------------------------------
// Name: UpdateRot()
// Desc: Update camera rotation
//-----------------------------------------------------------------------------
void UpdateRot()
{
    // init
    static int count = 0;
    if (count == 0)
    {
        lookat = glm::vec3{ 0.0, 0.0, 0.0 };
        lookfrom = glm::vec3{ 13.0, 2.0, -3.0 };
        glm::vec3 firstCameraDir = glm::normalize(lookat - lookfrom);

        g_pitch = asinf(firstCameraDir.y);
        g_yaw = atan2f(firstCameraDir.x, firstCameraDir.z);
        count++;
    }

    static const float sensitivity = 0.0005f;
    g_yaw += dx * sensitivity;
    g_pitch -= dy * sensitivity;

    // forward 방향 계산 (unit length)
    float cosP = cosf(g_pitch);
    glm::vec3 forward{
        cosP * sinf(g_yaw),   // x
        sinf(g_pitch),        // y
        cosP * cosf(g_yaw)    // z
    };

    // lookat 갱신
    lookat = lookfrom + forward;
}

//-----------------------------------------------------------------------------
// Name: UpdatePan()
// Desc: Update camera pan
//-----------------------------------------------------------------------------
void UpdatePan()
{
    const float panSpeed = 0.001f * glm::length(lookfrom - lookat); // (화면당 world 단위)

    glm::vec3 forward = glm::normalize(lookat - lookfrom);
    glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
    glm::vec3 up = glm::cross(right, forward);

    glm::vec3 translation =
        -right * (dx * panSpeed)   // 마우스 오른쪽 → world 왼쪽 이동
        + up * (dy * panSpeed);  // 마우스 아래 → world 위 이동

    lookat += translation;
    lookfrom += translation;
}

//-----------------------------------------------------------------------------
// Name: Render()
// Desc: Launches the CUDA kernels to fill the surface object <=> UAV texture2D
//-----------------------------------------------------------------------------
void Render()
{
    checkCudaErrors(cudaGraphicsMapResources(1, &cudaRes, cuda_stream));

    // Launch cuda kernel to generate sinewave in vertex buffer
    LaunchRaytracingKernel(
        dstSurfMipMap0, g_WindowWidth, g_WindowHeight, 
        lookfrom, lookat, g_fov,
        cuda_stream);
   
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaRes, cuda_stream));

    // Draw the scene using them
    DrawScene();
}

//-----------------------------------------------------------------------------
// Name: MsgProc()
// Desc: The window's message handler
//-----------------------------------------------------------------------------
static LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg) {
    case WM_LBUTTONDOWN:
        g_isDragging = true;
        g_prevMousePos = { LOWORD(lParam), HIWORD(lParam) };
        SetCapture(hWnd);
        break;

    case WM_LBUTTONUP:
        g_isDragging = false;
        dirtyflagRot = false;
        ReleaseCapture();
        break;

    case WM_RBUTTONDOWN:
        g_isPanning = true;
        SetCapture(hWnd);
        g_prevMousePos = { LOWORD(lParam), HIWORD(lParam) };
        break;

    case WM_RBUTTONUP:
        g_isPanning = false;
        dirtyflagPan = false;
        ReleaseCapture();
        break;

    case WM_MOUSEMOVE:
    {
        POINT curMousePos = { LOWORD(lParam), HIWORD(lParam) };
        dx = curMousePos.x - g_prevMousePos.x;
        dy = curMousePos.y - g_prevMousePos.y;
        g_prevMousePos = curMousePos;

        if (g_isDragging)
        {
            dirtyflagRot = true;
        }
        else if (g_isPanning)
        {
            dirtyflagPan = true;
        }

    }
    break;

    case WM_MOUSEWHEEL:
    {
        short delta = GET_WHEEL_DELTA_WPARAM(wParam);
        const float fovSpeed = 1.0f;
        g_fov -= delta / WHEEL_DELTA * fovSpeed;
        g_fov = glm::clamp(g_fov, 5.0f, 100.0f);
    }
        break;

    case WM_KEYDOWN:
        if (wParam == VK_ESCAPE) {
            g_bDone = true;
            Cleanup();
            PostQuitMessage(0);
            return 0;
        }
        break;

    case WM_DESTROY:
        g_bDone = true;
        Cleanup();
        PostQuitMessage(0);
        return 0;

    case WM_PAINT:
        ValidateRect(hWnd, NULL);
        return 0;
    }

    return DefWindowProc(hWnd, msg, wParam, lParam);
}