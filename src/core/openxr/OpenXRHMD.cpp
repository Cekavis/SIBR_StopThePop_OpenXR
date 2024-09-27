// Software Name : SIBR_core
// SPDX-FileCopyrightText: Copyright (c) 2023 Orange
// SPDX-License-Identifier: Apache 2.0
//
// This software is distributed under the Apache 2.0 License;
// see the LICENSE file for more details.
//
// Author: Cédric CHEDALEUX <cedric.chedaleux@orange.com> et al.

#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <string>
#include <chrono>
#include <cstring>
#include <thread>

#include <core/openxr/OpenXRHelper.hpp>
#include <core/openxr/OpenXRHMD.hpp>
#include <core/system/Config.hpp>

namespace sibr
{

    // functions belonging to extensions must be loaded with xrGetInstanceProcAddr before use
    static PFN_xrGetOpenGLGraphicsRequirementsKHR pfnGetOpenGLGraphicsRequirementsKHR = NULL;
#if defined(XR_USE_TIMESPEC)
    static PFN_xrConvertTimespecTimeToTimeKHR pfnConvertTimespecTimeToTimeKHR = NULL;
#elif defined(XR_USE_PLATFORM_WIN32)
    static PFN_xrConvertWin32PerformanceCounterToTimeKHR pfnConvertWin32PerformanceCounterToTimeKHR = NULL;
#endif

    static bool
    load_extension_function_pointers(XrInstance instance)
    {
        XrResult result =
            xrGetInstanceProcAddr(instance, "xrGetOpenGLGraphicsRequirementsKHR",
                                  (PFN_xrVoidFunction *)&pfnGetOpenGLGraphicsRequirementsKHR);
        if (!xrCheck(instance, result, "Failed to get OpenGL graphics requirements function!"))
            return false;
#if defined(XR_USE_TIMESPEC)
        result =
            xrGetInstanceProcAddr(instance, "xrConvertTimespecTimeToTimeKHR",
                                  (PFN_xrVoidFunction *)&pfnConvertTimespecTimeToTimeKHR);
        if (!xrCheck(instance, result, "Failed to get OpenXR convert time function!"))
            return false;
#elif defined(XR_USE_PLATFORM_WIN32)
        result =
            xrGetInstanceProcAddr(instance, "xrConvertWin32PerformanceCounterToTimeKHR",
                                  (PFN_xrVoidFunction *)&pfnConvertWin32PerformanceCounterToTimeKHR);
        if (!xrCheck(instance, result, "Failed to get win32 time conversion function!"))
            return false;
#endif

        return true;
    }

    static XrPosef createIdentityPose() {
        XrPosef pose;
        pose.orientation = XrQuaternionf{ 0, 0, 0, 1.0 };
        pose.position = XrVector3f{ 0.f, 0.f, 0.f };
        return pose;
    }
    // we need an identity pose for creating spaces without offsets
    static XrPosef identity_pose = createIdentityPose();

// See https://www.khronos.org/opengl/wiki/Load_OpenGL_Functions
#ifdef _WIN32
    void *GetAnyGLFuncAddress(const char *name)
    {
        void *p = (void *)wglGetProcAddress(name);
        if (p == 0 ||
            (p == (void *)0x1) || (p == (void *)0x2) || (p == (void *)0x3) ||
            (p == (void *)-1))
        {
            HMODULE module = LoadLibraryA("opengl32.dll");
            p = (void *)GetProcAddress(module, name);
        }

        return p;
    }
#endif

    OpenXRHMD::OpenXRHMD(const std::string &applicationName, bool seated) : m_applicationName(applicationName),
                                                                            m_playSpaceType(seated ? XR_REFERENCE_SPACE_TYPE_LOCAL : XR_REFERENCE_SPACE_TYPE_STAGE)
    {
    }

    OpenXRHMD::~OpenXRHMD()
    {
        closeSession();
        terminate();
    }

    const OpenXRHMD::FrameRefreshReport &OpenXRHMD::getRefreshReport() const
    {
        return m_lastFrameRefreshReport;
    }

    Eigen::Vector3f OpenXRHMD::quaternionToEulerAngles(const XrQuaternionf &q)
    {
        // roll (x-axis rotation)
        double sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
        double cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y);
        float roll = std::atan2(sinr_cosp, cosr_cosp);

        // pitch (y-axis rotation)
        double sinp = std::sqrt(1 + 2 * (q.w * q.y - q.x * q.z));
        double cosp = std::sqrt(1 - 2 * (q.w * q.y - q.x * q.z));
        float pitch = 2 * std::atan2(sinp, cosp) - M_PI / 2;

        // yaw (z-axis rotation)
        double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
        double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
        float yaw = std::atan2(siny_cosp, cosy_cosp);

        return Vector3f(roll, pitch, yaw);
    }

    XrQuaternionf OpenXRHMD::eulerAnglestoQuaternion(float roll, float pitch, float yaw) // roll (x), pitch (Y), yaw (z)
    {
        // Abbreviations for the various angular functions

        float cr = cos(roll * 0.5);
        float sr = sin(roll * 0.5);
        float cp = cos(pitch * 0.5);
        float sp = sin(pitch * 0.5);
        float cy = cos(yaw * 0.5);
        float sy = sin(yaw * 0.5);

        return XrQuaternionf{
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy};
    }

    Eigen::Vector3f OpenXRHMD::getPoseOrientation(OpenXRHMD::Eye eye, AngleUnit unit) const
    {
        uint32_t viewIndex = eyeToViewIndex(eye);
        if (viewIndex >= m_viewCount)
        {
            fprintf(stderr, "View for %s eye does not exist\n", eyeToString(eye));
            return Vector3f::Zero();
        }

        // XrPoseOrientation is a unit quaternion, convert it to euler angles
        Vector3f orientation = quaternionToEulerAngles(views[viewIndex].pose.orientation);
        return (unit == AngleUnit::RADIAN ? orientation : radianToDegree(orientation));
    }

    Eigen::Quaternionf OpenXRHMD::getPoseQuaternion(OpenXRHMD::Eye eye) const
    {
        uint32_t viewIndex = eyeToViewIndex(eye);
        if (viewIndex >= m_viewCount)
        {
            fprintf(stderr, "View for %s eye does not exist\n", eyeToString(eye));
            return Quaternionf();
        }

        // XrPoseOrientation is a unit quaternion
        return Quaternionf{views[viewIndex].pose.orientation.w, views[viewIndex].pose.orientation.x, views[viewIndex].pose.orientation.y, views[viewIndex].pose.orientation.z};
    }

    Eigen::Vector3f OpenXRHMD::getPosePosition(OpenXRHMD::Eye eye) const
    {
        uint32_t viewIndex = eyeToViewIndex(eye);
        if (viewIndex >= m_viewCount)
        {
            fprintf(stderr, "View for %s eye does not exist\n", eyeToString(eye));
            return Vector3f::Zero();
        }

        // XrPosePosition are in world coordinates
        return Vector3f(views[viewIndex].pose.position.x,
                        views[viewIndex].pose.position.y,
                        views[viewIndex].pose.position.z);
    }

    Eigen::Vector4f OpenXRHMD::getFieldOfView(OpenXRHMD::Eye eye, AngleUnit unit) const
    {
        uint32_t viewIndex = eyeToViewIndex(eye);
        if (viewIndex >= m_viewCount)
        {
            fprintf(stderr, "View for %s eye does not exist\n", eyeToString(eye));
            return Vector4f::Zero();
        }

        // Fov angles are in radians
        Vector4f fov(
            views[viewIndex].fov.angleLeft,
            views[viewIndex].fov.angleRight,
            views[viewIndex].fov.angleDown,
            views[viewIndex].fov.angleUp);
        return (unit == AngleUnit::RADIAN ? fov : radianToDegree(fov));
    }

    Eigen::Vector2f OpenXRHMD::getHVFieldOfView(AngleUnit unit) const
    {
        if (m_viewCount < 2)
        {
            fprintf(stderr, "No view exists\n");
            return Vector2f::Zero();
        }

        // Compute vertical and horizontal fov
        uint32_t viewIndex = eyeToViewIndex(Eye::LEFT);
        Vector2f hvFov(
            views[viewIndex].fov.angleRight - views[viewIndex].fov.angleLeft,
            views[viewIndex].fov.angleUp - views[viewIndex].fov.angleDown);

        // Fov angles are in radians
        return (unit == AngleUnit::RADIAN ? hvFov : radianToDegree(hvFov));
    }

    Eigen::Vector2f OpenXRHMD::getScreenCenter(Eye eye) const
    {
        uint32_t viewIndex = eyeToViewIndex(eye);
        if (viewIndex >= m_viewCount)
        {
            fprintf(stderr, "View for %s eye does not exist\n", eyeToString(eye));
            return Vector2f::Zero();
        }

        // Fov angles are in radians
        float tanLeft = tan(abs(views[viewIndex].fov.angleLeft));
        float tanRight = tan(abs(views[viewIndex].fov.angleRight));
        float tanUp = tan(abs(views[viewIndex].fov.angleUp));
        float tanDown = tan(abs(views[viewIndex].fov.angleDown));
        float centerX = tanLeft / (tanLeft + tanRight);
        float centerY = tanDown / (tanDown + tanUp);
        return Vector2f(centerX, centerY);
    }

    bool OpenXRHMD::init()
    {
        XrResult result = XR_SUCCESS;

        if (m_printApiLayers)
            printApiLayers();

        // xrEnumerate*() functions are usually called once with CapacityInput = 0.
        // The function will write the required amount into CountOutput. We then have
        // to allocate an array to hold CountOutput elements and call the function
        // with CountOutput as CapacityInput.
        uint32_t ext_count = 0;
        result = xrEnumerateInstanceExtensionProperties(NULL, 0, &ext_count, NULL);

        /* TODO: instance null will not be able to convert XrResult to string */
        if (!xrCheck(NULL, result, "Failed to enumerate number of extension properties"))
            return false;

        XrExtensionProperties *ext_props = (XrExtensionProperties *)malloc(sizeof(XrExtensionProperties) * ext_count);
        for (uint16_t i = 0; i < ext_count; i++)
        {
            // we usually have to fill in the type (for validation) and set
            // next to NULL (or a pointer to an extension specific struct)
            ext_props[i].type = XR_TYPE_EXTENSION_PROPERTIES;
            ext_props[i].next = NULL;
        }

        std::vector<const char *> expectedExtensions = {
            XR_KHR_VISIBILITY_MASK_EXTENSION_NAME,
            XR_KHR_OPENGL_ENABLE_EXTENSION_NAME,
#if defined(XR_USE_PLATFORM_WIN32)
            XR_KHR_WIN32_CONVERT_PERFORMANCE_COUNTER_TIME_EXTENSION_NAME,
#elif defined(XR_USE_TIMESPEC)
            XR_KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME,
#endif
        };

        result = xrEnumerateInstanceExtensionProperties(NULL, ext_count, &ext_count, ext_props);
        if (!xrCheck(NULL, result, "Failed to enumerate extension properties"))
            return false;

        bool opengl_supported = false;

        if (m_printRuntimeExtensions)
            SIBR_LOG << "Runtime supports " << ext_count << " extensions" << std::endl;
        for (uint32_t i = 0; i < ext_count; i++)
        {
            if (m_printRuntimeExtensions)
                SIBR_LOG << "\t" << ext_props[i].extensionName << " v" << ext_props[i].extensionVersion << std::endl;
            if (strcmp(XR_KHR_OPENGL_ENABLE_EXTENSION_NAME, ext_props[i].extensionName) == 0)
            {
                opengl_supported = true;
            }
        }
        free(ext_props);

        // A graphics extension like OpenGL is required to draw anything in VR
        if (!opengl_supported)
        {
            printf("Runtime does not support OpenGL extension!\n");
            return false;
        }

        if (m_printRuntimeExtensions)
        {
            SIBR_LOG << "Enable following extensions:" << std::endl;
            for (auto ext : expectedExtensions)
            {
                SIBR_LOG << "\t" << ext << std::endl;
            }
        }

        // Create XrInstance
        XrInstanceCreateInfo instance_create_info;
        instance_create_info.type = XR_TYPE_INSTANCE_CREATE_INFO;
        instance_create_info.next = NULL;
        instance_create_info.createFlags = 0;

        XrApplicationInfo applicationInfo;
        applicationInfo.applicationVersion = 1;
        applicationInfo.engineVersion = 0;
        applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
        instance_create_info.applicationInfo = applicationInfo;

        instance_create_info.enabledApiLayerCount = 0;
        instance_create_info.enabledApiLayerNames = NULL;
        instance_create_info.enabledExtensionCount = (uint32_t)expectedExtensions.size();
        instance_create_info.enabledExtensionNames = expectedExtensions.data();
        strncpy(instance_create_info.applicationInfo.applicationName, m_applicationName.c_str(),
                m_applicationName.length());
        strncpy(instance_create_info.applicationInfo.engineName, "SIBR_core", XR_MAX_ENGINE_NAME_SIZE);

        result = xrCreateInstance(&instance_create_info, &m_instance);
        if (!xrCheck(NULL, result, "Failed to create XR m_instance."))
            return false;

        if (!load_extension_function_pointers(m_instance))
            return false;

        // Get runtime name and version
        if (!getRuntimeNameAndVersion(m_instance, m_runtimeName, m_runtimeVersion))
            SIBR_LOG << "Unable to retrieve OpenXR runtime name and version" << std::endl;

        // --- Get XrSystemId
        XrSystemGetInfo system_get_info;
        system_get_info.type = XR_TYPE_SYSTEM_GET_INFO;
        system_get_info.next = NULL;
        system_get_info.formFactor = m_formFactor;

        result = xrGetSystem(m_instance, &system_get_info, &m_systemId);
        if (!xrCheck(m_instance, result, "Failed to get system for HMD form factor."))
            return false;

        if (m_printSystemProperties)
            printSystemProperties(m_instance, m_systemId);

        result = xrEnumerateViewConfigurationViews(m_instance, m_systemId, m_viewType, 0, &m_viewCount, NULL);
        if (!xrCheck(m_instance, result, "Failed to get view configuration view count!"))
            return false;

        m_viewConfigViews = (XrViewConfigurationView *)malloc(sizeof(XrViewConfigurationView) * m_viewCount);
        for (uint32_t i = 0; i < m_viewCount; i++)
        {
            m_viewConfigViews[i].type = XR_TYPE_VIEW_CONFIGURATION_VIEW;
            m_viewConfigViews[i].next = NULL;
        }

        result = xrEnumerateViewConfigurationViews(m_instance, m_systemId, m_viewType, m_viewCount,
                                                   &m_viewCount, m_viewConfigViews);
        if (!xrCheck(m_instance, result, "Failed to enumerate view configuration views!"))
            return false;

        if (m_printViewConfigInfos)
            printViewconfigViewInfo(m_viewCount, m_viewConfigViews);

        // Now we have the recommended resolution for the headset, set the output resolution
        m_resolution = getRecommendedResolution();

        // OpenXR requires checking graphics requirements before creating a m_session.
        XrGraphicsRequirementsOpenGLKHR opengl_reqs;
        opengl_reqs.type = XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_KHR;
        opengl_reqs.next = NULL;

        // this function pointer was loaded with xrGetInstanceProcAddr
        result = pfnGetOpenGLGraphicsRequirementsKHR(m_instance, m_systemId, &opengl_reqs);
        if (!xrCheck(m_instance, result, "Failed to get OpenGL graphics requirements!"))
            return false;

        return true;
    }

    Vector2i OpenXRHMD::getRecommendedResolution() const
    {
        return (m_viewCount > 0 ? Vector2i(m_viewConfigViews[0].recommendedImageRectWidth, m_viewConfigViews[0].recommendedImageRectHeight) : Vector2i::Zero());
    }

    void OpenXRHMD::setResolution(const Eigen::Vector2i &resolution)
    {
        if (!isSessionRunning())
        {
            m_resolution = resolution;
        }
        else
        {
            SIBR_WRG << "Cannot change the resolution: XR session is already running." << std::endl;
        }
    }

    const Eigen::Vector2i &OpenXRHMD::getResolution() const
    {
        return m_resolution;
    }

    template <typename T>
    bool OpenXRHMD::startSession(T graphicsBindingGL)
    {
        // Initialize the OpenGL stack to be able to render textures for each eye
        // and submit the OpenGL framebuffer to OpenXR
        GLenum err = glewInit();
        if (GLEW_OK != err)
        {
            printf("Error initializing OpenGL: %u\n", err);
            return false;
        }

        SIBR_LOG << "Starting XR session: OpenGL version = " << glGetString(GL_VERSION) << ", renderer = " << glGetString(GL_RENDERER) << std::endl;

        XrSessionCreateInfo m_sessionCreateInfo;
        m_sessionCreateInfo.type = XR_TYPE_SESSION_CREATE_INFO;
        m_sessionCreateInfo.next = &graphicsBindingGL;
        m_sessionCreateInfo.systemId = m_systemId;

        return createSession(m_sessionCreateInfo) && createReferenceSpace() && createSwapchain() && synchronizeSession();
    }

#ifdef XR_USE_PLATFORM_XLIB
    template bool OpenXRHMD::startSession<XrGraphicsBindingOpenGLXlibKHR>(XrGraphicsBindingOpenGLXlibKHR);
#endif
#ifdef XR_USE_PLATFORM_WIN32
    template bool OpenXRHMD::startSession<XrGraphicsBindingOpenGLWin32KHR>(XrGraphicsBindingOpenGLWin32KHR);
#endif

    bool OpenXRHMD::createSession(const XrSessionCreateInfo &m_sessionCreateInfo)
    {
        XrResult result = xrCreateSession(m_instance, &m_sessionCreateInfo, &m_session);
        if (!xrCheck(m_instance, result, "Failed to create session"))
            return false;
        return true;
    }

    bool OpenXRHMD::createReferenceSpace()
    {
        /* Many runtimes support at least STAGE and LOCAL but not all do.
         * Sophisticated apps might check with xrEnumerateReferenceSpaces() if the
         * chosen one is supported and try another one if not.
         * Here we will get an error from xrCreateReferenceSpace() and exit. */
        XrReferenceSpaceCreateInfo m_playSpace_create_info;
        m_playSpace_create_info.type = XR_TYPE_REFERENCE_SPACE_CREATE_INFO;
        m_playSpace_create_info.next = NULL;
        m_playSpace_create_info.referenceSpaceType = m_playSpaceType;
        m_playSpace_create_info.poseInReferenceSpace = identity_pose;

        XrResult result = xrCreateReferenceSpace(m_session, &m_playSpace_create_info, &m_playSpace);
        if (!xrCheck(m_instance, result, "Failed to create play space!"))
            return false;

        return true;
    }

    bool OpenXRHMD::createSwapchain()
    {
        // Create Swapchains
        uint32_t swapchain_format_count;
        XrResult result = xrEnumerateSwapchainFormats(m_session, 0, &swapchain_format_count, NULL);
        if (!xrCheck(m_instance, result, "Failed to get number of supported swapchain formats"))
            return false;

        int64_t *swapchain_formats = new int64_t[swapchain_format_count];
        result = xrEnumerateSwapchainFormats(m_session, swapchain_format_count, &swapchain_format_count,
                                             swapchain_formats);
        if (!xrCheck(m_instance, result, "Failed to enumerate swapchain formats"))
            return false;

        // Select swapchain with SRGB format
        int64_t color_format = selectSwapchainFormat(m_instance, m_session, GL_SRGB8_ALPHA8_EXT, true);

        // Create swapchain for main VR rendering
        {
            // We request the runtime here to create a swapchain for each view. The swapchain
            // attributes define the size (width, height) of the textures that application should render to
            // as well as number of texture to perform single, double or triple buffering
            // Here, we use values (resolution, sample count) recommended by the runtime
            m_swapchains = (XrSwapchain_T **)malloc(sizeof(XrSwapchain) * m_viewCount);
            m_swapchainsLengths = (uint32_t *)malloc(sizeof(uint32_t) * m_viewCount);
            m_swapchainsImages = (XrSwapchainImageOpenGLKHR **)malloc(sizeof(XrSwapchainImageOpenGLKHR *) * m_viewCount);
            for (uint32_t i = 0; i < m_viewCount; i++)
            {
                XrSwapchainCreateInfo swapchain_create_info;
				swapchain_create_info.type = XR_TYPE_SWAPCHAIN_CREATE_INFO;
				swapchain_create_info.next = NULL;
				swapchain_create_info.createFlags = 0;
				swapchain_create_info.usageFlags = XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
				swapchain_create_info.format = color_format;
				swapchain_create_info.sampleCount = m_viewConfigViews[i].recommendedSwapchainSampleCount;
				swapchain_create_info.width = (uint32_t)m_resolution.x();
				swapchain_create_info.height = (uint32_t)m_resolution.y();
				swapchain_create_info.faceCount = 1;
				swapchain_create_info.arraySize = 1;
				swapchain_create_info.mipCount = 1;

                result = xrCreateSwapchain(m_session, &swapchain_create_info, &m_swapchains[i]);
                if (!xrCheck(m_instance, result, "Failed to create swapchain %d!", i))
                    return false;

                // The runtime controls how many textures we have to be able to render to
                result = xrEnumerateSwapchainImages(m_swapchains[i], 0, &m_swapchainsLengths[i], NULL);
                if (!xrCheck(m_instance, result, "Failed to enumerate m_swapchains"))
                    return false;

                m_swapchainsImages[i] = (XrSwapchainImageOpenGLKHR *)malloc(sizeof(XrSwapchainImageOpenGLKHR) * m_swapchainsLengths[i]);
                for (uint32_t j = 0; j < m_swapchainsLengths[i]; j++)
                {
                    m_swapchainsImages[i][j].type = XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_KHR;
                    m_swapchainsImages[i][j].next = NULL;
                }
                result =
                    xrEnumerateSwapchainImages(m_swapchains[i], m_swapchainsLengths[i], &m_swapchainsLengths[i],
                                               (XrSwapchainImageBaseHeader *)m_swapchainsImages[i]);
                if (!xrCheck(m_instance, result, "Failed to enumerate swapchain images"))
                    return false;
            }
        }

        // Do not allocate these every frame to save some resources
        views = (XrView *)malloc(sizeof(XrView) * m_viewCount);
        for (uint32_t i = 0; i < m_viewCount; i++)
        {
            views[i].type = XR_TYPE_VIEW;
            views[i].next = NULL;
        }

        m_projectionViews = (XrCompositionLayerProjectionView *)malloc(
            sizeof(XrCompositionLayerProjectionView) * m_viewCount);
        for (uint32_t i = 0; i < m_viewCount; i++)
        {
            m_projectionViews[i].type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW;
            m_projectionViews[i].next = NULL;

            m_projectionViews[i].subImage.swapchain = m_swapchains[i];
            m_projectionViews[i].subImage.imageArrayIndex = 0;
            m_projectionViews[i].subImage.imageRect.offset.x = 0;
            m_projectionViews[i].subImage.imageRect.offset.y = 0;
            m_projectionViews[i].subImage.imageRect.extent.width = m_resolution.x();
            m_projectionViews[i].subImage.imageRect.extent.height = m_resolution.y();
        };

        return true;
    }

    bool OpenXRHMD::synchronizeSession()
    {
        // Keep polling events until we successfully synchronized with the headset
        while (m_status != SessionStatus::SYNCHRONIZED)
        {
            pollEvents();

            // Damn, we got an error!
            if (m_status == SessionStatus::FAILURE)
            {
                return false;
            }
        }

        return true;
    }

    bool OpenXRHMD::pollEvents()
    {
        // Handle runtime Events
        XrEventDataBuffer runtime_event;
        runtime_event.type = XR_TYPE_EVENT_DATA_BUFFER;
        runtime_event.next = NULL;
        XrResult poll_result = xrPollEvent(m_instance, &runtime_event);
        if (poll_result == XR_SUCCESS)
        {
            switch (runtime_event.type)
            {
            case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING:
            {
                XrEventDataInstanceLossPending *event = (XrEventDataInstanceLossPending *)&runtime_event;
                SIBR_WRG << "EVENT: instance loss pending at " << event->lossTime << "! Destroying instance." << std::endl;
                XrResult result = xrDestroyInstance(m_instance);
                if (!xrCheck(NULL, result, "Failed to destroy XR instance."))
                {
                    m_status = SessionStatus::FAILURE;
                }
            }
            case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED:
            {
                XrEventDataSessionStateChanged *event = (XrEventDataSessionStateChanged *)&runtime_event;
                updateCurrentSessionState(event->state);
            }
            case XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED:
            {
                // Do not handle it for now
                break;
            }
            default:
                SIBR_LOG << "EVENT: Unhandled event (type " << runtime_event.type << ")" << std::endl;
            }
        }
        else if (poll_result == XR_EVENT_UNAVAILABLE)
        {
            // No new events
        }
        else
        {
            SIBR_WRG << "EVENT: Failed to poll XR Runtime events!" << std::endl;
        }

        // In case of transitory states, keep polling
        bool keepPolling = m_status == SessionStatus::BEGINNING ||
                           m_status == SessionStatus::ENDING;
        if (m_status == SessionStatus::BEGINNING)
        {
            // Ok, we initiated the m_session, the runtime is now awaiting for a first frame loop
            // synchronization (see https://registry.khronos.org/OpenXR/specs/1.0/man/html/XrSessionState.html).
            // Let's call xrWaitFrame, xrBeginFrame and xrEndFrame to trigger the state change to
            // SYNCHRONIZED state. Let's submitting an empty frame.
            // Hopefully, we will get the first head-pose information
            waitNextFrame();
            submitFrame();
        }

        if (keepPolling)
        {
            return pollEvents();
        }

        return m_status != SessionStatus::FAILURE;
    }

    void OpenXRHMD::setIdleAppCallback(const std::function<void()> &callback)
    {
        m_idleCallback = callback;
    }

    void OpenXRHMD::setVisibleAppCallback(const std::function<void()> &callback)
    {
        m_visibleCallback = callback;
    }

    void OpenXRHMD::setFocusedAppCallback(const std::function<void()> &callback)
    {
        m_focusedCallback = callback;
    }

    bool OpenXRHMD::isSessionRunning() const
    {
        return m_status != SessionStatus::FAILURE && m_status != SessionStatus::STOPPED;
    }

    void OpenXRHMD::updateCurrentSessionState(const XrSessionState &state)
    {
        SIBR_LOG << "XR session state change: '" << sessionStateToString(m_currentState) << "' -> '" << sessionStateToString(state) << "'" << std::endl;
        m_currentState = state;

        /*
         * React to session state changes, see OpenXR spec 9.3 diagram. What we need to react to:
         *
         * * READY -> xrBeginSession STOPPING -> xrEndSession (note that the same session can be restarted)
         * * EXITING -> xrDestroySession (EXITING only happens after we went through STOPPING and called xrEndSession)
         * * IDLE -> don't run render loop, but keep polling for events
         * * SYNCHRONIZED, VISIBLE, FOCUSED -> run render loop
         */
        switch (m_currentState)
        {
        case XR_SESSION_STATE_MAX_ENUM: // must be a bug
        case XR_SESSION_STATE_UNKNOWN:
            break; // state handling switch

        // normal state
        case XR_SESSION_STATE_IDLE:
        {
            m_status = SessionStatus::IDLE;
            if (m_idleCallback)
            {
                m_idleCallback();
            }
            break; // state handling switch
        }
        case XR_SESSION_STATE_SYNCHRONIZED:
        {
            // Successful session start
            m_status = SessionStatus::SYNCHRONIZED;
            break; // state handling switch
        }
        case XR_SESSION_STATE_FOCUSED:
        {
            if (m_focusedCallback)
            {
                m_focusedCallback();
            }
            break; // state handling switch
        }
        case XR_SESSION_STATE_VISIBLE:
        {
            if (m_visibleCallback)
            {
                m_visibleCallback();
            }
            break; // state handling switch
        }

        // Session state is ready, let's request for beginning the m_session
        case XR_SESSION_STATE_READY:
        {
            // start session only if it is not starting, i.e. not when we already called xrBeginSession
            // but the runtime did not switch to the next state yet
            if (m_status != SessionStatus::BEGINNING)
            {
                XrSessionBeginInfo m_session_begin_info;
                m_session_begin_info.type = XR_TYPE_SESSION_BEGIN_INFO;
                m_session_begin_info.next = NULL;
                m_session_begin_info.primaryViewConfigurationType = m_viewType;
                XrResult result = xrBeginSession(m_session, &m_session_begin_info);
                if (!xrCheck(m_instance, result, "Failed to begin ession!"))
                {
                    m_status = SessionStatus::FAILURE;
                    return;
                }
                m_status = SessionStatus::BEGINNING;
            }
            break; // state handling switch
        }
        case XR_SESSION_STATE_STOPPING:
        {
            // end session only if it is running, i.e. not when we already called xrEndSession but the
            // runtime did not switch to the next state yet
            if (m_status != SessionStatus::ENDING)
            {
                XrResult result = xrEndSession(m_session);
                if (!xrCheck(m_instance, result, "Failed to end session!"))
                {
                    m_status = SessionStatus::FAILURE;
                    return;
                }
                m_status = SessionStatus::ENDING;
                // On sucess, session will go to STATE_IDLE
            }
            break; // state handling switch
        }
        // Destroy session
        case XR_SESSION_STATE_LOSS_PENDING: // The session is in the process of being lost
        case XR_SESSION_STATE_EXITING:
            XrResult result = xrDestroySession(m_session);
            if (!xrCheck(m_instance, result, "Failed to destroy session!"))
            {
                m_status = SessionStatus::FAILURE;
                return;
            }
            m_session = XR_NULL_HANDLE;
            m_status = SessionStatus::STOPPED;
            break; // state handling switch
        }
    }

    bool OpenXRHMD::waitNextFrame()
    {
        // Wait for head-pose move predication to render next frame
        m_lastFrameState.type = XR_TYPE_FRAME_STATE;
        m_lastFrameState.next = NULL;
        m_lastFrameState.predictedDisplayPeriod = 0;
        m_lastFrameState.predictedDisplayTime = 0;

        XrFrameWaitInfo frame_wait_info;
        frame_wait_info.type = XR_TYPE_FRAME_WAIT_INFO;
        frame_wait_info.next = NULL;
        XrResult result = xrWaitFrame(m_session, &frame_wait_info, &m_lastFrameState);
        if (!xrCheck(m_instance, result, "xrWaitFrame() failed"))
            return false;

        XrTime now_ns = 0;
#if defined(XR_USE_PLATFORM_WIN32)
        LARGE_INTEGER ticks;
        if (QueryPerformanceCounter(&ticks))
        {
            pfnConvertWin32PerformanceCounterToTimeKHR(m_instance, &ticks, &now_ns); // See https://registry.khronos.org/OpenXR/specs/1.0/man/html/xrConvertTimeToWin32PerformanceCounterKHR.html
        }
        else
        {
            SIBR_WRG << "Failed to get performance counter" << std::endl;
        }
#elif defined(XR_USE_TIMESPEC)
        struct timespec timespecTime;
        clock_gettime(CLOCK_MONOTONIC, &timespecTime);
        pfnConvertTimespecTimeToTimeKHR(m_instance, &timespecTime, &now_ns); // See https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#XR_KHR_convert_timespec_time
#endif

        // Get the new view locations
        XrViewLocateInfo view_locate_info;
        view_locate_info.type = XR_TYPE_VIEW_LOCATE_INFO;
        view_locate_info.next = NULL;
        view_locate_info.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
        view_locate_info.displayTime = m_lastFrameState.predictedDisplayTime;
        view_locate_info.space = m_playSpace;

        XrViewState view_state;
        view_state.type = XR_TYPE_VIEW_STATE;
        view_state.next = NULL;
        result = xrLocateViews(m_session, &view_locate_info, &view_state, m_viewCount, &m_viewCount, views);
        if (!xrCheck(m_instance, result, "Could not locate views"))
            return false;

        return true;
    }

    bool OpenXRHMD::closeSession()
    {
        // If session is already idle, just destroy it
        if (m_status == SessionStatus::IDLE)
        {
            XrResult result = xrDestroySession(m_session);
            if (!xrCheck(m_instance, result, "Failed to destroy session!"))
            {
                m_status = SessionStatus::FAILURE;
            }
            m_session = XR_NULL_HANDLE;
            m_status = SessionStatus::STOPPED;
        }
        // If not, proper close by requesting an exit
        else if (m_status != SessionStatus::STOPPED)
        {
            XrResult result = xrRequestExitSession(m_session);
            if (!xrCheck(m_instance, result, "Failed to request exit session!"))
            {
                m_status = SessionStatus::FAILURE;
                return false;
            }
            // Keep polling event until session is properly ended and destroyed
            while (isSessionRunning())
            {
                pollEvents();
            }
        }
        return m_status != SessionStatus::STOPPED;
    }

    bool OpenXRHMD::terminate()
    {
        if (isSessionRunning())
        {
            closeSession();
        }

        // Even if session failed to close, try to destroy the instance
        if (m_instance != XR_NULL_HANDLE)
        {
            XrResult result = xrDestroyInstance(m_instance);
            if (!xrCheck(m_instance, result, "Failed to destroy instance!"))
            {
                m_status = SessionStatus::FAILURE;
                return false;
            }
            m_instance = XR_NULL_HANDLE;
        }
        return true;
    }

    bool OpenXRHMD::shouldRender() const
    {
        return m_lastFrameState.shouldRender == XR_TRUE;
    }

    bool OpenXRHMD::submitFrame(RenderFunc renderFunc)
    {
        // Let's skip the rendering if we do not need to
        if (!shouldRender())
        {
            submitFrame();
            return true;
        }

        XrFrameBeginInfo frame_begin_info;
        frame_begin_info.type = XR_TYPE_FRAME_BEGIN_INFO;
        frame_begin_info.next = NULL;

        XrResult result = xrBeginFrame(m_session, &frame_begin_info);
        if (!xrCheck(m_instance, result, "failed to begin frame!"))
            return false;

        // Render each eye and fill projectionViews with the result
        for (uint32_t i = 0; i < m_viewCount; i++)
        {

            XrSwapchainImageAcquireInfo acquire_info;
            acquire_info.type = XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO;
            acquire_info.next = NULL;
            uint32_t acquired_index;
            result = xrAcquireSwapchainImage(m_swapchains[i], &acquire_info, &acquired_index);
            if (!xrCheck(m_instance, result, "failed to acquire swapchain image!"))
                break;

            XrSwapchainImageWaitInfo wait_info;
            wait_info.type = XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO;
            wait_info.next = NULL;
            wait_info.timeout = 1000;
            result = xrWaitSwapchainImage(m_swapchains[i], &wait_info);
            if (!xrCheck(m_instance, result, "failed to wait for swapchain image!"))
                break;

            // Update projection views
            m_projectionViews[i].pose = views[i].pose;
            m_projectionViews[i].fov = views[i].fov;

            // Let's call renderFunc to delegate the rendering to the calling function
            // renderFunc will render the scene into the framebuffer texture referenced in the view's SwapChainImage (one for each eye)
            renderFunc(i, m_swapchainsImages[i][acquired_index].image);

            XrSwapchainImageReleaseInfo release_info;
            release_info.type = XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO;
            release_info.next = NULL;
            result = xrReleaseSwapchainImage(m_swapchains[i], &release_info);
            if (!xrCheck(m_instance, result, "failed to release swapchain image!"))
                break;
        }

        XrCompositionLayerProjection projection_layer;
        projection_layer.type = XR_TYPE_COMPOSITION_LAYER_PROJECTION;
        projection_layer.next = NULL;
        projection_layer.layerFlags = 0;
        projection_layer.space = m_playSpace;
        projection_layer.viewCount = m_viewCount;
        projection_layer.views = m_projectionViews;

        uint32_t submitted_layer_count = 1;
        const XrCompositionLayerBaseHeader *submitted_layers[1] = {
            (const XrCompositionLayerBaseHeader *const)&projection_layer};

        // Check if the frame meets the deadline (aka predictedDisplayTime).
        updateRefreshReport();

        XrFrameEndInfo frameEndInfo;
        frameEndInfo.type = XR_TYPE_FRAME_END_INFO;
        frameEndInfo.next = NULL;
        frameEndInfo.displayTime = m_lastFrameState.predictedDisplayTime;
        frameEndInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
        frameEndInfo.layerCount = submitted_layer_count;
        frameEndInfo.layers = submitted_layers;

        result = xrEndFrame(m_session, &frameEndInfo);
        if (!xrCheck(m_instance, result, "failed to end frame!"))
            return false;

        return true;
    }

    bool OpenXRHMD::submitFrame()
    {
        XrFrameBeginInfo frame_begin_info;
        frame_begin_info.type = XR_TYPE_FRAME_BEGIN_INFO;
        frame_begin_info.next = NULL;

        XrResult result = xrBeginFrame(m_session, &frame_begin_info);
        if (!xrCheck(m_instance, result, "failed to begin frame!"))
            return false;

        XrFrameEndInfo frameEndInfo;
		frameEndInfo.type = XR_TYPE_FRAME_END_INFO;
		frameEndInfo.next = NULL;
		frameEndInfo.displayTime = m_lastFrameState.predictedDisplayTime;
		frameEndInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
		frameEndInfo.layerCount = 0;
		frameEndInfo.layers = NULL;
        result = xrEndFrame(m_session, &frameEndInfo);
        if (!xrCheck(m_instance, result, "failed to end frame!"))
            return false;

        return true;
    }

    void OpenXRHMD::updateRefreshReport()
    {
        XrTime now_ns = 0;
#if defined(XR_USE_PLATFORM_WIN32)
        LARGE_INTEGER ticks;
        if (QueryPerformanceCounter(&ticks))
        {
            pfnConvertWin32PerformanceCounterToTimeKHR(m_instance, &ticks, &now_ns); // See https://registry.khronos.org/OpenXR/specs/1.0/man/html/xrConvertTimeToWin32PerformanceCounterKHR.html
        }
        else
        {
            SIBR_WRG << "Failed to get performance counter" << std::endl;
        }
#elif defined(XR_USE_TIMESPEC)
        struct timespec timespecTime;
        clock_gettime(CLOCK_MONOTONIC, &timespecTime);
        pfnConvertTimespecTimeToTimeKHR(m_instance, &timespecTime, &now_ns); // See https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#XR_KHR_convert_timespec_time
#endif

        // Update current report
        if (m_lastFrameState.predictedDisplayTime - now_ns < 0)
        {
            m_currentFrameRefreshReport.nbMissedFrames++;
        }
        m_currentFrameRefreshReport.totalRenderedFrames++;
        m_currentFrameRefreshReport.expectedFramerate = 1000000000.f / m_lastFrameState.predictedDisplayPeriod;

        // Validate the report after 100 frames
        if (m_currentFrameRefreshReport.totalRenderedFrames == 100)
        {

            m_currentFrameRefreshReport.measuredFramerate = 1000.f * m_currentFrameRefreshReport.totalRenderedFrames /
                                                            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - m_currentFrameRefreshReport.firstFrameTimestamp).count();
            m_lastFrameRefreshReport = m_currentFrameRefreshReport;
            m_currentFrameRefreshReport = {0, 0, 0.f, 0.f, std::chrono::high_resolution_clock::now()};
        }
    }

    uint32_t OpenXRHMD::eyeToViewIndex(Eye eye)
    {
        return static_cast<uint32_t>(eye);
    }

    const char *OpenXRHMD::eyeToString(Eye eye)
    {
        switch (eye)
        {
        case Eye::LEFT:
            return "LEFT";
        case Eye::RIGHT:
            return "RIGHT";
        }
        return "UNKNOWN";
    }

    const char *OpenXRHMD::sessionStateToString(const XrSessionState &state)
    {
        switch (state)
        {
        case XR_SESSION_STATE_MAX_ENUM:
            return "MAX_ENUM";
        case XR_SESSION_STATE_IDLE:
            return "IDLE";
        case XR_SESSION_STATE_UNKNOWN:
            return "UNKNOWN";
        case XR_SESSION_STATE_FOCUSED:
            return "FOCUSED";
        case XR_SESSION_STATE_SYNCHRONIZED:
            return "SYNCHRONIZED";
        case XR_SESSION_STATE_VISIBLE:
            return "VISIBLE";
        case XR_SESSION_STATE_READY:
            return "READY";
        case XR_SESSION_STATE_STOPPING:
            return "STOPPING";
        case XR_SESSION_STATE_LOSS_PENDING:
            return "PENDING";
        case XR_SESSION_STATE_EXITING:
            return "EXITING";
        }
        return "UNKNOWN";
    }

    const char *OpenXRHMD::getReferenceSpaceType() const
    {
        switch (m_playSpaceType)
        {
        case XR_REFERENCE_SPACE_TYPE_VIEW:
            return "VIEW";
        case XR_REFERENCE_SPACE_TYPE_LOCAL:
            return "LOCAL";
        case XR_REFERENCE_SPACE_TYPE_STAGE:
            return "STAGE";
        case XR_REFERENCE_SPACE_TYPE_UNBOUNDED_MSFT:
            return "UNBOUNDED_MSFT";
        case XR_REFERENCE_SPACE_TYPE_COMBINED_EYE_VARJO:
            return "COMBINED_EYE_VARJO";
        default:
            return "UNKNOWN";
        }
    }

    void OpenXRHMD::getVisibilityMask(Eye eye, XrVisibilityMaskKHR &visibilityMask) const
    {
        uint32_t viewIndex = eyeToViewIndex(eye);
        if (viewIndex >= m_viewCount)
        {
            fprintf(stderr, "View for %s eye does not exist\n", eyeToString(eye));
            return;
        }

        PFN_xrGetVisibilityMaskKHR xrGetVisibilityMaskKHR = nullptr;
        XrResult result = xrGetInstanceProcAddr(m_instance, "xrGetVisibilityMaskKHR", (PFN_xrVoidFunction *)&xrGetVisibilityMaskKHR);
        if (!xrCheck(m_instance, result, "Failed to get xrGetVisibilityMaskKHR function pointer"))
            return;

        visibilityMask.type = XR_TYPE_VISIBILITY_MASK_KHR;
        visibilityMask.next = NULL;
        visibilityMask.vertexCapacityInput = 0;
        visibilityMask.indexCapacityInput = 0;
        result = xrGetVisibilityMaskKHR(m_session, m_viewType, viewIndex, XrVisibilityMaskTypeKHR::XR_VISIBILITY_MASK_TYPE_LINE_LOOP_KHR, &visibilityMask);
        if (!xrCheck(m_instance, result, "Failed to get visibility mask 1"))
            return;

        if (visibilityMask.vertexCountOutput == 0)
        {
            // Visibility mask is not supported
            return;
        }

        // Two-call idiom
        visibilityMask.vertices = (XrVector2f *)malloc(sizeof(XrVector2f) * (visibilityMask.vertexCountOutput + 1));
        visibilityMask.indices = (uint32_t *)malloc(sizeof(uint32_t) * visibilityMask.indexCountOutput);
        visibilityMask.vertexCapacityInput = visibilityMask.vertexCountOutput + 1;
        visibilityMask.indexCapacityInput = visibilityMask.indexCountOutput;
        printf("vertexCountOutput: %d\n", visibilityMask.vertexCountOutput);
        result = xrGetVisibilityMaskKHR(m_session, m_viewType, viewIndex, XrVisibilityMaskTypeKHR::XR_VISIBILITY_MASK_TYPE_LINE_LOOP_KHR, &visibilityMask);
        if (!xrCheck(m_instance, result, "Failed to get visibility mask 2"))
            return;
    }

} /*namespace sibr*/
