// Software Name : SIBR_core
// SPDX-FileCopyrightText: Copyright (c) 2023 Orange
// SPDX-License-Identifier: Apache 2.0
//
// This software is distributed under the Apache 2.0 License;
// see the LICENSE file for more details.
//
// Author: Cédric CHEDALEUX <cedric.chedaleux@orange.com> et al.

#pragma once

#include <vector>
#include <chrono>

// Should be defined before Xlib.h which define SUCCESS preprocessor commands
#include <Eigen/Dense>

#ifdef _WIN32
// #define GLEW_STATIC
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")
#endif

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#ifdef __linux__
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_GLX
#elif _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#endif
#include <GLFW/glfw3native.h> // for glfwGetX11Display and glfwGetGLXContext

// Required headers for OpenGL rendering, as well as for including openxr_platform
#define GL3_PROTOTYPES
#include <GL/gl.h>
#ifdef __linux__
#define GL_GLEXT_PROTOTYPES
#include <GL/glext.h>
#endif

#ifdef __linux__
// Required headers for windowing, as well as the XrGraphicsBindingOpenGLXlibKHR
#include <GL/glx.h>
#include <X11/Xlib.h>
#endif

#ifdef __linux__
#define XR_USE_PLATFORM_XLIB
#define XR_USE_TIMESPEC
#elif _WIN32
#define XR_USE_PLATFORM_WIN32
#endif
#define XR_USE_GRAPHICS_API_OPENGL

#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

using namespace Eigen;

namespace sibr
{

    /**
     *	Class that communicates with the Headed-Mounted display (aka VR headset) through an OpenXR loader.
     *  The OpenXR loader enumarates all available OpenXR runtimes for the requested form factor (i.e. HMD)
     *  and returns an instance to communicate with the HMD.
     *   \ingroup sibr_openxr
     */
    class OpenXRHMD
    {
    public:
        enum class Eye
        {
            LEFT = 0,
            RIGHT = 1
        };

        enum class AngleUnit
        {
            RADIAN,
            DEGREE
        };

        enum class SessionStatus
        {
            STOPPED = 0,
            IDLE,
            BEGINNING,
            SYNCHRONIZED,
            ENDING,
            FAILURE // ERROR type cannot be used in Windows
        };

        struct FrameRefreshReport
        {
            unsigned int nbMissedFrames = 0;
            unsigned int totalRenderedFrames = 0;
            float expectedFramerate = 0.f;
            float measuredFramerate = 0.f;
            std::chrono::time_point<std::chrono::high_resolution_clock> firstFrameTimestamp;
        };

        typedef std::function<void(int viewIndex, uint32_t renderTexture)> RenderFunc;

        OpenXRHMD(const std::string &applicationName, bool seated = false);
        ~OpenXRHMD();

        /**
         * @brief Initialize the OpenXR player
         *
         * Scan all connected XR runtimes and find a headset device,
         * then extract the device's recommended resolution
         */
        bool init();

        /**
         * @brief Disconnect from the XR runtime bound to the headset device
         *
         * To be called after closeSession()
         */
        bool terminate();

        /**
         * @brief Return the report of frame refresh
         * @return Frame refresh report with:
         *         - number of missed deadline frames
         *         - total submitted frames
         *         - framerate expected by the headset
         */
        const FrameRefreshReport &getRefreshReport() const;

        /**
         * @brief Create a session with the connected headset
         * @param: XrGraphicsBindingOpenGLXlibKHR or XrGraphicsBindingOpenGLWin32KHR holding information about OpenGL window
         * @return true if session has been established and device is waiting for frame loop rendering
         *
         */
        template <typename T>
        bool startSession(T graphicsBindingGL);

        /**
         * @brief Close the running XR m_session
         */
        bool closeSession();

        /**
         * @brief Poll the XR runtime to know the XR session state
         *
         * Should be called before each frame loop sequence (waitFrame/submitFrame)
         */
        bool pollEvents();

        /**
         * @brief Callback to be notified when the headset idles the XR session
         */
        void setIdleAppCallback(const std::function<void()> &callback);

        /**
         * @brief Callback to be notified when the headset makes the XR app visible
         */
        void setVisibleAppCallback(const std::function<void()> &callback);

        /**
         * @brief Callback to be notified when the headset makes the XR app focused
         */
        void setFocusedAppCallback(const std::function<void()> &callback);

        /**
         * @brief Return if the XR session is currently running on the device
         */
        bool isSessionRunning() const;

        /**
         * @brief Wait for the headset to provides the next frame's predicted display time and eye poses
         *
         * Block until the headset provides the frame info
         *
         * To be called before submitFrame() call.
         */
        bool waitNextFrame();

        /**
         * @brief Return if the application should render the next frame or just call submitFrame
         */
        bool shouldRender() const;

        /**
         * @brief Submit an empty frame
         */
        bool submitFrame();

        /**
         * @brief Submit an empty frame
         *
         * To be called after waitFrame() if the next frame does not await a rendered frame
         *
         * renderFunc is called for each view
         */
        bool submitFrame(RenderFunc renderFunc);

        /**
         * @brief Return the resolution recommended for the connected headset
         */
        Eigen::Vector2i getRecommendedResolution() const;

        /**
         * @brief Change the headset resolution
         * @param resolution
         *
         * To be called before startSession(). If not called, the recommended resolution is used.
         */
        void setResolution(const Eigen::Vector2i &resolution);

        /**
         * @brief Get the headset resolution
         * @return resolution:
         */
        const Eigen::Vector2i &getResolution() const;

        /**
         * @brief Return the yaw, roll, pich of the eye pose
         * @param eye: LEFT or RIGHT
         * @param unit: RADIAN or DEGREE (RADIAN by default)
         */
        Eigen::Vector3f getPoseOrientation(Eye eye, AngleUnit unit = AngleUnit::RADIAN) const;

        /**
         * @brief Return the quaternion of the eye pose
         * @param eye: LEFT or RIGHT
         */
        Eigen::Quaternionf getPoseQuaternion(Eye eye) const;

        /**
         * @brief Return the position of the eye pose (in OpenXR world coordinates - +x is right, +y is down, +z is backward)
         * @param eye: LEFT or RIGHT
         */
        Eigen::Vector3f getPosePosition(Eye eye) const;

        /**
         * @brief Return the fields of view of each eye
         * @return a list of angles: left, right, down and up
         * @param eye: LEFT or RIGHT
         * @param unit: RADIAN or DEGREE (RADIAN by default)
         *
         * Each eye has a different left and right field of view due to IPD (Inter pupillary distance)
         */
        Eigen::Vector4f getFieldOfView(Eye eye, AngleUnit unit = AngleUnit::RADIAN) const;

        /**
         * @brief Return the horizontal and vertical fields of view of each eye
         * @return horizontal and vertical field of view angle
         *
         * Each eye has the same horizontal and vertical field of view
         */
        Eigen::Vector2f getHVFieldOfView(AngleUnit unit = AngleUnit::RADIAN) const;

        /**
         * @brief Return the screen center (between 0.0f and 1.f) of each eye
         *
         * Each eye has a different horizontal center (due to IPD)
         */
        Eigen::Vector2f getScreenCenter(Eye eye) const;

        /**
         * @brief Return the current reference space type
         * Possible values are:
         * - VIEW: Head-view locked
         * - LOCAL: Seated position
         * - STAGE: Room-scale
         */
        const char *getReferenceSpaceType() const;

        /**
         * @brief Return the runtime name
         */
        const std::string &getRuntimeName() const
        {
            return m_runtimeName;
        }

        /**
         * @brief Return the runtime version
         */
        const std::string &getRuntimeVersion() const
        {
            return m_runtimeVersion;
        }

        /**
         * @brief Return the visilibity mask of each eye
         */
        void getVisibilityMask(Eye eye, XrVisibilityMaskKHR &visibilityMask) const;

    private:
        // Name of the application displayed in the headset
        std::string m_applicationName;

        // Typically STAGE for room scale/standing, LOCAL for seated - defined in constructor
        XrReferenceSpaceType m_playSpaceType;

        // Last frame state updated after a successful xrWaitFrame call
        XrFrameState m_lastFrameState;

        // Store the current XR Session state
        XrSessionState m_currentState = XR_SESSION_STATE_UNKNOWN;

        // Internal status to handle m_session connection/deconnection sequence
        SessionStatus m_status = SessionStatus::STOPPED;

        // Headset resolution
        Eigen::Vector2i m_resolution{0, 0};

        // Form factor set at HEADSET type
        XrFormFactor m_formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;

        // So far, we only handle stereo headset displays (other view type is for AR devices with one primary display)
        XrViewConfigurationType m_viewType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;

        // The play space instance
        XrSpace m_playSpace = XR_NULL_HANDLE;
        // The instance handle can be thought of as the basic connection to the OpenXR runtime
        XrInstance m_instance = XR_NULL_HANDLE;
        // The system represents an (opaque) set of XR devices in use, managed by the runtime
        XrSystemId m_systemId = XR_NULL_SYSTEM_ID;
        // The session deals with the renderloop submitting frames to the runtime
        XrSession m_session = XR_NULL_HANDLE;

        // Each physical Display/Eye is described by a view.
        // Dynamically allocating all view related structs instead of assuming 2
        uint32_t m_viewCount = 0;
        // The viewconfiguration views contain information like resolution about each view
        XrViewConfigurationView *m_viewConfigViews = NULL;

        // Array of containers for submitting swapchains with rendered VR frames
        XrCompositionLayerProjectionView *m_projectionViews = NULL;
        // Array of views, filled by the runtime with current HMD display pose
        XrView *views = NULL;

        // Array of handles for swapchains
        XrSwapchain *m_swapchains = NULL;
        // Array of integers, storing the length of swapchains
        uint32_t *m_swapchainsLengths = NULL;
        // Array of array of swapchains containers holding an OpenGL texture
        // that is allocated by the runtime
        XrSwapchainImageOpenGLKHR **m_swapchainsImages = NULL;

        // Store info about frame submitting
        FrameRefreshReport m_lastFrameRefreshReport;
        FrameRefreshReport m_currentFrameRefreshReport;

        // Runtime name and version
        std::string m_runtimeName = "";
        std::string m_runtimeVersion = "";

        // Debug purpose
        bool m_printApiLayers = false;
        bool m_printSystemProperties = false;
        bool m_printViewConfigInfos = false;
        bool m_printRuntimeExtensions = false;

        // Callbacks for notifying XR Application state change
        std::function<void()> m_idleCallback;
        std::function<void()> m_visibleCallback;
        std::function<void()> m_focusedCallback;

        // Internal methods
        bool createSession(const XrSessionCreateInfo &m_sessionCreateInfo);
        bool createReferenceSpace();
        bool createSwapchain();
        bool synchronizeSession();
        void updateCurrentSessionState(const XrSessionState &state);
        void updateRefreshReport();

        // Helper methods
        static Eigen::Vector3f quaternionToEulerAngles(const XrQuaternionf &q);
        static XrQuaternionf eulerAnglestoQuaternion(float roll, float pitch, float yaw);
        static uint32_t eyeToViewIndex(Eye eye);
        static const char *eyeToString(Eye eye);
        static const char *sessionStateToString(const XrSessionState &state);
    };

} /*namespace sibr*/