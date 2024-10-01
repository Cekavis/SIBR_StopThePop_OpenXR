// Software Name : SIBR_core
// SPDX-FileCopyrightText: Copyright (c) 2023 Orange
// SPDX-License-Identifier: Apache 2.0
//
// This software is distributed under the Apache 2.0 License;
// see the LICENSE file for more details.
//
// Author: Cédric CHEDALEUX <cedric.chedaleux@orange.com> et al.

#include <core/graphics/GUI.hpp>
#include <core/assets/Resources.hpp>
#include <core/openxr/OpenXRRdrMode.hpp>
#include <core/openxr/SwapchainImageRenderTarget.hpp>
#include <fstream>

namespace sibr
{

#ifdef XR_USE_PLATFORM_XLIB
    XrGraphicsBindingOpenGLXlibKHR createXrGraphicsBindingOpenGLXlibKHR(Display *display, GLXDrawable drawable, GLXContext context)
    {
        return XrGraphicsBindingOpenGLXlibKHR{
            .type = XR_TYPE_GRAPHICS_BINDING_OPENGL_XLIB_KHR,
            .xDisplay = display,
            .glxDrawable = drawable,
            .glxContext = context};
    }
#endif

#ifdef XR_USE_PLATFORM_WIN32
    XrGraphicsBindingOpenGLWin32KHR createXrGraphicsBindingOpenGLWin32KHR(HDC hdc, HGLRC hglrc)
    {
        // Windows C++ compiler does not support C99 designated initializers
        return XrGraphicsBindingOpenGLWin32KHR{
            XR_TYPE_GRAPHICS_BINDING_OPENGL_WIN32_KHR, // .type
            NULL,                                      // .next
            hdc,                                       // .hDC
            hglrc                                      // .hGLRC
        };
    }
#endif

    OpenXRRdrMode::OpenXRRdrMode(sibr::Window &window)
    {
        m_quadShader.init("Texture",
                          sibr::loadFile(sibr::Resources::Instance()->getResourceFilePathName("texture.vp")),
                          sibr::loadFile(sibr::Resources::Instance()->getResourceFilePathName("texture.fp")));

        m_openxrHmd = std::make_unique<OpenXRHMD>("Gaussian splatting");
        m_openxrHmd->init();

        bool sessionCreated = false;
#if defined(XR_USE_PLATFORM_XLIB)
        sessionCreated = m_openxrHmd->startSession(createXrGraphicsBindingOpenGLXlibKHR(glfwGetX11Display(), glXGetCurrentDrawable(), glfwGetGLXContext(window.GLFW())));
#elif defined(XR_USE_PLATFORM_WIN32)
        sessionCreated = m_openxrHmd->startSession(createXrGraphicsBindingOpenGLWin32KHR(wglGetCurrentDC(), wglGetCurrentContext()));
#endif
        if (!sessionCreated)
        {
            SIBR_ERR << "Failed to connect to OpenXR" << std::endl;
        }


        // Initialize visibility mask
        bool write = false;
        m_visibilityMask_fullres[0] = initVisibilityMask(OpenXRHMD::Eye::LEFT, 16, false, "fullres_mask_left", write);
        m_visibilityMask_fullres[1] = initVisibilityMask(OpenXRHMD::Eye::RIGHT, 16, false, "fullres_mask_right", write);
        
        m_visibilityMask_halfres[0] = initVisibilityMask(OpenXRHMD::Eye::LEFT, 32, false, "halfres_mask_left", write);
        m_visibilityMask_halfres[1] = initVisibilityMask(OpenXRHMD::Eye::RIGHT, 32, false, "halfres_mask_right", write);

        SIBR_LOG << "Disable VSync: use headset synchronization." << std::endl;
        window.setVsynced(false);

        m_openxrHmd->setIdleAppCallback([this]()
                                        { m_appFocused = false; });
        m_openxrHmd->setVisibleAppCallback([this]()
                                           { m_appFocused = false; });
        m_openxrHmd->setFocusedAppCallback([this]()
                                           { m_appFocused = true; });
    }

    OpenXRRdrMode::~OpenXRRdrMode()
    {
        m_RTPool.clear();
        m_openxrHmd->closeSession();
        m_openxrHmd->terminate();
    }

    void OpenXRRdrMode::render(ViewBase &view, const sibr::Camera &camera, const sibr::Viewport &viewport, IRenderTarget *optDest)
    {
        // Render the UI with OpenXR infos
        onGui();

        if (!m_openxrHmd->isSessionRunning())
        {
            return;
        }

        const int w = m_openxrHmd->getResolution().x();
        const int h = m_openxrHmd->getResolution().y();

        // Prepare the view to render at a specific resolution
        view.setResolution(sibr::Vector2i(w / m_downscaleResolution, h / m_downscaleResolution));

        // The callback is called for each single view (left view then right view) with the texture to render to
        m_openxrHmd->submitFrame([this, w, h, &view, &camera, optDest](int viewIndex, uint32_t texture)
                                 {
                                    OpenXRHMD::Eye eye = viewIndex == 0 ? OpenXRHMD::Eye::LEFT : OpenXRHMD::Eye::RIGHT;

                                    auto fov = this->m_openxrHmd->getFieldOfView(eye);
                                    float scaley = tan(fov.w()) - tan(fov.z());
                                    float scalex = tan(fov.y()) - tan(fov.x());
                                    float aspect = scalex / scaley;

                                    auto q = viewIndex == 0 ? camera.rotation() : camera.rightTransform().rotation();
                                    auto pos = viewIndex == 0 ? camera.position() : camera.rightTransform().position();

                                    // Define camera from OpenXR eye view position/orientation/fov
                                    Camera cam;
                                    cam.rotate(q);
                                    cam.position(pos);
                                    cam.zfar(camera.zfar());
                                    cam.znear(camera.znear());

                                    // all fov values are needed to calculate a correct perspective matrix
                                    cam.setAllFov(fov);

                                     // Note: setStereoCam() used in SteroAnaglyph canno be reused here,
                                     // because headset eye views have asymetric fov
                                     // We therefore use the perspective() method with principal point positioning instead
                                     cam.principalPoint(Eigen::Vector2f(1.f, 1.f) - this->m_openxrHmd->getScreenCenter(eye));
                                     cam.perspective(fov.w() - fov.z(), aspect, cam.znear(), cam.zfar());

                                     cam.setVisibilityMaskFullres(m_visibilityMask_fullres[viewIndex]);
                                     cam.setVisibilityMaskHalfres(m_visibilityMask_halfres[viewIndex]);

                                     // Get the render target holding the swapchain image's texture from the pool
                                     auto rt = getRenderTarget(texture, w, h);
                                     if (!rt)
                                     {
                                         return;
                                     }

                                     // Perform the scene rendering for the given view into the RenderTarget's FBO
                                     rt->clear();
                                     rt->bind();
                                     glViewport(0, 0, w, h);

                                     auto start = std::chrono::steady_clock::now();
                                     view.onRenderIBR(*rt.get(), cam);
                                     auto end = std::chrono::steady_clock::now();
                                     double elapsed_seconds = std::chrono::duration<double>(end - start).count();
                                     static double avg = 0;
                                     static int frame_count = 0;
                                     avg += elapsed_seconds;
                                     ++frame_count;
                                     if (frame_count == 500)
                                     {
                                         SIBR_LOG << "Rendering time: " << avg * 2 / frame_count * 1000 << "ms" << std::endl;
                                         avg = 0;
                                         frame_count = 0;
                                     }

                                     rt->unbind();

                                     // Draw the left and right textures into the UI window
                                     if (optDest)
                                     {
                                         glViewport(eye == OpenXRHMD::Eye::LEFT ? 0.f : optDest->w() / 2.f, 0, optDest->w() / 2.f, optDest->h());
                                         glScissor(eye == OpenXRHMD::Eye::LEFT ? 0.f : optDest->w() / 2.f, 0, optDest->w() / 2.f, optDest->h());
                                         optDest->bind();
                                     }
                                     else
                                     {
                                         glViewport(eye == OpenXRHMD::Eye::LEFT ? 0.f : w / 2.f, 0, w / 2.f, h);
                                         glScissor(eye == OpenXRHMD::Eye::LEFT ? 0.f : w / 2.f, 0, w / 2.f, h);
                                     }
                                     glEnable(GL_SCISSOR_TEST);
                                     glDisable(GL_BLEND);
                                     glDisable(GL_DEPTH_TEST);
                                     glClearColor(1.f, 0.f, 0.f, 1.f);
                                     glClear(GL_COLOR_BUFFER_BIT);
                                     m_quadShader.begin();
                                     glActiveTexture(GL_TEXTURE0);
                                     glBindTexture(GL_TEXTURE_2D, texture);
                                     RenderUtility::renderScreenQuad();
                                     glBindTexture(GL_TEXTURE_2D, 0);
                                     m_quadShader.end();
                                     glDisable(GL_SCISSOR_TEST);
                                     if (optDest)
                                     {
                                         optDest->unbind();
                                     }
                                 });
    }

    void OpenXRRdrMode::preparePosePrediction()
    {
        // Get next pose prediction for rendering
        m_openxrHmd->pollEvents();
        if (!m_openxrHmd->waitNextFrame())
        {
            return;
        }
    }

    Eigen::Quaternionf OpenXRRdrMode::getRotation(int idx)
    {
        return m_openxrHmd->getPoseQuaternion(OpenXRHMD::Eye(idx));
    }

    Eigen::Vector3f OpenXRRdrMode::getPosition(int idx)
    {
        return m_openxrHmd->getPosePosition(OpenXRHMD::Eye(idx));
    }

    void OpenXRRdrMode::onGui()
    {
        const std::string guiName = "OpenXR";
        ImGui::Begin(guiName.c_str());
        std::string status = "KO";
        if (m_openxrHmd->isSessionRunning())
        {
            status = m_appFocused ? "FOCUSED" : "IDLE";
        }
        ImGui::Text("Session status: %s", status.c_str());
        ImGui::Text("Runtime: %s (%s)", m_openxrHmd->getRuntimeName().c_str(), m_openxrHmd->getRuntimeVersion().c_str());
        ImGui::Text("Reference space type: %s", m_openxrHmd->getReferenceSpaceType());
        ImGui::RadioButton("Free world standing", &m_vrExperience, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Seated", &m_vrExperience, 1);
        ImGui::Checkbox("Y-Invert scene", &m_flipY);
        if (m_openxrHmd->isSessionRunning())
        {
            const auto report = m_openxrHmd->getRefreshReport();
            ImGui::Text("Framerate: %.2f FPS (expected: %.2f FPS)", report.measuredFramerate, report.expectedFramerate);
            const auto w = m_openxrHmd->getResolution().x();
            const auto h = m_openxrHmd->getResolution().y();
            ImGui::Text("Headset resolution (per eye): %ix%i", w, h);
            ImGui::Text("Rendering resolution (per eye): %ix%i", w / m_downscaleResolution, h / m_downscaleResolution);
			ImGui::SliderInt("Down scale factor", &m_downscaleResolution, 1, 8);
            const auto leftPos = this->m_openxrHmd->getPosePosition(OpenXRHMD::Eye::LEFT);
            const auto rightPos = this->m_openxrHmd->getPosePosition(OpenXRHMD::Eye::RIGHT);
            const float eyeDist = sqrt(pow(leftPos.x() - rightPos.x(), 2) + pow(leftPos.y() - rightPos.y(), 2) + pow(leftPos.z() - rightPos.z(), 2));
            ImGui::Text("IPD: %.1fcm", eyeDist * 100.f);
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenOverlapped))
            {
                ImGui::SetTooltip("Inter-pupillary distance");
            }
            if (ImGui::CollapsingHeader("Left eye:"))
            {
                const auto fov = m_openxrHmd->getFieldOfView(OpenXRHMD::Eye::LEFT, OpenXRHMD::AngleUnit::DEGREE);
                ImGui::Text("FOV: %.2f°, %.2f°, %.2f°, %.2f°", fov.x(), fov.y(), fov.z(), fov.w());
                ImGui::Text("Position : %.2f, %.2f, %.2f", leftPos.x(), leftPos.y(), leftPos.z());
            }
            if (ImGui::CollapsingHeader("Right eye:"))
            {
                const auto fov = m_openxrHmd->getFieldOfView(OpenXRHMD::Eye::RIGHT, OpenXRHMD::AngleUnit::DEGREE);
                ImGui::Text("FOV: %.2f°, %.2f°, %.2f°, %.2f°", fov.x(), fov.y(), fov.z(), fov.w());
                ImGui::Text("Position : %.2f, %.2f, %.2f", rightPos.x(), rightPos.y(), rightPos.z());
            }
        }
        ImGui::End();
    }

    SwapchainImageRenderTarget::Ptr OpenXRRdrMode::getRenderTarget(uint32_t texture, uint w, uint h)
    {
        auto i = m_RTPool.find(texture);
        if (i != m_RTPool.end())
        {
            return i->second;
        }
        else
        {
            SwapchainImageRenderTarget::Ptr newRt = std::make_shared<SwapchainImageRenderTarget>(texture, w, h);
            auto pair = m_RTPool.insert(std::make_pair<int, SwapchainImageRenderTarget::Ptr>(texture, std::move(newRt)));
            if (pair.second)
            {
                return (*pair.first).second;
            }
        }
        return SwapchainImageRenderTarget::Ptr();
    }

    bool pointInPolygon(const XrVector2f *p, const int n, const float x, const float y)
    {
        bool c = false;
        for (int i = 0, j = n - 1; i < n; j = i++)
        {
            if (((p[j].y < y) != (p[i].y < y)) && (((p[i].x - x) * (p[j].y - y) > (p[j].x - x) * (p[i].y - y)) ^ (p[j].y >= y)))
                c = !c;
        }
        return c;
    }

    std::pair<uint32_t*, uint32_t*> OpenXRRdrMode::initVisibilityMask(OpenXRHMD::Eye eye, const int tileSize, bool innerInvisible, const std::string filename, const bool write)
    {
        // Get visibility mask
        XrVisibilityMaskKHR visibilityMask;
        m_openxrHmd->getVisibilityMask(eye, visibilityMask);

        auto fov = m_openxrHmd->getFieldOfView(eye);
        float xmin = tan(fov.x());
        float xmax = tan(fov.y());
        float ymin = tan(fov.z());
        float ymax = tan(fov.w());

        const int w = m_openxrHmd->getResolution().x();
        const int h = m_openxrHmd->getResolution().y();

        const int tileW = (w + tileSize - 1) / tileSize;
        const int tileH = (h + tileSize - 1) / tileSize;

        const int mask_size = (tileW * tileH + 31) / 32 * 4;
        uint32_t* mask = (uint32_t*) malloc(mask_size);
        memset(mask, -1, mask_size);

        const int mask_sum_size = (tileW + 1) * (tileH + 1) * 4;
        uint32_t* mask_sum = (uint32_t*) malloc(mask_sum_size);
        memset(mask_sum, 0, mask_sum_size);

        if (visibilityMask.vertexCountOutput > 0)
        {
            auto vertices = visibilityMask.vertices;
            for (int i = 0; i < visibilityMask.vertexCountOutput; i++)
            {
                vertices[i].x = (vertices[i].x - xmin) / (xmax - xmin);
                vertices[i].y = (vertices[i].y - ymin) / (ymax - ymin);
            }
            vertices[visibilityMask.vertexCountOutput] = vertices[0];

            for (int i = 0; i < tileW; i++) for (int j = 0; j < tileH; j++)
            {
                int idx = i * tileH + j;
                float x = (i * tileSize + 0.5f) / w;
                float y = (j * tileSize + 0.5f) / h;
                bool in = pointInPolygon(vertices, visibilityMask.vertexCountOutput, x, y) | 
                          pointInPolygon(vertices, visibilityMask.vertexCountOutput, x + (tileSize - 1.0f) / w, y) | 
                          pointInPolygon(vertices, visibilityMask.vertexCountOutput, x, y + (tileSize - 1.0f) / h) | 
                          pointInPolygon(vertices, visibilityMask.vertexCountOutput, x + (tileSize - 1.0f) / w, y + (tileSize - 1.0f) / h);

                if (!in) mask[idx / 32] &= ~(1u << (idx % 32));
            }
        }

        // Mask central part
		int cx = w / (tan(fov.y()) - tan(fov.x())) * tan(-fov.x()) / 2;
        int cy = h / (tan(fov.w()) - tan(fov.z())) * tan(fov.w()) / 2;
        float ratio = 0.1;
        for (int i = 0; i < tileW; i++) for (int j = 0; j < tileH; j++)
        {
            int idx = i * tileH + j;
            int x = i * tileSize;
            int y = j * tileSize;
            bool in = x < cx + w * ratio / 2 ||
                      x + tileSize > cx + w / 2 - w * ratio / 2 ||
                      y < cy + h * ratio / 2 ||
                      y + tileSize > cy + h / 2 - h * ratio / 2;
            if (innerInvisible && !in) mask[idx / 32] &= ~(1u << (idx % 32));
        }

        for (int i = 1; i <= tileW; i++) for (int j = 1; j <= tileH; j++)
        {
            int idx = (i - 1) * tileH + j - 1;
            mask_sum[i * (tileH + 1) + j] = mask[idx / 32] >> (idx % 32) & 1;
        }

        for (int i = 1; i <= tileW; i++) for (int j = 1; j <= tileH; j++)
        {
            mask_sum[i * (tileH + 1) + j] += mask_sum[(i - 1) * (tileH + 1) + j];
        }

        for (int i = 1; i <= tileW; i++) for (int j = 1; j <= tileH; j++)
        {
            mask_sum[i * (tileH + 1) + j] += mask_sum[i * (tileH + 1) + j - 1];
        }

        // for (int j = tileH - 1; j >= 0; j--)
        // {
        //     for (int i = 0; i < tileW; i++)
        //     {
        //         int idx = i * tileH + j;
        //         if (mask[idx / 32] & (1 << (idx % 32)))
        //         {
        //             printf("X");
        //         }
        //         else
        //         {
        //             printf(".");
        //         }
        //     }
        //     printf("\n");
        // }

        if (write)
        {
            std::ofstream fout(filename + ".dat", std::ios::binary);
            fout.write((char*) mask, mask_size);
            fout.close();
            
            std::ofstream fout_sum(filename + "_sum.dat", std::ios::binary);
            fout_sum.write((char*) mask_sum, mask_sum_size);
            fout_sum.close();
        }

        return std::make_pair(mask, mask_sum);
    }

} /*namespace sibr*/