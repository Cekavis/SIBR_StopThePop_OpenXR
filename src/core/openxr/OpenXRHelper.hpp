// Software Name : SIBR_core
// SPDX-FileCopyrightText: Copyright (c) 2023 Orange
// SPDX-License-Identifier: Apache 2.0
//
// This software is distributed under the Apache 2.0 License;
// see the LICENSE file for more details.
//
// Author: Cédric CHEDALEUX <cedric.chedaleux@orange.com> et al.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <stdarg.h>
#include <math.h>
#include <openxr/openxr.h>
#include <core/system/Config.hpp>

namespace sibr
{

    bool xrCheck(XrInstance instance, XrResult result, const char *format, ...)
    {
        if (XR_SUCCEEDED(result))
        {
            return true;
        }

        char resultString[XR_MAX_RESULT_STRING_SIZE];
        xrResultToString(instance, result, resultString);

        char formatRes[XR_MAX_RESULT_STRING_SIZE + 1024];
        snprintf(formatRes, XR_MAX_RESULT_STRING_SIZE + 1023, "%s [%s] (%d)\n", format, resultString,
                 result);

        va_list args;
        va_start(args, format);
        vprintf(formatRes, args); // TODO: inject in SIBR_WARN
        va_end(args);

        return false;
    }

    // returns the preferred swapchain format if it is supported
    // else:
    // - if fallback is true, return the first supported format
    // - if fallback is false, return -1
    int64_t
    selectSwapchainFormat(XrInstance instance,
                          XrSession session,
                          int64_t preferred_format,
                          bool fallback)
    {
        XrResult result;

        uint32_t swapchain_format_count;
        result = xrEnumerateSwapchainFormats(session, 0, &swapchain_format_count, NULL);
        if (!xrCheck(instance, result, "Failed to get number of supported swapchain formats"))
            return -1;

        int64_t *swapchain_formats = (int64_t *)malloc(sizeof(int64_t) * swapchain_format_count);
        result = xrEnumerateSwapchainFormats(session, swapchain_format_count, &swapchain_format_count,
                                             swapchain_formats);
        if (!xrCheck(instance, result, "Failed to enumerate swapchain formats"))
            return -1;

        int64_t chosen_format = fallback ? swapchain_formats[0] : -1;

        for (uint32_t i = 0; i < swapchain_format_count; i++)
        {
            if (swapchain_formats[i] == preferred_format)
            {
                chosen_format = swapchain_formats[i];
                break;
            }
        }
        if (fallback && chosen_format != preferred_format)
        {
            SIBR_LOG << "Falling back to non preferred swapchain format " << chosen_format << std::endl;
        }

        free(swapchain_formats);

        return chosen_format;
    }

    void printApiLayers()
    {
        uint32_t count = 0;
        XrResult result = xrEnumerateApiLayerProperties(0, &count, NULL);
        if (!xrCheck(NULL, result, "Failed to enumerate api layer count"))
            return;

        if (count == 0)
            return;

        XrApiLayerProperties *props = (XrApiLayerProperties *)malloc(count * sizeof(XrApiLayerProperties));
        for (uint32_t i = 0; i < count; i++)
        {
            props[i].type = XR_TYPE_API_LAYER_PROPERTIES;
            props[i].next = NULL;
        }

        result = xrEnumerateApiLayerProperties(count, &count, props);
        if (!xrCheck(NULL, result, "Failed to enumerate api layers"))
            return;

        SIBR_LOG << "API layers:" << std::endl;
        for (uint32_t i = 0; i < count; i++)
        {
            SIBR_LOG << "\t " << props[i].layerName << " v" << props[i].layerVersion << ": " << props[i].description << std::endl;
        }

        free(props);
    }

    bool getRuntimeNameAndVersion(XrInstance instance, std::string &name, std::string &version)
    {
        XrResult result;
        XrInstanceProperties instance_props;
        instance_props.type = XR_TYPE_INSTANCE_PROPERTIES;
        instance_props.next = NULL;

        result = xrGetInstanceProperties(instance, &instance_props);
        if (!xrCheck(NULL, result, "Failed to get instance info"))
            return false;

        name = std::string(instance_props.runtimeName);
        std::stringstream ssVersion;
        ssVersion << XR_VERSION_MAJOR(instance_props.runtimeVersion) << "." << XR_VERSION_MINOR(instance_props.runtimeVersion) << "."
                  << XR_VERSION_PATCH(instance_props.runtimeVersion);
        version = ssVersion.str();

        return true;
    }

    void printSystemProperties(XrInstance instance, XrSystemId systemId)
    {
        XrResult result;
        XrSystemProperties system_props;
        system_props.type = XR_TYPE_SYSTEM_PROPERTIES;
        system_props.next = NULL;

        result = xrGetSystemProperties(instance, systemId, &system_props);
        if (!xrCheck(instance, result, "Failed to get System properties"))
            return;

        SIBR_LOG << "System properties for system " << system_props.systemId << ": " << system_props.systemName << ", vendor ID " << system_props.vendorId << std::endl;
        SIBR_LOG << "\tMax layers          : " << system_props.graphicsProperties.maxLayerCount << std::endl;
        SIBR_LOG << "\tMax swapchain height: " << system_props.graphicsProperties.maxSwapchainImageHeight << std::endl;
        SIBR_LOG << "\tMax swapchain width : " << system_props.graphicsProperties.maxSwapchainImageWidth << std::endl;
        SIBR_LOG << "\tOrientation Tracking: " << system_props.trackingProperties.orientationTracking << std::endl;
        SIBR_LOG << "\tPosition Tracking   : " << system_props.trackingProperties.positionTracking << std::endl;
    }

    void printViewconfigViewInfo(uint32_t m_viewCount, XrViewConfigurationView *m_viewConfigViews)
    {
        for (uint32_t i = 0; i < m_viewCount; i++)
        {
            SIBR_LOG << "View Configuration View " << i << std::endl;
            SIBR_LOG << "\tResolution       : Recommended " << m_viewConfigViews[i].recommendedImageRectWidth << "x" << m_viewConfigViews[i].recommendedImageRectHeight
                     << ", Max:" << m_viewConfigViews[i].maxImageRectWidth << "x" << m_viewConfigViews[i].maxImageRectHeight << std::endl;
            SIBR_LOG << "\tSwapchain Samples Count: Recommended: " << m_viewConfigViews[i].recommendedSwapchainSampleCount << ", Max: " << m_viewConfigViews[i].maxSwapchainSampleCount << std::endl;
        }
    }

    template <typename T>
    T radianToDegree(T value)
    {
        return value * 180.f / M_PI;
    }

} /*namespace sibr*/