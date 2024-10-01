/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */

#include <projects/gaussianviewer/renderer/GaussianView.hpp>
#include <core/graphics/GUI.hpp>
#include <thread>
#include <boost/asio.hpp>
#include <imgui_internal.h>
#include <npp.h>
#include <nppi.h>

// Define the types and sizes that make up the contents of each Gaussian 
// in the trained model.
typedef sibr::Vector3f Pos;
template<int D>
struct SHs
{
	float shs[(D+1)*(D+1)*3];
};
struct Scale
{
	float scale[3];
};
struct Rot
{
	float rot[4];
};
template<int D>
struct RichPoint
{
	Pos pos;
	float n[3];
	SHs<D> shs;
	float opacity;
	Scale scale;
	Rot rot;
};

float sigmoid(const float m1)
{
	return 1.0f / (1.0f + exp(-m1));
}

float inverse_sigmoid(const float m1)
{
	return log(m1 / (1.0f - m1));
}

# define CUDA_SAFE_CALL_ALWAYS(A) \
A; \
cudaDeviceSynchronize(); \
if (cudaPeekAtLastError() != cudaSuccess) \
SIBR_ERR << cudaGetErrorString(cudaGetLastError());

#if DEBUG || _DEBUG
# define CUDA_SAFE_CALL(A) CUDA_SAFE_CALL_ALWAYS(A)
#else
# define CUDA_SAFE_CALL(A) A
#endif

// Load the Gaussians from the given file.
template<int D>
int loadPly(const char* filename,
	std::vector<Pos>& pos,
	std::vector<SHs<3>>& shs,
	std::vector<float>& opacities,
	std::vector<Scale>& scales,
	std::vector<Rot>& rot,
	sibr::Vector3f& minn,
	sibr::Vector3f& maxx)
{
	std::ifstream infile(filename, std::ios_base::binary);

	if (!infile.good())
		SIBR_ERR << "Unable to find model's PLY file, attempted:\n" << filename << std::endl;

	// "Parse" header (it has to be a specific format anyway)
	std::string buff;
	std::getline(infile, buff);
	std::getline(infile, buff);

	std::string dummy;
	std::getline(infile, buff);
	std::stringstream ss(buff);
	int count;
	ss >> dummy >> dummy >> count;

	// Output number of Gaussians contained
	SIBR_LOG << "Loading " << count << " Gaussian splats" << std::endl;

	while (std::getline(infile, buff))
		if (buff.compare("end_header") == 0)
			break;

	// Read all Gaussians at once (AoS)
	std::vector<RichPoint<D>> points(count);
	infile.read((char*)points.data(), count * sizeof(RichPoint<D>));

	// Resize our SoA data
	pos.resize(count);
	shs.resize(count);
	scales.resize(count);
	rot.resize(count);
	opacities.resize(count);

	// Gaussians are done training, they won't move anymore. Arrange
	// them according to 3D Morton order. This means better cache
	// behavior for reading Gaussians that end up in the same tile 
	// (close in 3D --> close in 2D).
	minn = sibr::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
	maxx = -minn;
	for (int i = 0; i < count; i++)
	{
		maxx = maxx.cwiseMax(points[i].pos);
		minn = minn.cwiseMin(points[i].pos);
	}
	std::vector<std::pair<uint64_t, int>> mapp(count);
	for (int i = 0; i < count; i++)
	{
		sibr::Vector3f rel = (points[i].pos - minn).array() / (maxx - minn).array();
		sibr::Vector3f scaled = ((float((1 << 21) - 1)) * rel);
		sibr::Vector3i xyz = scaled.cast<int>();

		uint64_t code = 0;
		for (int i = 0; i < 21; i++) {
			code |= ((uint64_t(xyz.x() & (1 << i))) << (2 * i + 0));
			code |= ((uint64_t(xyz.y() & (1 << i))) << (2 * i + 1));
			code |= ((uint64_t(xyz.z() & (1 << i))) << (2 * i + 2));
		}

		mapp[i].first = code;
		mapp[i].second = i;
	}
	auto sorter = [](const std::pair < uint64_t, int>& a, const std::pair < uint64_t, int>& b) {
		return a.first < b.first;
	};
	std::sort(mapp.begin(), mapp.end(), sorter);

	// Move data from AoS to SoA
	int SH_N = (D + 1) * (D + 1);
	for (int k = 0; k < count; k++)
	{
		int i = mapp[k].second;
		pos[k] = points[i].pos;

		// Normalize quaternion
		float length2 = 0;
		for (int j = 0; j < 4; j++)
			length2 += points[i].rot.rot[j] * points[i].rot.rot[j];
		float length = sqrt(length2);
		for (int j = 0; j < 4; j++)
			rot[k].rot[j] = points[i].rot.rot[j] / length;

		// Exponentiate scale
		for(int j = 0; j < 3; j++)
			scales[k].scale[j] = exp(points[i].scale.scale[j]);

		// Activate alpha
		opacities[k] = sigmoid(points[i].opacity);

		shs[k].shs[0] = points[i].shs.shs[0];
		shs[k].shs[1] = points[i].shs.shs[1];
		shs[k].shs[2] = points[i].shs.shs[2];
		for (int j = 1; j < SH_N; j++)
		{
			shs[k].shs[j * 3 + 0] = points[i].shs.shs[(j - 1) + 3];
			shs[k].shs[j * 3 + 1] = points[i].shs.shs[(j - 1) + SH_N + 2];
			shs[k].shs[j * 3 + 2] = points[i].shs.shs[(j - 1) + 2 * SH_N + 1];
		}
	}
	return count;
}

void savePly(const char* filename,
	const std::vector<Pos>& pos,
	const std::vector<SHs<3>>& shs,
	const std::vector<float>& opacities,
	const std::vector<Scale>& scales,
	const std::vector<Rot>& rot,
	const sibr::Vector3f& minn,
	const sibr::Vector3f& maxx)
{
	// Read all Gaussians at once (AoS)
	int count = 0;
	for (int i = 0; i < pos.size(); i++)
	{
		if (pos[i].x() < minn.x() || pos[i].y() < minn.y() || pos[i].z() < minn.z() ||
			pos[i].x() > maxx.x() || pos[i].y() > maxx.y() || pos[i].z() > maxx.z())
			continue;
		count++;
	}
	std::vector<RichPoint<3>> points(count);

	// Output number of Gaussians contained
	SIBR_LOG << "Saving " << count << " Gaussian splats" << std::endl;

	std::ofstream outfile(filename, std::ios_base::binary);

	outfile << "ply\nformat binary_little_endian 1.0\nelement vertex " << count << "\n";

	std::string props1[] = { "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"};
	std::string props2[] = { "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3" };

	for (auto s : props1)
		outfile << "property float " << s << std::endl;
	for (int i = 0; i < 45; i++)
		outfile << "property float f_rest_" << i << std::endl;
	for (auto s : props2)
		outfile << "property float " << s << std::endl;
	outfile << "end_header" << std::endl;

	count = 0;
	for (int i = 0; i < pos.size(); i++)
	{
		if (pos[i].x() < minn.x() || pos[i].y() < minn.y() || pos[i].z() < minn.z() ||
			pos[i].x() > maxx.x() || pos[i].y() > maxx.y() || pos[i].z() > maxx.z())
			continue;
		points[count].pos = pos[i];
		points[count].rot = rot[i];
		// Exponentiate scale
		for (int j = 0; j < 3; j++)
			points[count].scale.scale[j] = log(scales[i].scale[j]);
		// Activate alpha
		points[count].opacity = inverse_sigmoid(opacities[i]);
		points[count].shs.shs[0] = shs[i].shs[0];
		points[count].shs.shs[1] = shs[i].shs[1];
		points[count].shs.shs[2] = shs[i].shs[2];
		for (int j = 1; j < 16; j++)
		{
			points[count].shs.shs[(j - 1) + 3] = shs[i].shs[j * 3 + 0];
			points[count].shs.shs[(j - 1) + 18] = shs[i].shs[j * 3 + 1];
			points[count].shs.shs[(j - 1) + 33] = shs[i].shs[j * 3 + 2];
		}
		count++;
	}
	outfile.write((char*)points.data(), sizeof(RichPoint<3>) * points.size());
}

namespace sibr
{
	// A simple copy renderer class. Much like the original, but this one
	// reads from a buffer instead of a texture and blits the result to
	// a render target. 
	class BufferCopyRenderer
	{

	public:

		BufferCopyRenderer()
		{
			_shader.init("CopyShader",
				sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/copy.vert"),
				sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/copy.frag"));

			_flip.init(_shader, "flip");
			_width.init(_shader, "width");
			_height.init(_shader, "height");
		}

		void process(uint bufferID, IRenderTarget& dst, int width, int height, bool disableTest = true)
		{
			if (disableTest)
				glDisable(GL_DEPTH_TEST);
			else
				glEnable(GL_DEPTH_TEST);

			_shader.begin();
			_flip.send();
			_width.send();
			_height.send();

			dst.clear();
			dst.bind();

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferID);

			sibr::RenderUtility::renderScreenQuad();

			dst.unbind();
			_shader.end();
		}

		/** \return option to flip the texture when copying. */
		bool& flip() { return _flip.get(); }
		int& width() { return _width.get(); }
		int& height() { return _height.get(); }

	private:

		GLShader			_shader; 
		GLuniform<bool>		_flip = false; ///< Flip the texture when copying.
		GLuniform<int>		_width = 1000;
		GLuniform<int>		_height = 800;
	};
}

std::function<char* (size_t N)> resizeFunctional(void** ptr, size_t& S) {
	auto lambda = [ptr, &S](size_t N) {
		if (N > S)
		{
			if (*ptr)
				CUDA_SAFE_CALL(cudaFree(*ptr));
			CUDA_SAFE_CALL(cudaMalloc(ptr, 2 * N));
			S = 2 * N;
		}
		return reinterpret_cast<char*>(*ptr);
	};
	return lambda;
}

sibr::GaussianView::GaussianView(const sibr::BasicIBRScene::Ptr & ibrScene, uint render_w, uint render_h, const char* file, bool* messageRead, int sh_degree, bool white_bg, bool useInterop, int device) :
	_scene(ibrScene),
	_dontshow(messageRead),
	_sh_degree(sh_degree),
	sibr::ViewBase(render_w, render_h)
{
	int num_devices;
	CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceCount(&num_devices));
	_device = device;
	if (device >= num_devices)
	{
		if (num_devices == 0)
			SIBR_ERR << "No CUDA devices detected!";
		else
			SIBR_ERR << "Provided device index exceeds number of available CUDA devices!";
	}
	CUDA_SAFE_CALL_ALWAYS(cudaSetDevice(device));
	cudaDeviceProp prop;
	CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceProperties(&prop, device));
	if (prop.major < 7)
	{
		SIBR_ERR << "Sorry, need at least compute capability 7.0+!";
	}

	_pointbasedrenderer.reset(new PointBasedRenderer());
	_copyRenderer = new BufferCopyRenderer();
	_copyRenderer->flip() = true;
	_copyRenderer->width() = _resolution.x();
	_copyRenderer->height() = _resolution.y();

	std::vector<uint> imgs_ulr;
	const auto & cams = ibrScene->cameras()->inputCameras();
	for(size_t cid = 0; cid < cams.size(); ++cid) {
		if(cams[cid]->isActive()) {
			imgs_ulr.push_back(uint(cid));
		}
	}
	_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);

	// Load the PLY data (AoS) to the GPU (SoA)
	std::vector<Pos> pos;
	std::vector<Rot> rot;
	std::vector<Scale> scale;
	std::vector<float> opacity;
	std::vector<SHs<3>> shs;
	if (sh_degree == 0)
	{
		count = loadPly<0>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
	}
	else if (sh_degree == 1)
	{
		count = loadPly<1>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
	}
	else if (sh_degree == 2)
	{
		count = loadPly<2>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
	}
	else if (sh_degree == 3)
	{
		count = loadPly<3>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
	}

	_boxmin = _scenemin;
	_boxmax = _scenemax;

	int P = count;

	// Allocate and fill the GPU data
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&pos_cuda, sizeof(Pos) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos_cuda, pos.data(), sizeof(Pos) * P, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rot_cuda, sizeof(Rot) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot_cuda, rot.data(), sizeof(Rot) * P, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&shs_cuda, sizeof(SHs<3>) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs_cuda, shs.data(), sizeof(SHs<3>) * P, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&opacity_cuda, sizeof(float) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity_cuda, opacity.data(), sizeof(float) * P, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&scale_cuda, sizeof(Scale) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale_cuda, scale.data(), sizeof(Scale) * P, cudaMemcpyHostToDevice));

	// Create space for view parameters
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&view_cuda, sizeof(sibr::Matrix4f)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&proj_cuda, sizeof(sibr::Matrix4f)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&proj_inv_cuda, sizeof(sibr::Matrix4f)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&cam_pos_cuda, 3 * sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&background_cuda, 3 * sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rect_cuda, 2 * P * sizeof(int)));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&vismask_cuda, (((_resolution.x() + 15) / 16) * ((_resolution.y() + 15) / 16) + 31) / 32 * 4));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&vismasksum_cuda, ((_resolution.x() + 15) / 16 + 1) * ((_resolution.y() + 15) / 16 + 1) * 4));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc(&image_cuda_hier[0], (_resolution.x() / 2) * (_resolution.y() / 2) * 3 * sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc(&image_cuda_hier[1], (_resolution.x() / 2) * (_resolution.y() / 2) * 3 * sizeof(float)));
	

	float bg[3] = { white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f };
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_cuda, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));

	gData = new GaussianData(P, 
		(float*)pos.data(),
		(float*)rot.data(),
		(float*)scale.data(),
		opacity.data(),
		(float*)shs.data());

	_gaussianRenderer = new GaussianSurfaceRenderer();

	createImageBuffer();
	geomBufferFunc = resizeFunctional(&geomPtr, allocdGeom);
	binningBufferFunc = resizeFunctional(&binningPtr, allocdBinning);
	imgBufferFunc = resizeFunctional(&imgPtr, allocdImg);

	parseJSON();
}

void sibr::GaussianView::parseJSON()
{
	std::string json_path = _scene->data()->configPath();
	std::ifstream json_file(json_path, std::ios::in);

	splatting_settings = CudaRasterizer::SplattingSettings();

	// return if no config file is found - use default parameters (Vanilla 3DGS)
 	if (json_file.fail()) return;
	nlohmann::json js = nlohmann::json::parse(json_file);

	// get settings from config
	splatting_settings = js.get<CudaRasterizer::SplattingSettings>();

	// log splatting settings just in case
	if (splatting_settings.foveated_rendering)
		SIBR_LOG << "Using Foveated Rendering" << std::endl;
	else
		SIBR_LOG << "Not Using Foveated Rendering" << std::endl;

	// sanity checks
	if (CudaRasterizer::isInvalidSortMode(splatting_settings.sort_settings.sort_mode))
	{
		SIBR_LOG << "Invalid Sort Mode in " << json_path << " ("<< splatting_settings.sort_settings.sort_mode << "): continuing with default" << std::endl;
		splatting_settings.sort_settings.sort_mode = CudaRasterizer::SortMode::GLOBAL;
	}
	if (CudaRasterizer::isInvalidSortOrder(splatting_settings.sort_settings.sort_order))
	{
		SIBR_LOG << "Invalid Sort Order in " << json_path << " ("<< splatting_settings.sort_settings.sort_order << "): continuing with default" << std::endl;
		splatting_settings.sort_settings.sort_order = CudaRasterizer::GlobalSortOrder::VIEWSPACE_Z;
	}
	if (splatting_settings.sort_settings.hasModifiableWindowSize())
	{
		auto sort_mode = splatting_settings.sort_settings.sort_mode;
		auto test_function = [&](std::vector<int> vec, const char* what, int default_value, int& variable)
		{
			if (std::find(vec.begin(), vec.end(), variable) == vec.end())
			{
				SIBR_LOG << "Invalid " << what << " Size in " << json_path << " ("<< variable << "): continuing with default" << std::endl;
				variable = default_value;
			}
		};
		if (sort_mode == CudaRasterizer::SortMode::HIERARCHICAL)
		{
			test_function(CudaRasterizer::per_pixel_queue_sizes_hier, "Per-Pixel Queue", 4, splatting_settings.sort_settings.queue_sizes.per_pixel);
			test_function(CudaRasterizer::twobytwo_tile_queue_sizes, "2x2-Tile Queue", 8, splatting_settings.sort_settings.queue_sizes.tile_2x2);
		}
		if (sort_mode == CudaRasterizer::SortMode::PER_PIXEL_KBUFFER)
		{
			test_function(CudaRasterizer::per_pixel_queue_sizes, "Per-Pixel Queue", 1, splatting_settings.sort_settings.queue_sizes.per_pixel);
		}
	}
}

void sibr::GaussianView::setResolution(const Vector2i &size)
{
	if (size != getResolution())
	{
		SIBR_LOG << "Set resolution => " << size << std::endl;
		ViewBase::setResolution(size);
		destroyImageBuffer();
		createImageBuffer();

		_copyRenderer->width() = _resolution.x();
		_copyRenderer->height() = _resolution.y();
	}
}

void sibr::GaussianView::createImageBuffer()
{
	// Create GL buffer ready for CUDA/GL interop
	glCreateBuffers(1, &imageBuffer);
	glNamedBufferStorage(imageBuffer, _resolution.x() * _resolution.y() * 3 * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);

	if (_use_interop)
	{
		if (cudaPeekAtLastError() != cudaSuccess)
		{
			SIBR_ERR << "A CUDA error occurred in setup:" << cudaGetErrorString(cudaGetLastError()) << ". Please rerun in Debug to find the exact line!";
		}
		cudaGraphicsGLRegisterBuffer(&imageBufferCuda, imageBuffer, cudaGraphicsRegisterFlagsWriteDiscard);
		_use_interop &= (cudaGetLastError() == cudaSuccess);
	}
	if (!_use_interop)
	{
		fallback_bytes.resize(_resolution.x() * _resolution.y() * 3 * sizeof(float));
		cudaMalloc(&fallbackBufferCuda, fallback_bytes.size());
		_interop_failed = true;
	}
}

void sibr::GaussianView::destroyImageBuffer()
{
	if (_use_interop)
	{
		cudaGraphicsUnregisterResource(imageBufferCuda);
	}
	else
	{
		cudaFree(fallbackBufferCuda);
	}
	glDeleteBuffers(1, &imageBuffer);
	imageBuffer = 0;
}

void sibr::GaussianView::setScene(const sibr::BasicIBRScene::Ptr & newScene)
{
	_scene = newScene;

	// Tell the scene we are a priori using all active cameras.
	std::vector<uint> imgs_ulr;
	const auto & cams = newScene->cameras()->inputCameras();
	for (size_t cid = 0; cid < cams.size(); ++cid) {
		if (cams[cid]->isActive()) {
			imgs_ulr.push_back(uint(cid));
		}
	}
	_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);
}

void sibr::GaussianView::initMaskCuda(const sibr::Camera& eye, int w, int h, uint32_t* contributing_tiles)
{
	if (mask_cuda == nullptr)
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&mask_cuda, sizeof(float) * h * w));

	CUDA_SAFE_CALL_ALWAYS(cudaMemset(mask_cuda, 0, sizeof(float) * h * w));

	auto fov = eye.allFov();

	// obtain mask with the modified blendCuda
	CudaRasterizer::getAlphaMask(
		w / 2, h / 2,
		mask_cuda,
		w, h,
		w / (tan(fov.y()) - tan(fov.x())) * tan(-fov.x()) / 2 + 0.5,
		h / (tan(fov.w()) - tan(fov.z())) * tan(fov.w()) / 2 + 0.5,
		0.1f
	);

	int num_16x16_tiles{((w + 16 - 1) / 16) * ((h + 16 - 1) / 16)};
	int num_32x32_tiles{((w + 32 - 1) / 32) * ((h + 32 - 1) / 32)};

	if (rangemap_cuda == nullptr)
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rangemap_cuda, sizeof(uint32_t) * num_16x16_tiles));

	CUDA_SAFE_CALL_ALWAYS(cudaMemset(rangemap_cuda, 0, sizeof(uint32_t) *num_16x16_tiles));
	_num_tiles = CudaRasterizer::Rasterizer::computeTileBoundaries(rangemap_cuda, w, h, mask_cuda, contributing_tiles);
}

void sibr::GaussianView::onRenderIBR(sibr::IRenderTarget & dst, const sibr::Camera & eye)
{
	if (currMode == "Ellipsoids")
	{
		_gaussianRenderer->process(count, *gData, eye, dst, 0.2f);
	}
	else if (currMode == "Initial Points")
	{
		_pointbasedrenderer->process(_scene->proxies()->proxy(), eye, dst);
	}
	else
	{
		auto forward = [&](const sibr::Camera& eye, float* image_cuda_curr, int x, int y, bool mask, bool fullres) {
			// Convert view and projection to target coordinate system
			auto view_mat = eye.view();
			auto proj_mat = eye.viewproj();
			view_mat.row(1) *= -1;
			view_mat.row(2) *= -1;
			proj_mat.row(1) *= -1;

			// Compute additional view parameters
			float tan_fovy = tan(eye.fovy() * 0.5f);
			float tan_fovx = tan_fovy * eye.aspect();

			auto proj_inv_mat = sibr::Matrix4f(proj_mat.inverse());

			// Copy frame-dependent data to GPU
			CUDA_SAFE_CALL(cudaMemcpy(view_cuda, view_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(proj_cuda, proj_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(proj_inv_cuda, proj_inv_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(cam_pos_cuda, &eye.position(), sizeof(float) * 3, cudaMemcpyHostToDevice));
			
			if (mask)
			{				
				auto uploadVisibilityMask = [&](bool fullres)
				{
					const int tileSize = fullres ? 16 : 32;
					const int tileW = (_resolution.x() + tileSize - 1) / tileSize;
					const int tileH = (_resolution.y() + tileSize - 1) / tileSize;

					const int mask_size = (tileW * tileH + 31) / 32 * 4;
					const int mask_sum_size = (tileW + 1) * (tileH + 1) * 4;
					
					CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(vismask_cuda, eye.visibilityMask(fullres).first, mask_size, cudaMemcpyHostToDevice));
					CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(vismasksum_cuda, eye.visibilityMask(fullres).second, mask_sum_size, cudaMemcpyHostToDevice));
				};
				
				uploadVisibilityMask(fullres);
				
				initMaskCuda(eye, x, y, vismask_cuda);
			}

			// Rasterize
			float* boxmin = _cropping ? (float*)&_boxmin : nullptr;
			float* boxmax = _cropping ? (float*)&_boxmax : nullptr;
			CudaRasterizer::Rasterizer::forward(
				geomBufferFunc,
				binningBufferFunc,
				imgBufferFunc,
				count, _sh_degree, 16, _num_tiles,
				background_cuda,
				x, y,
				splatting_settings,
				debugMode,
				pos_cuda,
				shs_cuda,
				nullptr,
				opacity_cuda,
				scale_cuda,
				_scalingModifier,
				rot_cuda,
				nullptr,
				view_cuda,
				proj_cuda,
				proj_inv_cuda,
				cam_pos_cuda,
				rangemap_cuda,
				mask_cuda,
				tan_fovx,
				tan_fovy,
				false,
				image_cuda_curr,
				nullptr,
				true,
				mask ? vismask_cuda : nullptr,
				mask ? vismasksum_cuda : nullptr
			);
		};

		float* image_cuda = nullptr;
		if (!_interop_failed)
		{
			// Map OpenGL buffer resource for use with CUDA
			size_t bytes;
			CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &imageBufferCuda));
			CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&image_cuda, &bytes, imageBufferCuda));
		}
		else
		{
			image_cuda = fallbackBufferCuda;
		}

		int w = _resolution.x();
		int h = _resolution.y();

		Camera eye2 = eye;
		if (eye.isSym() && m_visibilityMask_fullres.first == nullptr)
		{
			auto loadVisibilityMask = [&](const std::string filename, const int tileSize) {

				const int tileW = (w + tileSize - 1) / tileSize;
				const int tileH = (h + tileSize - 1) / tileSize;

				const int mask_size = (tileW * tileH + 31) / 32 * 4;
				uint32_t* mask = (uint32_t*) malloc(mask_size);

				std::string mask_folder = "masks_2064x2272";
				std::string mask_filename = mask_folder + "/" + filename + ".dat";
				std::string mask_sum_filename = mask_folder + "/" + filename + "_sum.dat";
				
				std::ifstream infile{mask_filename, std::ios::binary};
				infile.read((char *) mask, mask_size);
				infile.close();

				const int mask_sum_size = (tileW + 1) * (tileH + 1) * 4;
				uint32_t* mask_sum = (uint32_t*) malloc(mask_sum_size);

				std::ifstream infile_sum{mask_sum_filename, std::ios::binary};
				infile_sum.read((char *) mask_sum, mask_sum_size);
				infile_sum.close();

				return std::pair<uint32_t*, uint32_t*>(mask, mask_sum);
			};
			
			m_visibilityMask_fullres = loadVisibilityMask("fullres_mask_right", 16);
			m_visibilityMask_halfres = loadVisibilityMask("halfres_mask_right", 32);
		}

		if (eye.isSym())
		{
			float fovv = eye.fovy();
			float fovh = fovv * eye2.aspect();
			eye2.setAllFov({ -fovh / 2, fovh / 2, -fovv / 2, fovv / 2 });

			if (m_visibilityMask_fullres.first != nullptr)
			{
				eye2.setVisibilityMaskFullres(m_visibilityMask_fullres);
				eye2.setVisibilityMaskHalfres(m_visibilityMask_halfres);
			}
		}
		
		static CudaRasterizer::Timer timer({ "Low", "High", "Processing" }, 25);
		
		timer.setActive(true);
		timer();
		
		std::vector<std::pair<std::string, float>> timings;
		if (true)
		{
			timer();
			forward(eye2, image_cuda, w, h, true, !splatting_settings.foveated_rendering);
			timer();
		}
		else
		{
			// Low-res
			forward(eye2, image_cuda_hier[0], w / 2, h / 2, true, splatting_settings.foveated_rendering);

			timer();

			// High-res
			auto fov = eye2.allFov();
			// eye2.fovy(atan(tan((fov.w() - fov.z()) / 2) * 0.5f) * 2);
			eye2.fovy(atan(tan(fov.w()) * 0.5f) - atan(tan(fov.z()) * 0.5f));
			eye2.setAllFov({atan(tan(fov.x()) * 0.5f), atan(tan(fov.y()) * 0.5f), atan(tan(fov.z()) * 0.5f), atan(tan(fov.w()) * 0.5f)});
			// fov = eye2.allFov();
			forward(eye2, image_cuda_hier[1], w / 2, h / 2, false, true);

			timer();

			// Upsample
			{
				NppiSize srcSize = { w / 2, h / 2 };
				const float* pSrc[] = {
					image_cuda_hier[0],
					image_cuda_hier[0] + srcSize.width * srcSize.height,
					image_cuda_hier[0] + 2 * srcSize.width * srcSize.height
				};
				int srcStep = srcSize.width * sizeof(float);
				NppiRect srcRect = { 0, 0, srcSize.width, srcSize.height };
				NppiSize dstSize = { w, h };
				float* pDst[] = {
					image_cuda,
					image_cuda + w * h,
					image_cuda + 2 * w * h
				};
				int dstStep = w * sizeof(float);
				NppiRect dstRect = { 0, 0, w / 2 * 2, h / 2 * 2 };
				auto status = nppiResize_32f_P3R(
					pSrc, srcStep, srcSize, srcRect,
					pDst, dstStep, dstSize, dstRect,
					NPPI_INTER_CUBIC
					// NPPI_INTER_LANCZOS
				);
				if (status != NPP_SUCCESS)
				{
					SIBR_ERR << "NPP error: " << status << std::endl;
				}
			}

			// Move high-res image

			CudaRasterizer::blend(
				image_cuda_hier[1],
				w / 2, h / 2,
				image_cuda,
				w, h,
				w / (tan(fov.y()) - tan(fov.x())) * tan(-fov.x()) / 2 + 0.5,
				h / (tan(fov.w()) - tan(fov.z())) * tan(fov.w()) / 2 + 0.5,
				0.1f
			);
			// NppiSize srcSize = { w / 2, h / 2 };
			// const float* pSrc[] = {
			// 	image_cuda_hier[1],
			// 	image_cuda_hier[1] + srcSize.width * srcSize.height,
			// 	image_cuda_hier[1] + 2 * srcSize.width * srcSize.height
			// };
			// int srcStep = srcSize.width * sizeof(float);
			// NppiRect srcRect = { 0, 0, srcSize.width, srcSize.height };
			// NppiSize dstSize = { w, h };
			// float* pDst[] = {
			// 	image_cuda,
			// 	image_cuda + w * h,
			// 	image_cuda + 2 * w * h
			// };
			// int dstStep = w * sizeof(float);
			// int cx = w / (tan(fov.y()) - tan(fov.x())) * tan(-fov.x()) / 2;
			// int cy = h / (tan(fov.w()) - tan(fov.z())) * tan(fov.w()) / 2;
			// NppiRect dstRect = { cx, cy, w / 2, h / 2 };
			// auto status = nppiResize_32f_P3R(
			// 	pSrc, srcStep, srcSize, srcRect,
			// 	pDst, dstStep, dstSize, dstRect,
			// 	NPPI_INTER_NN
			// );
			// if (status != NPP_SUCCESS)
			// {
			// 	SIBR_ERR << "NPP error: " << status << std::endl;
			// }
		}

		timer();
		timer.syncAddReport(timings);
	
		

		if (timings.size() > 0)
		{
			std::stringstream ss;
			ss << "Timings: \n";
			for (auto const& x : timings)
				ss << " - " << x.first << ": " << x.second * 2 << "ms\n";
			std::cout << ss.str() << std::endl;
		}

		// static int frame_counter = 0;
		// if (++frame_counter % 2000 == 1999)
		// {
		// 	float *img = new float[w * h * 3], *img2 = new float[w * h * 3];
		// 	CUDA_SAFE_CALL(cudaMemcpy(img, image_cuda, w * h * 3 * sizeof(float), cudaMemcpyDeviceToHost));
		// 	for (int i = 0; i < w; i++) for (int j = 0; j < h; j++) for (int k = 0; k < 3; k++)
		// 		img2[j * w * 3 + i * 3 + k] = img[(2 - k) * w * h + j * w + i] * 255;
		// 	std::string filename = "stop-optimal-fr.png";
		// 	cv::imwrite(filename, cv::Mat(h, w, CV_32FC3, img2));

		// 	w /= 2, h /= 2;
		// 	CUDA_SAFE_CALL(cudaMemcpy(img, image_cuda_hier[0], w * h * 3 * sizeof(float), cudaMemcpyDeviceToHost));
		// 	for (int i = 0; i < w; i++) for (int j = 0; j < h; j++) for (int k = 0; k < 3; k++)
		// 		img2[j * w * 3 + i * 3 + k] = img[(2 - k) * w * h + j * w + i] * 255;
		// 	filename = "stop-optimal-fr-low.png";
		// 	cv::imwrite(filename, cv::Mat(h, w, CV_32FC3, img2));

		// 	CUDA_SAFE_CALL(cudaMemcpy(img, image_cuda_hier[1], w * h * 3 * sizeof(float), cudaMemcpyDeviceToHost));
		// 	for (int i = 0; i < w; i++) for (int j = 0; j < h; j++) for (int k = 0; k < 3; k++)
		// 		img2[j * w * 3 + i * 3 + k] = img[(2 - k) * w * h + j * w + i] * 255;
		// 	filename = "stop-optimal-fr-high.png";
		// 	cv::imwrite(filename, cv::Mat(h, w, CV_32FC3, img2));

		// 	delete[] img;
		// 	delete[] img2;
		// 	printf("success\n");
		// }

		if (!_interop_failed)
		{
			// Unmap OpenGL resource for use with OpenGL
			CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &imageBufferCuda));
		}
		else
		{
			CUDA_SAFE_CALL(cudaMemcpy(fallback_bytes.data(), fallbackBufferCuda, fallback_bytes.size(), cudaMemcpyDeviceToHost));
			glNamedBufferSubData(imageBuffer, 0, fallback_bytes.size(), fallback_bytes.data());
		}
		// Copy image contents to framebuffer
		_copyRenderer->process(imageBuffer, dst, _resolution.x(), _resolution.y());
	}

	if (cudaPeekAtLastError() != cudaSuccess)
	{
		SIBR_ERR << "A CUDA error occurred during rendering:" << cudaGetErrorString(cudaGetLastError()) << ". Please rerun in Debug to find the exact line!";
	}
}

void sibr::GaussianView::onUpdate(Input & input)
{
	if (input.mouseButton().isReleased(Mouse::Code::Right))
	{
		updateDebugPixelLocation = !updateDebugPixelLocation;
	}
}

void sibr::GaussianView::onGUI()
{
	if (updateDebugPixelLocation && updateWithMouse)
	{
		auto viewWindowPos = sibr::getImGuiWindowPosition("Point view");
		auto mousePos = ImGui::GetMousePos();
		debugMode.debugPixel[0] = mousePos.x - viewWindowPos.x;
		debugMode.debugPixel[1] = mousePos.y - viewWindowPos.y;
	}

	// Generate and update UI elements
	const std::string guiName = "3D Gaussians";
	if (ImGui::Begin(guiName.c_str())) 
	{
		if (ImGui::BeginCombo("Render Mode", currMode.c_str()))
		{
			if (ImGui::Selectable("Splats"))
				currMode = "Splats";
			if (ImGui::Selectable("Initial Points"))
				currMode = "Initial Points";
			if (ImGui::Selectable("Ellipsoids"))
				currMode = "Ellipsoids";
			ImGui::EndCombo();
		}
	}
	if (currMode == "Splats")
	{
		ImGui::SliderFloat("Scaling Modifier", &_scalingModifier, 0.001f, 1.0f);

		ImGui::Checkbox("Crop Box", &_cropping);
		if (_cropping)
		{
			ImGui::SliderFloat("Box Min X", &_boxmin.x(), _scenemin.x(), _scenemax.x());
			ImGui::SliderFloat("Box Min Y", &_boxmin.y(), _scenemin.y(), _scenemax.y());
			ImGui::SliderFloat("Box Min Z", &_boxmin.z(), _scenemin.z(), _scenemax.z());
			ImGui::SliderFloat("Box Max X", &_boxmax.x(), _scenemin.x(), _scenemax.x());
			ImGui::SliderFloat("Box Max Y", &_boxmax.y(), _scenemin.y(), _scenemax.y());
			ImGui::SliderFloat("Box Max Z", &_boxmax.z(), _scenemin.z(), _scenemax.z());
			ImGui::InputText("File", _buff, 512);
			if (ImGui::Button("Save"))
			{
				std::vector<Pos> pos(count);
				std::vector<Rot> rot(count);
				std::vector<float> opacity(count);
				std::vector<SHs<3>> shs(count);
				std::vector<Scale> scale(count);
				CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos.data(), pos_cuda, sizeof(Pos) * count, cudaMemcpyDeviceToHost));
				CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot.data(), rot_cuda, sizeof(Rot) * count, cudaMemcpyDeviceToHost));
				CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity.data(), opacity_cuda, sizeof(float) * count, cudaMemcpyDeviceToHost));
				CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs.data(), shs_cuda, sizeof(SHs<3>) * count, cudaMemcpyDeviceToHost));
				CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale.data(), scale_cuda, sizeof(Scale) * count, cudaMemcpyDeviceToHost));
				savePly(_buff, pos, shs, opacity, scale, rot, _boxmin, _boxmax);
			}
		}

		if (ImGui::CollapsingHeader("StopThePop", ImGuiTreeNodeFlags_DefaultOpen))
		{
			if (ImGui::BeginCombo("Sort Order", toString(splatting_settings.sort_settings.sort_order).c_str()))
			{
				if (ImGui::Selectable("VIEWSPACE_Z"))
					splatting_settings.sort_settings.sort_order = CudaRasterizer::GlobalSortOrder::VIEWSPACE_Z;
				if (ImGui::Selectable("DISTANCE"))
					splatting_settings.sort_settings.sort_order = CudaRasterizer::GlobalSortOrder::DISTANCE;
				if (ImGui::Selectable("PER_TILE_DEPTH_CENTER"))
					splatting_settings.sort_settings.sort_order = CudaRasterizer::GlobalSortOrder::PER_TILE_DEPTH_CENTER;
				if (ImGui::Selectable("PER_TILE_DEPTH_MAXPOS"))
					splatting_settings.sort_settings.sort_order = CudaRasterizer::GlobalSortOrder::PER_TILE_DEPTH_MAXPOS;
				ImGui::EndCombo();
			}

			if (ImGui::BeginCombo("Sort Mode", toString(splatting_settings.sort_settings.sort_mode).c_str()))
			{
				if (ImGui::Selectable("GLOBAL"))
					splatting_settings.sort_settings.sort_mode = CudaRasterizer::SortMode::GLOBAL;
				if (ImGui::Selectable("FULL SORT"))
					splatting_settings.sort_settings.sort_mode = CudaRasterizer::SortMode::PER_PIXEL_FULL;
				if (ImGui::Selectable("KBUFFER"))
					splatting_settings.sort_settings.sort_mode = CudaRasterizer::SortMode::PER_PIXEL_KBUFFER;
				if (ImGui::Selectable("HIERARCHICAL"))
					splatting_settings.sort_settings.sort_mode = CudaRasterizer::SortMode::HIERARCHICAL;
				ImGui::EndCombo();
			}

			if (splatting_settings.sort_settings.sort_mode == CudaRasterizer::SortMode::PER_PIXEL_KBUFFER)
			{
				if (ImGui::BeginCombo("Per-Pixel Queue Size", std::to_string(splatting_settings.sort_settings.queue_sizes.per_pixel).c_str()))
				{
					for (auto z : CudaRasterizer::per_pixel_queue_sizes){
						if (ImGui::Selectable(std::to_string(z).c_str()))
							splatting_settings.sort_settings.queue_sizes.per_pixel = z;
					}
					ImGui::EndCombo();
				}
			}

			if (splatting_settings.sort_settings.sort_mode == CudaRasterizer::SortMode::HIERARCHICAL)
			{
				if (ImGui::BeginCombo("Per-Pixel Queue Size", std::to_string(splatting_settings.sort_settings.queue_sizes.per_pixel).c_str()))
				{
					for (auto z : CudaRasterizer::per_pixel_queue_sizes_hier){
						if (ImGui::Selectable(std::to_string(z).c_str()))
							splatting_settings.sort_settings.queue_sizes.per_pixel = z;
					}
					ImGui::EndCombo();
				}

				if (ImGui::BeginCombo("2x2 Tile Queue Size", std::to_string(splatting_settings.sort_settings.queue_sizes.tile_2x2).c_str()))
				{
					for (auto z : CudaRasterizer::twobytwo_tile_queue_sizes){
						if (ImGui::Selectable(std::to_string(z).c_str()))
							splatting_settings.sort_settings.queue_sizes.tile_2x2 = z;
					}
					ImGui::EndCombo();
				}

				ImGui::Checkbox("Hier. 4x4 Tile Culling", &splatting_settings.culling_settings.hierarchical_4x4_culling);
			}


			ImGui::Checkbox("Foveated Rendering", &splatting_settings.foveated_rendering);
			ImGui::Checkbox("Rect Culling", &splatting_settings.culling_settings.rect_bounding);
			ImGui::Checkbox("Opacity Culling", &splatting_settings.culling_settings.tight_opacity_bounding);
			ImGui::Checkbox("Tile-based Culling", &splatting_settings.culling_settings.tile_based_culling);
			ImGui::Checkbox("Load Balancing", &splatting_settings.load_balancing);
			ImGui::Checkbox("Optimal Projection", &splatting_settings.optimal_projection);
			ImGui::Checkbox("Proper EWA Scaling", &splatting_settings.proper_ewa_scaling);
		}


		if (ImGui::CollapsingHeader("Debug", ImGuiTreeNodeFlags_DefaultOpen))
		{
			if (ImGui::BeginCombo("Debug Visualization", toString(debugMode.type).data()))
			{
				if (ImGui::Selectable("Disabled"))
					debugMode.type = DebugVisualization::Disabled;
				if (ImGui::Selectable("Sort Error: Opacity Weighted"))
					debugMode.type = DebugVisualization::SortErrorOpacity;
				if (ImGui::Selectable("Sort Error: Distance Weighted"))
					debugMode.type = DebugVisualization::SortErrorDistance;
				if (ImGui::Selectable("Gaussian Count Per Tile"))
					debugMode.type = DebugVisualization::GaussianCountPerTile;
				if (ImGui::Selectable("Gaussian Count Per Pixel"))
					debugMode.type = DebugVisualization::GaussianCountPerPixel;
				if (ImGui::Selectable("Depth"))
					debugMode.type = DebugVisualization::Depth;
				if (ImGui::Selectable("Transmittance"))
					debugMode.type = DebugVisualization::Transmittance;
				ImGui::EndCombo();
			}

			if (debugMode.type != DebugVisualization::Disabled)
			{
				ImGui::Checkbox("Manual Normalization", &debugMode.debug_normalize);
				if (debugMode.debug_normalize)
				{
					ImGui::InputFloat2("Normalize Min/Max", debugMode.minMax);
				}
				ImGui::Checkbox("Input With Mouse", &updateWithMouse);
				ImGui::InputInt2("Mouse Debug Pos", debugMode.debugPixel);
			}

			ImGui::Checkbox("Timing", &debugMode.timing_enabled);
			if (debugMode.timing_enabled)
				ImGui::Text("%s", (char*) debugMode.timings_text.c_str());
			else
				debugMode.timings_text = "";
		}
		
	}
	ImGui::End();

	if(!*_dontshow && !accepted && _interop_failed)
		ImGui::OpenPopup("Error Using Interop");

	if (!*_dontshow && !accepted && _interop_failed && ImGui::BeginPopupModal("Error Using Interop", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
		ImGui::SetItemDefaultFocus();
		ImGui::SetWindowFontScale(2.0f);
		ImGui::Text("This application tries to use CUDA/OpenGL interop.\n"\
			" It did NOT work for your current configuration.\n"\
			" For highest performance, OpenGL and CUDA must run on the same\n"\
			" GPU on an OS that supports interop.You can try to pass a\n"\
			" non-zero index via --device on a multi-GPU system, and/or try\n" \
			" attaching the monitors to the main CUDA card.\n"\
			" On a laptop with one integrated and one dedicated GPU, you can try\n"\
			" to set the preferred GPU via your operating system.\n\n"\
			" FALLING BACK TO SLOWER RENDERING WITH CPU ROUNDTRIP\n");

		ImGui::Separator();

		if (ImGui::Button("  OK  ")) {
			ImGui::CloseCurrentPopup();
			accepted = true;
		}
		ImGui::SameLine();
		ImGui::Checkbox("Don't show this message again", _dontshow);
		ImGui::EndPopup();
	}
}

sibr::GaussianView::~GaussianView()
{
	// Cleanup
	cudaFree(pos_cuda);
	cudaFree(rot_cuda);
	cudaFree(scale_cuda);
	cudaFree(opacity_cuda);
	cudaFree(shs_cuda);

	cudaFree(view_cuda);
	cudaFree(proj_cuda);
	cudaFree(proj_inv_cuda);
	cudaFree(cam_pos_cuda);
	cudaFree(background_cuda);
	cudaFree(rect_cuda);

	for (int i = 0; i < 2; i++)
		cudaFree(image_cuda_hier[i]);

	if (!_interop_failed)
	{
		cudaGraphicsUnregisterResource(imageBufferCuda);
	}
	else
	{
		cudaFree(fallbackBufferCuda);
	}
	glDeleteBuffers(1, &imageBuffer);

	if (geomPtr)
		cudaFree(geomPtr);
	if (binningPtr)
		cudaFree(binningPtr);
	if (imgPtr)
		cudaFree(imgPtr);

	delete _copyRenderer;
}
