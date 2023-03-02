#include "polyscope/polyscope.h"
#include "polyscope/pick.h"

#include <igl/invert_diag.h>
#include <igl/per_vertex_normals.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/boundary_loop.h>
#include <igl/doublearea.h>
#include <igl/loop.h>
#include <igl/file_dialog_open.h>
#include <igl/file_dialog_save.h>
#include <igl/cotmatrix_entries.h>
#include <igl/cylinder.h>
#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/view.h"

#include <iostream>
#include <filesystem>
#include <utility>

#include "../../MeshLib/Mesh.h"
#include "../../json.hpp"
#include "../../CommonTools.h"
#include "../../LoadSaveIO.h"
#include "../../PaintGeometry.h"

#include "../../KnoppelStripePatterns.h"
#include "../../Upsampling/BaseLoop.h"
#include "../../Upsampling/ComplexLoop.h"
#include "../../Decomposition/CWFDecomposition.h"


#include <CLI/CLI.hpp>

MatrixX initUpPos, initWrinkledPos, optUpPos, optWrinkledPos, restPos, restWrinkledPos, wrinkledPos;
Eigen::MatrixXi upFaces, restFaces, restWrinkledFaces, wrinkledFaces;
Mesh restMesh, restWrinkledMesh, wrinkledMesh;

CWF initCWF, optCWF;

VectorX upInitAmp, upInitOmega, upInitPhase, upOptAmp, upOptOmega, upOptPhase;
ComplexVectorX upInitZvals, upOptZvals;

MatrixX initFaceOmega, optFaceOmega;
MatrixX initUpFaceOmega, optUpFaceOmega;

std::string workingFolder = "";
int upsampleTimes = 0;
double wrinkleAmpRatio = 1.0;

float vecratio = 0.01;
bool isFixedBnd = true;
double ampMax = 1;
bool isLoadCWF = false;

double youngs = 1e7;
double thickness = 1e-4;
double poisson = 0.44;

PaintGeometry mPaint;
CWFDecomposition decompModel;

int updateViewHelper(
		const MatrixX& basePos, const Eigen::MatrixXi& baseFaces,
		const MatrixX& upsampledPos, const Eigen::MatrixXi& upsampledFaces, const MatrixX& wrinkledPos,
		const VectorX& baseAmplitude, const MatrixX& baseFaceOmega,
		const VectorX& upsampledAmplitude, const VectorX& upsampledPhase, const MatrixX& upsampledFaceOmega,
		double shiftx, double shifty, std::string meshPreffix = "ref_", bool isFirstTime = true)
{
	int curShift = 0;
	if(isFirstTime)
	{
		polyscope::registerSurfaceMesh(meshPreffix + "base mesh", basePos, baseFaces);
		polyscope::getSurfaceMesh(meshPreffix + "base mesh")->translate({ curShift * shiftx, shifty, 0 });
	}
	else
	{
		polyscope::getSurfaceMesh(meshPreffix + "base mesh")->updateVertexPositions(basePos);
	}

	auto baseOmegaPatterns = polyscope::getSurfaceMesh(meshPreffix + "base mesh")->addFaceVectorQuantity("frequency field", vecratio * baseFaceOmega, polyscope::VectorType::AMBIENT);
	baseOmegaPatterns->setEnabled(true);
	auto baseAmpPatterns = polyscope::getSurfaceMesh(meshPreffix + "base mesh")->addVertexScalarQuantity("opt amplitude", baseAmplitude);
	baseAmpPatterns->setMapRange({ 0, ampMax });
	baseAmpPatterns->setColorMap("coolwarm");
	baseAmpPatterns->setEnabled(true);

	curShift++;
	// phase pattern
	if(isFirstTime)
	{
		polyscope::registerSurfaceMesh(meshPreffix + "upsampled phase mesh", upsampledPos, upsampledFaces);
		polyscope::getSurfaceMesh(meshPreffix + "upsampled phase mesh")->translate({ curShift * shiftx, shifty, 0 });
	}
	else
	{
		polyscope::getSurfaceMesh(meshPreffix + "upsampled phase mesh")->updateVertexPositions(upsampledPos);
	}

	mPaint.setNormalization(false);
	MatrixX phaseColor = mPaint.paintPhi(upsampledPhase);
	auto phasePatterns = polyscope::getSurfaceMesh(meshPreffix + "upsampled phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);
	phasePatterns->setEnabled(true);

	polyscope::getSurfaceMesh(meshPreffix + "upsampled phase mesh")->addFaceVectorQuantity("subdivided frequency field", vecratio * upsampledFaceOmega, polyscope::VectorType::AMBIENT);
	curShift++;

	// amp pattern
	if(isFirstTime)
	{
		polyscope::registerSurfaceMesh(meshPreffix + "upsampled ampliude mesh", upsampledPos, upsampledFaces);
		polyscope::getSurfaceMesh(meshPreffix + "upsampled ampliude mesh")->translate({ curShift * shiftx, shifty, 0 });
	}
	else
	{
		polyscope::getSurfaceMesh(meshPreffix + "upsampled ampliude mesh")->updateVertexPositions(upsampledPos);
	}

	auto ampPatterns = polyscope::getSurfaceMesh(meshPreffix + "upsampled ampliude mesh")->addVertexScalarQuantity("vertex amplitude", upsampledAmplitude);
	ampPatterns->setMapRange({ 0, ampMax });
	ampPatterns->setColorMap("coolwarm");
	ampPatterns->setEnabled(true);

	curShift++;

	// wrinkle mesh
	if(isFirstTime)
	{
		polyscope::registerSurfaceMesh(meshPreffix + "wrinkled mesh", wrinkledPos, upsampledFaces);
		polyscope::getSurfaceMesh(meshPreffix + "wrinkled mesh")->setSurfaceColor({ 80 / 255.0, 122 / 255.0, 91 / 255.0 });
		polyscope::getSurfaceMesh(meshPreffix + "wrinkled mesh")->translate({ curShift * shiftx, shifty, 0 });
	}

	else
		polyscope::getSurfaceMesh(meshPreffix + "wrinkled mesh")->updateVertexPositions(wrinkledPos);
	return curShift;
}

void updateView(bool isFirstTime = true)
{
	double shiftx = 1.5 * (initCWF._mesh.GetPos().col(0).maxCoeff() - initCWF._mesh.GetPos().col(0).minCoeff());
	double shifty = 1.5 * (initCWF._mesh.GetPos().col(1).maxCoeff() - initCWF._mesh.GetPos().col(1).minCoeff());

	ampMax = std::max(std::max(std::max(upInitAmp.maxCoeff(), upOptAmp.maxCoeff()), initCWF._amp.maxCoeff()), optCWF._amp.maxCoeff());

	int curShift = updateViewHelper(initCWF._mesh.GetPos(), initCWF._mesh.GetFace(), initUpPos, upFaces, initWrinkledPos, initCWF._amp, initFaceOmega, upInitAmp, upInitPhase, initUpFaceOmega, shiftx, 0, "int ", isFirstTime);

	curShift = updateViewHelper(optCWF._mesh.GetPos(), optCWF._mesh.GetFace(), optUpPos, upFaces, optWrinkledPos, optCWF._amp, optFaceOmega, upOptAmp, upOptPhase, optUpFaceOmega, shiftx, shifty, "opt ", isFirstTime);

	polyscope::registerSurfaceMesh("reference wrinkled mesh", wrinkledPos, wrinkledFaces);
	polyscope::getSurfaceMesh("reference wrinkled mesh")->setSurfaceColor({ 80 / 255.0, 122 / 255.0, 91 / 255.0 });
	polyscope::getSurfaceMesh("reference wrinkled mesh")->translate({ curShift * shiftx, 2 * shifty, 0 });

	std::cout << "dist: " << (initWrinkledPos - wrinkledPos).norm() << ", after optimization: " << (optWrinkledPos - wrinkledPos).norm() << std::endl;
}

void subdivideMeshHelper(
	const CWF& cwf,
	const bool isFixedBoundary, const int upLevel, const double wrinkleAmpRatio,
	MatrixX& faceOmega,
	MatrixX& upsampledV, Eigen::MatrixXi& upsampledF, MatrixX& wrinkledPos, Eigen::MatrixXi& wrinkledFace,
	VectorX& upsampledOmega, ComplexVectorX& upsampledZvals,
	MatrixX& upsampledFaceOmega, VectorX& upsampledPhase, VectorX& upsampledAmp
	)
{
    CWF upcwf;
	std::shared_ptr<BaseLoop> subOp;
	subOp = std::make_shared<ComplexLoop>();
    subOp->SetBndFixFlag(isFixedBoundary);
	subOp->CWFSubdivide(cwf, upcwf, upLevel);

	rescaleZvals(upcwf._zvals, upcwf._amp, upsampledZvals);
    upcwf._mesh.GetPos(upsampledV);
    upcwf._mesh.GetFace(upsampledF);
	getWrinkledMesh(upsampledV, upsampledF, upsampledZvals, wrinkledPos, wrinkleAmpRatio, false);
	wrinkledFace = upsampledF;

	faceOmega = intrinsicEdgeVec2FaceVec(cwf._omega, cwf._mesh);
	upsampledFaceOmega = intrinsicEdgeVec2FaceVec(upcwf._omega, upcwf._mesh);
	

	upsampledAmp.setZero(upsampledZvals.size());
	upsampledPhase.setZero(upsampledZvals.size());

	for(int i = 0; i < upsampledZvals.size(); i++)
	{
		upsampledAmp[i] = std::abs(upsampledZvals[i]);
		upsampledPhase[i] = std::arg(upsampledZvals[i]);
	}
}

void subdivideMesh(bool isSubdivInit = true)
{
	if(isSubdivInit)
		subdivideMeshHelper(initCWF, isFixedBnd, upsampleTimes, wrinkleAmpRatio, initFaceOmega, initUpPos, upFaces, initWrinkledPos, upFaces, upInitOmega, upInitZvals, initUpFaceOmega, upInitPhase, upInitAmp);
	else
        subdivideMeshHelper(optCWF, isFixedBnd, upsampleTimes, wrinkleAmpRatio, optFaceOmega, optUpPos, upFaces, optWrinkledPos, upFaces, upOptOmega, upOptZvals, optUpFaceOmega, upOptPhase, upOptAmp);
}


bool loadProblem(std::string loadFileName = "")
{
    if(loadFileName == "")
        loadFileName = igl::file_dialog_open();

    std::cout << "load file in: " << loadFileName << std::endl;
    using json = nlohmann::json;
    std::ifstream inputJson(loadFileName);
    if (!inputJson) {
        std::cerr << "missing json file in " << loadFileName << std::endl;
        return false;
    }

    std::string filePath = loadFileName;
    std::replace(filePath.begin(), filePath.end(), '\\', '/'); // handle the backslash issue for windows
    int id = filePath.rfind("/");
    workingFolder = filePath.substr(0, id + 1);
    std::cout << "working folder: " << workingFolder << std::endl;

    json jval;
    inputJson >> jval;

    // base mesh
    std::string meshFile = jval["base_mesh"];
    upsampleTimes = jval["upsampled_times"];
    meshFile = workingFolder + meshFile;

    MatrixX initBasePos;
    Eigen::MatrixXi baseFaces;

    if(!igl::readOBJ(meshFile, initBasePos, baseFaces))
    {
        std::cerr << "Failed to load base mesh!" << std::endl;
        exit(EXIT_FAILURE);
    }
    initCWF._mesh.Populate(initBasePos, baseFaces);

    // rest mesh
    meshFile = jval["rest_mesh"];
    meshFile = workingFolder + meshFile;
    if(!igl::readOBJ(meshFile, restPos, restFaces))
    {
        std::cerr << "Failed to load rest mesh!" << std::endl;
        exit(EXIT_FAILURE);
    }
    restMesh.Populate(restPos, restFaces);


    // wrinkled mesh
    meshFile = jval["wrinkled_mesh"];
    meshFile = workingFolder + meshFile;
    if(!igl::readOBJ(meshFile, wrinkledPos, wrinkledFaces))
    {
        std::cerr << "Failed to load wrinkled mesh!" << std::endl;
        exit(EXIT_FAILURE);
    }
    wrinkledMesh.Populate(wrinkledPos, wrinkledFaces);

    // wrinkled mesh
    meshFile = jval["rest_wrinkled_mesh"];
    meshFile = workingFolder + meshFile;
    if(!igl::readOBJ(meshFile, restWrinkledPos, restWrinkledFaces))
    {
        std::cerr << "Failed to load rest wrinkled mesh!" << std::endl;
        exit(EXIT_FAILURE);
    }
    restWrinkledMesh.Populate(restWrinkledPos, restWrinkledFaces);


    // amp, frequency and phase
    int nedges = initCWF._mesh.GetEdgeCount();
    int nverts = initBasePos.rows();

    std::string initAmpPath = jval["init_amp"];
    std::string initOmegaPath = jval["init_omega"];
    std::string initZValsPath = "zvals.txt";
    if (jval.contains(std::string_view{ "init_zvals" }))
    {
        initZValsPath = jval["init_zvals"];
    }

    if (jval.contains(std::string_view{ "wrinkle_amp_ratio" }))
    {
        wrinkleAmpRatio = jval["wrinkle_amp_ratio"];
    }
    std::cout << "wrinkle amplitude scaling ratio: " << wrinkleAmpRatio << std::endl;

    if (!loadEdgeOmega(workingFolder + initOmegaPath, nedges, initCWF._omega))
    {
        std::cout << "missing init edge omega file." << std::endl;
        isLoadCWF = false;
    }
    else
    {
        // convert old stored edge omega to the current order
        initCWF._omega = swapEdgeVec(baseFaces, initCWF._omega, 0);
        std::cout << "convert finished, omega size: " << initCWF._omega.rows() << std::endl;

        if (!loadVertexZvals(workingFolder + initZValsPath, initBasePos.rows(), initCWF._zvals))
        {
            std::cout << "missing init zval file, try to load amp file, and round zvals from amp and omega" << std::endl;
            if (!loadVertexAmp(workingFolder + initAmpPath, initBasePos.rows(), initCWF._amp))
            {
                std::cout << "missing init amp file: " << std::endl;
                isLoadCWF = false;
            }

            else
            {
                VectorX edgeArea, vertArea;
                edgeArea = getEdgeArea(initCWF._mesh);
                vertArea = getVertArea(initCWF._mesh);
                roundZvalsFromEdgeOmegaVertexMag(initCWF._mesh, initCWF._omega, initCWF._amp, edgeArea, vertArea, nverts, initCWF._zvals);
            }
        }
        else
        {
            initCWF._amp.setZero(initBasePos.rows());
            for (int i = 0; i < initCWF._zvals.size(); i++)
            {
                initCWF._amp(i) = std::abs(initCWF._zvals[i]);
            }
        }
    }

	// clamped vertices
	bool isLoadClamp = true;
	std::string clampedDOFFile = jval["clamped_vertices"];
	std::ifstream ifs(workingFolder + clampedDOFFile);
	std::unordered_set<int> clampedVerts = {};
	if (!ifs)
	{
		std::cout << "Missing " << clampedDOFFile << std::endl;
		isLoadClamp = false;
	}
	else
	{
		int nclamped;
		ifs >> nclamped;
		char dummy;
		ifs >> dummy;
		if (!ifs)
		{
			std::cout << "Error in " << clampedDOFFile << std::endl;
			exit(EXIT_FAILURE);
		}
		ifs.ignore(std::numeric_limits<int>::max(), '\n');
		std::cout << "num of clamped DOFs: " << nclamped << std::endl;
		for (int i = 0; i < nclamped; i++)
		{
			std::string line;
			std::getline(ifs, line);
			std::stringstream ss(line);

			int vid;
			ss >> vid;
			if (!ss || vid < 0 || vid >= initCWF._mesh.GetVertCount())
			{
				std::cout << "Error in " << clampedDOFFile << ", vid overflow!" << std::endl;
				exit(EXIT_FAILURE);
			}
			clampedVerts.insert(vid);
			std::string x; // x, y, z
			ss >> x;
			if (!ss)
			{
				std::cout << "-using rest position for clamped vertices " << vid << std::endl;
				for (int j = 0; j < 3; j++)
				{
					// should set clamped position
				}
			}
			else
			{
				for (int j = 0; j < 3; j++)
				{
					if (x[0] != '#')
					{
						// should set clamped position
					}
					ss >> x;
				}
			}
		}

		if (!ifs)
		{
			std::cout << "Error in " << clampedDOFFile << std::endl;
			exit(EXIT_FAILURE);
		}
	}
	

    // materials
    youngs = jval["Youngs_modulus"];
    poisson = jval["Poisson_ratio"];
    thickness = jval["thickness"];


    if(isLoadCWF)
    {
        initCWF._zvals = normalizeZvals(initCWF._zvals);
        optCWF = initCWF;
        decompModel.initialization(optCWF, upsampleTimes, restMesh, restWrinkledMesh, wrinkledMesh, youngs, poisson, thickness);
    }
    else
    {
        decompModel.initialization(upsampleTimes, isFixedBnd, restMesh, initCWF._mesh, restWrinkledMesh, wrinkledMesh, youngs, poisson, thickness, clampedVerts);
        decompModel.getCWF(initCWF);
        optCWF = initCWF;
    }
    std::cout << "start to subdivide" << std::endl;
    subdivideMesh(false);		// initial mesh
    subdivideMesh(true);		// opt mesh
    std::cout << "subdivide done, start to update view" << std::endl;
    updateView();
    return true;
}

void callback() {
	ImGui::PushItemWidth(100);
	float w = ImGui::GetContentRegionAvailWidth();
	float p = ImGui::GetStyle().FramePadding.x;
	if (ImGui::Button("Load", ImVec2((w - p) / 2.f, 0)))
	{
		loadProblem();
	}
	ImGui::SameLine(0, p);
	if (ImGui::Button("Save", ImVec2((w - p) / 2.f, 0)))
	{

	}
	if (ImGui::CollapsingHeader("Wrinkle Mesh Upsampling Options", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::InputInt("upsampled level", &upsampleTimes))
		{
			if (upsampleTimes >= 0)
			{
				subdivideMesh(true);		// reference mesh
				subdivideMesh(false);		// initial mesh
				updateView(true);
			}
		}
		if (ImGui::Checkbox("fix bnd", &isFixedBnd))
		{
			subdivideMesh(true);		// reference mesh
			subdivideMesh(false);		// initial mesh
			updateView(true);
		}
	}

	if (ImGui::CollapsingHeader("Visualization Options", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::DragFloat("vec ratio", &(vecratio), 0.00005, 0, 1))
		{
			updateView(false);
		}
	}
	if (ImGui::Button("CWF Projection", ImVec2(-1, 0)))
	{
        decompModel.optimizeCWF();
//		//decompModel.precomputationForPhase();
//		decompModel.optimizeBasemesh();
		decompModel.getCWF(optCWF);

		ComplexVectorX unitZvals = optCWF._zvals;
		subdivideMesh(false);
		updateView(false);
	}

	ImGui::PopItemWidth();
}

int main(int argc, char** argv)
{
	std::string inputFile = "";
	CLI::App app("Wrinkle Upsampling");
	app.add_option("input,-i,--input", inputFile, "Input model")->check(CLI::ExistingFile);

	try {
		app.parse(argc, argv);
	}
	catch (const CLI::ParseError& e) {
		return app.exit(e);
	}

	// Options
	polyscope::options::autocenterStructures = true;
	polyscope::view::windowWidth = 1024;
	polyscope::view::windowHeight = 1024;

	// Initialize polyscope
	polyscope::init();
	loadProblem(inputFile);

	polyscope::view::upDir = polyscope::view::UpDir::ZUp;

	// Add the callback
	polyscope::state::userCallback = callback;

	polyscope::options::groundPlaneHeightFactor = 0.25; // adjust the plane height

	// Show the gui
	polyscope::show();

	return 0;
}