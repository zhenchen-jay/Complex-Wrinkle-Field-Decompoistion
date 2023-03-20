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
#include <igl/remesh_along_isoline.h>
#include <igl/principal_curvature.h>
#include <igl/isolines.h>
#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/curve_network.h"
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
#include "../../ExtractIsoline.h"


#include <CLI/CLI.hpp>

MatrixX wrinkledPos;
Eigen::MatrixXi wrinkledFaces;
Mesh wrinkledMesh;
VectorX minCurvature, maxCurvature, meanCurvature;

std::string workingFolder = "";
float vecratio = 0.01;
bool isFixedBnd = true;
double ampMax = 1;
bool isLoadCWF = true;

double youngs = 1e7;
double thickness = 1e-4;
double poisson = 0.44;

PaintGeometry mPaint;
CWFDecomposition decompModel;

Eigen::MatrixXi wrinkledFaceNeighbors;

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

void updateView(bool isFirstTime = true, bool drawIsolines = false)
{
	double shiftx = 0;
	double shifty = 0;

	polyscope::registerSurfaceMesh("reference wrinkled mesh", wrinkledPos, wrinkledFaces);
	polyscope::getSurfaceMesh("reference wrinkled mesh")->setSurfaceColor({ 80 / 255.0, 122 / 255.0, 91 / 255.0 });
	polyscope::getSurfaceMesh("reference wrinkled mesh")->translate({ 2 * shiftx, 2 * shifty, 0 });

    if(isFirstTime)
    {
        MatrixX maxDir, minDir;
        igl::principal_curvature(wrinkledPos, wrinkledFaces, minDir, maxDir, minCurvature, maxCurvature);
        meanCurvature = (minCurvature + maxCurvature) / 2;
    }

    auto minCurvPattens = polyscope::getSurfaceMesh("reference wrinkled mesh")->addVertexScalarQuantity("min curvature", minCurvature);
    minCurvPattens->setColorMap("coolwarm");
    auto maxCurvPattens = polyscope::getSurfaceMesh("reference wrinkled mesh")->addVertexScalarQuantity("max curvature", maxCurvature);
    maxCurvPattens->setColorMap("coolwarm");
    auto meanCurvPattens = polyscope::getSurfaceMesh("reference wrinkled mesh")->addVertexScalarQuantity("mean curvature", meanCurvature);
    meanCurvPattens->setColorMap("coolwarm");

    if(drawIsolines)
    {
        Eigen::MatrixXi isoE;
        MatrixX isoV;
//        zeroLevelSetIsopoints(wrinkledMesh, meanCurvature, isoV);
        extractIsoline(wrinkledPos, wrinkledFaces, wrinkledFaceNeighbors, meanCurvature, 0, isoV, isoE);
//        igl::isolines(wrinkledPos, wrinkledFaces, meanCurvature, 10, isoV, isoE);
//        polyscope::registerPointCloud("0-isopoints", isoV);
//        polyscope::getPointCloud("0-isopoints")->translate({ curShift * shiftx, 2 * shifty, 0 });

        polyscope::registerCurveNetwork("0-isolines", isoV, isoE);
        polyscope::getCurveNetwork("0-isolines")->translate({ 2 * shiftx, 2 * shifty, 0 });
    }

//	std::cout << "dist: " << (initWrinkledPos - wrinkledPos).norm() << ", after optimization: " << (optWrinkledPos - wrinkledPos).norm() << std::endl;
}



bool loadProblem(std::string loadFileName = "")
{
    if(loadFileName == "")
        loadFileName = igl::file_dialog_open();

    if(!igl::readOBJ(loadFileName, wrinkledPos, wrinkledFaces))
    {
        std::cerr << "Failed to load wrinkled mesh!" << std::endl;
        exit(EXIT_FAILURE);
    }
    wrinkledMesh.Populate(wrinkledPos, wrinkledFaces);
    wrinkledFaceNeighbors.resize(wrinkledFaces.rows(), 3);
    for(int i = 0; i < wrinkledFaces.rows(); i++)
    {
        for(int j = 0; j < 3; j++)
        {
            int eid = wrinkledMesh.GetFaceEdges(i)[(j + 1) % 3];
            if(wrinkledMesh.GetEdgeFaces(eid).size() == 1)
                wrinkledFaceNeighbors(i, j) = -1;
            else
                wrinkledFaceNeighbors(i, j) = wrinkledMesh.GetEdgeFaces(eid)[0] != i ? wrinkledMesh.GetEdgeFaces(eid)[0] : wrinkledMesh.GetEdgeFaces(eid)[1];
        }
    }

    std::cout << "load file in: " << loadFileName << std::endl;
    std::string filePath = loadFileName;
    std::replace(filePath.begin(), filePath.end(), '\\', '/'); // handle the backslash issue for windows
    int id = filePath.rfind("/");
    workingFolder = filePath.substr(0, id + 1);
    std::cout << "working folder: " << workingFolder << std::endl;
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
    if (ImGui::Button("Extract 0-level set", ImVec2(-1, 0)))
    {
        updateView(false, true);
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