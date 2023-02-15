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


#include <CLI/CLI.hpp>

Eigen::MatrixXd triV, upV, wrinkledV;
Eigen::MatrixXi triF, upF, wrinkledF;
Mesh baseMesh, upMesh;

Eigen::VectorXd amp, omega;
Eigen::VectorXd upAmp, upOmega, upPhase;
std::vector<std::complex<double>> zvals, upZvals;

Eigen::MatrixXd faceOmega;
Eigen::MatrixXd upFaceOmega;

std::string workingFolder = "";
int upsampleTimes = 0;
double wrinkleAmpRatio = 1.0;

std::shared_ptr<BaseLoop> subOp;

float vecratio = 0.1;
bool isFixedBnd = false;

PaintGeometry mPaint;

void updateView(bool isFirstTime = true)
{
    int curShift = 0;
    double shiftx = 1.5 * (triV.col(0).maxCoeff() - triV.col(0).minCoeff());
    if(isFirstTime)
    {
        polyscope::registerSurfaceMesh("base mesh", triV, triF);
    }

    polyscope::getSurfaceMesh("base mesh")->addFaceVectorQuantity("frequency field", vecratio * faceOmega, polyscope::VectorType::AMBIENT);

    Eigen::VectorXd baseAmplitude(zvals.size());
    for(int i = 0 ; i < zvals.size(); i++)
    {
        baseAmplitude(i) = std::abs(zvals[i]);
    }
    auto baseAmp = polyscope::getSurfaceMesh("base mesh")->addVertexScalarQuantity("opt amplitude", baseAmplitude);
    baseAmp->setEnabled(true);

    curShift++;
    // phase pattern
    if(isFirstTime)
    {
        polyscope::registerSurfaceMesh("phase mesh", upV, upF);
        polyscope::getSurfaceMesh("phase mesh")->translate({ curShift * shiftx, 0, 0 });
    }

    mPaint.setNormalization(false);
    Eigen::MatrixXd phaseColor = mPaint.paintPhi(upPhase);
    auto phasePatterns = polyscope::getSurfaceMesh("phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);
    phasePatterns->setEnabled(true);
    curShift++;

    // amp pattern
    if(isFirstTime)
    {
        polyscope::registerSurfaceMesh("upsampled ampliude and frequency mesh", upV, upF);
        polyscope::getSurfaceMesh("upsampled ampliude and frequency mesh")->translate({ curShift * shiftx, 0, 0 });
    }

    auto ampPatterns = polyscope::getSurfaceMesh("upsampled ampliude and frequency mesh")->addVertexScalarQuantity("vertex amplitude", upAmp);
    ampPatterns->setEnabled(true);
    polyscope::getSurfaceMesh("upsampled ampliude and frequency mesh")->addFaceVectorQuantity("subdivided frequency field", vecratio * upFaceOmega, polyscope::VectorType::AMBIENT);

    curShift++;

    // wrinkle mesh
    if(isFirstTime)
    {
        polyscope::registerSurfaceMesh("wrinkled mesh", wrinkledV, upF);
        polyscope::getSurfaceMesh("wrinkled mesh")->setSurfaceColor({ 80 / 255.0, 122 / 255.0, 91 / 255.0 });
        polyscope::getSurfaceMesh("wrinkled mesh")->translate({ curShift * shiftx, 0, 0 });
    }

    else
        polyscope::getSurfaceMesh("wrinkled mesh")->updateVertexPositions(wrinkledV);
}

void subdivideMesh()
{
    subOp->SetMesh(baseMesh);
    subOp->SetBndFixFlag(isFixedBnd);

    upMesh = subOp->meshSubdivide(upsampleTimes);
    subOp->CWFSubdivide(omega, zvals, upOmega, upZvals, upsampleTimes);

    upMesh.GetPos(upV);
    upMesh.GetFace(upF);

    getWrinkledMesh(upV, upF, upZvals, wrinkledV, wrinkleAmpRatio, false);
    wrinkledF = upF;

    faceOmega = intrinsicEdgeVec2FaceVec(omega, baseMesh);
    upFaceOmega = intrinsicEdgeVec2FaceVec(upOmega, upMesh);

    upAmp.setZero(upZvals.size());
    upPhase.setZero(upZvals.size());

    for(int i = 0; i < upZvals.size(); i++)
    {
        upAmp[i] = std::abs(upZvals[i]);
        upPhase[i] = std::arg(upZvals[i]);
    }
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

    std::string meshFile = jval["mesh_name"];
    upsampleTimes = jval["upsampled_times"];
    if (upsampleTimes > 2)
        upsampleTimes = 2;


    meshFile = workingFolder + meshFile;
    igl::readOBJ(meshFile, triV, triF);
    baseMesh.Populate(triV, triF);


    int nedges = baseMesh.GetEdgeCount();
    int nverts = triV.rows();

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

    if (!loadEdgeOmega(workingFolder + initOmegaPath, nedges, omega)) {
        std::cout << "missing init edge omega file." << std::endl;
        return false;
    }

    if (!loadVertexZvals(workingFolder + initZValsPath, triV.rows(), zvals))
    {
        std::cout << "missing init zval file, try to load amp file, and round zvals from amp and omega" << std::endl;
        if (!loadVertexAmp(workingFolder + initAmpPath, triV.rows(), amp))
        {
            std::cout << "missing init amp file: " << std::endl;
            return false;
        }

        else
        {
            Eigen::VectorXd edgeArea, vertArea;
            edgeArea = getEdgeArea(baseMesh);
            vertArea = getVertArea(baseMesh);
            roundZvalsFromEdgeOmegaVertexMag(baseMesh, omega, amp, edgeArea, vertArea, nverts, zvals);
        }
    }
    else
    {
        amp.setZero(triV.rows());
        for (int i = 0; i < zvals.size(); i++)
        {
            amp(i) = std::abs(zvals[i]);
        }

    }
    subOp = std::make_shared<ComplexLoop>();
    subdivideMesh();
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
    ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_None;
    if (ImGui::BeginTabBar("Wrinkle Mesh Upsampling Options", tab_bar_flags))
    {
        if (ImGui::InputInt("upsampled level", &upsampleTimes))
        {
            if (upsampleTimes >= 0)
            {
                subdivideMesh();
                updateView(true);
            }
        }
        if (ImGui::Checkbox("fix bnd", &isFixedBnd))
        {
            subdivideMesh();
            updateView(true);
        }
    }

    if (ImGui::CollapsingHeader("Visualization Options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::InputDouble("wrinkle amp scaling ratio", &wrinkleAmpRatio))
        {
            if (wrinkleAmpRatio >= 0)
            {
                getWrinkledMesh(upV, upF, upZvals, wrinkledV, wrinkleAmpRatio, false);
                updateView();
            }
        }
        if (ImGui::DragFloat("vec ratio", &(vecratio), 0.00005, 0, 1))
        {
            updateView(false);
        }
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
    loadProblem(inputFile);

	// Options
	polyscope::options::autocenterStructures = true;
	polyscope::view::windowWidth = 1024;
	polyscope::view::windowHeight = 1024;

	// Initialize polyscope
	polyscope::init();

	polyscope::view::upDir = polyscope::view::UpDir::ZUp;

	// Add the callback
	polyscope::state::userCallback = callback;

	polyscope::options::groundPlaneHeightFactor = 0.25; // adjust the plane height

	
	// Show the gui
	polyscope::show();


	return 0;
}