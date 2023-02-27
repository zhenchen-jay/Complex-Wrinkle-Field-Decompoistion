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
#include "../../CWFTypes.h"
#include "../../json.hpp"
#include "../../CommonTools.h"
#include "../../LoadSaveIO.h"
#include "../../PaintGeometry.h"

#include "../../KnoppelStripePatterns.h"
#include "../../Upsampling/BaseLoop.h"
#include "../../Upsampling/ComplexLoop.h"


#include <CLI/CLI.hpp>

MatrixX triV, upV, wrinkledV, wrinkledV1, compositeWrinkledV;
Eigen::MatrixXi triF, upF, wrinkledF;
Mesh baseMesh, upMesh;

CWF baseCWF, upCWF, baseCWF1, upCWF1;

VectorX amp, omega, amp1, omega1;
VectorX upAmp, upOmega, upPhase, upAmp1, upOmega1, upPhase1;
ComplexVectorX zvals, upZvals, zvals1, upZvals1;

MatrixX faceOmega, faceOmega1;
MatrixX upFaceOmega, upFaceOmega1;

std::string workingFolder = "";
int upsampleTimes = 0;
double wrinkleAmpRatio = 1.0;
double secFrequencyRatio = 0;
double secAmpRatio = 0;

std::shared_ptr<BaseLoop> subOp;

float vecratio = 0.1;
bool isFixedBnd = false;

PaintGeometry mPaint;

int updateViewHelper(
        const MatrixX& basePos, const Eigen::MatrixXi& baseFaces,
        const MatrixX& upsampledPos, const Eigen::MatrixXi& upsampledFaces, const MatrixX& wrinkledPos,
        const VectorX& baseAmplitude, const MatrixX& baseFaceOmega,
        const VectorX& upsampledAmplitude, const VectorX& upsampledPhase, const MatrixX& upsampledFaceOmega,
        double shiftx, double shifty, int meshId = 0, bool isFirstTime = true)
{
    int curShift = 0;
    if(isFirstTime)
    {
        polyscope::registerSurfaceMesh("base mesh" + std::to_string(meshId), basePos, baseFaces);
        polyscope::getSurfaceMesh("base mesh" + std::to_string(meshId))->translate({ curShift * shiftx, shifty, 0 });
    }

    polyscope::getSurfaceMesh("base mesh" + std::to_string(meshId))->addFaceVectorQuantity("frequency field", vecratio * baseFaceOmega, polyscope::VectorType::AMBIENT);
    auto baseAmpPatterns = polyscope::getSurfaceMesh("base mesh" + std::to_string(meshId))->addVertexScalarQuantity("opt amplitude", baseAmplitude);
    baseAmpPatterns->setEnabled(true);

    curShift++;
    // phase pattern
    if(isFirstTime)
    {
        polyscope::registerSurfaceMesh("upsampled phase mesh" + std::to_string(meshId), upsampledPos, upsampledFaces);
        polyscope::getSurfaceMesh("upsampled phase mesh" + std::to_string(meshId))->translate({ curShift * shiftx, shifty, 0 });
    }

    mPaint.setNormalization(false);
    MatrixX phaseColor = mPaint.paintPhi(upsampledPhase);
    auto phasePatterns = polyscope::getSurfaceMesh("upsampled phase mesh" + std::to_string(meshId))->addVertexColorQuantity("vertex phi", phaseColor);
    phasePatterns->setEnabled(true);

    polyscope::getSurfaceMesh("upsampled phase mesh" + std::to_string(meshId))->addFaceVectorQuantity("subdivided frequency field", vecratio * upsampledFaceOmega, polyscope::VectorType::AMBIENT);
    curShift++;

    // amp pattern
    if(isFirstTime)
    {
        polyscope::registerSurfaceMesh("upsampled ampliude mesh" + std::to_string(meshId), upsampledPos, upsampledFaces);
        polyscope::getSurfaceMesh("upsampled ampliude mesh" + std::to_string(meshId))->translate({ curShift * shiftx, shifty, 0 });
    }

    auto ampPatterns = polyscope::getSurfaceMesh("upsampled ampliude mesh" + std::to_string(meshId))->addVertexScalarQuantity("vertex amplitude", upsampledAmplitude);
    ampPatterns->setEnabled(true);

    curShift++;

    // wrinkle mesh
    if(isFirstTime)
    {
        polyscope::registerSurfaceMesh("wrinkled mesh" + std::to_string(meshId), wrinkledPos, upsampledFaces);
        polyscope::getSurfaceMesh("wrinkled mesh" + std::to_string(meshId))->setSurfaceColor({ 80 / 255.0, 122 / 255.0, 91 / 255.0 });
        polyscope::getSurfaceMesh("wrinkled mesh" + std::to_string(meshId))->translate({ curShift * shiftx, shifty, 0 });
    }

    else
        polyscope::getSurfaceMesh("wrinkled mesh" + std::to_string(meshId))->updateVertexPositions(wrinkledPos);
    return curShift;
}

void updateView(bool isFirstTime = true)
{
    double shiftx = 1.5 * (triV.col(0).maxCoeff() - triV.col(0).minCoeff());
    double shifty = 1.5 * (triV.col(1).maxCoeff() - triV.col(1).minCoeff());

    int curShift = updateViewHelper(triV, triF, upV, upF, wrinkledV, amp, faceOmega, upAmp, upPhase, upFaceOmega, shiftx, 0, 0, isFirstTime);
    if(secAmpRatio && secFrequencyRatio)
    {
        updateViewHelper(triV, triF, upV, upF, wrinkledV1, amp1, faceOmega1, upAmp1, upPhase1, upFaceOmega1, shiftx, shifty, isFirstTime);

        // wrinkle mesh
        if(isFirstTime)
        {
            polyscope::registerSurfaceMesh("composite wrinkled mesh", compositeWrinkledV, upF);
            polyscope::getSurfaceMesh("composite wrinkled mesh")->setSurfaceColor({ 80 / 255.0, 122 / 255.0, 91 / 255.0 });
            polyscope::getSurfaceMesh("composite wrinkled mesh")->translate({ curShift * shiftx, 2 * shifty, 0 });
        }

        else
            polyscope::getSurfaceMesh("composite wrinkled mesh")->updateVertexPositions(compositeWrinkledV);
    }
}

void subdivideMesh()
{
    //upsampleTimes = 1;
    subOp->SetMesh(baseMesh);
    subOp->SetBndFixFlag(isFixedBnd);
    subOp->CWFSubdivide(baseCWF, upCWF, upsampleTimes);

    upMesh = upCWF._mesh;
    upMesh.GetPos(upV);
    upMesh.GetFace(upF);

    rescaleZvals(upCWF._zvals, upCWF._amp, upZvals);
    getWrinkledMesh(upV, upF, upZvals, wrinkledV, wrinkleAmpRatio, false);
    wrinkledF = upF;
    upOmega = upCWF._omega;

    faceOmega = intrinsicEdgeVec2FaceVec(omega, baseMesh);
    upFaceOmega = intrinsicEdgeVec2FaceVec(upOmega, upMesh);

    upAmp.setZero(upZvals.size());
    upPhase.setZero(upZvals.size());

    for(int i = 0; i < upZvals.size(); i++)
    {
        upAmp[i] = std::abs(upZvals[i]);
        upPhase[i] = std::arg(upZvals[i]);
    }
    std::cout << "first wave compute done!" << std::endl;

    if(secFrequencyRatio && secAmpRatio)
    {
        omega1 = secFrequencyRatio * omega;
        amp1 = secAmpRatio * amp;

        VectorX edgeArea, vertArea;
        edgeArea = getEdgeArea(baseMesh);
        vertArea = getVertArea(baseMesh);
        roundZvalsFromEdgeOmegaVertexMag(baseMesh, omega1, amp1, edgeArea, vertArea, amp1.rows(), zvals1);
        baseCWF1 = CWF(amp1, omega1, normalizeZvals(zvals1), baseMesh);

        subOp->CWFSubdivide(baseCWF1, upCWF1, upsampleTimes);
        rescaleZvals(upCWF1._zvals, upCWF1._amp, upZvals1);
        std::cout << "second subdivision done" << std::endl;

        MatrixX upN;
        igl::per_vertex_normals(upV, upF, upN);
        wrinkledV1 = upV;
        compositeWrinkledV = upV;
        for(int i = 0; i < wrinkledV.rows(); i++)
        {
            compositeWrinkledV.row(i) = upV.row(i) + upN.row(i) * (upZvals[i].real() + upZvals1[i].real()) * wrinkleAmpRatio;
            wrinkledV1.row(i) = upV.row(i) + upN.row(i) * (upZvals1[i].real()) * wrinkleAmpRatio;
        }

        upOmega1 = upCWF1._omega;

        faceOmega1 = intrinsicEdgeVec2FaceVec(omega1, baseMesh);
        upFaceOmega1 = intrinsicEdgeVec2FaceVec(upOmega1, upMesh);

        upAmp1.setZero(upZvals1.size());
        upPhase1.setZero(upZvals1.size());

        for(int i = 0; i < upZvals1.size(); i++)
        {
            upAmp1[i] = std::abs(upZvals1[i]);
            upPhase1[i] = std::arg(upZvals1[i]);
        }
        std::cout << "secondary wave compute done!" << std::endl;

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

    // convert old stored edge omega to the current order
    omega = swapEdgeVec(triF, omega, 0);
    std::cout << "convert finished, omega size: " << omega.rows() << std::endl;

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
            VectorX edgeArea, vertArea;
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
    std::cout << "start to subdivide" << std::endl;
    subOp = std::make_shared<ComplexLoop>();
    baseCWF = CWF(amp, omega, normalizeZvals(zvals), baseMesh);

    subdivideMesh();
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
        std::string savePath = igl::file_dialog_save();
        igl::writeOBJ(savePath, wrinkledV, wrinkledF);

        if(secFrequencyRatio && secAmpRatio)
        {
            savePath = igl::file_dialog_save();
            igl::writeOBJ(savePath, wrinkledV1, wrinkledF);

            savePath = igl::file_dialog_save();
            igl::writeOBJ(savePath, compositeWrinkledV, wrinkledF);
        }

	}
    if (ImGui::CollapsingHeader("Wrinkle Mesh Upsampling Options", ImGuiTreeNodeFlags_DefaultOpen))
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

    if (ImGui::CollapsingHeader("Second wave options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if(ImGui::InputDouble("frequency ratio", &secFrequencyRatio))
        {
            if(secFrequencyRatio < 0)
                secFrequencyRatio = 0;
        }
        if(ImGui::InputDouble("Amplitude ratio", &secAmpRatio))
        {
            if(secAmpRatio < 0)
                secAmpRatio = 0;
        }
        if (ImGui::Button("wrinkles on wrinkles"))
        {
            subdivideMesh();
            updateView();
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